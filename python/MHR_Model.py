import numpy as np
from scipy.integrate import odeint
from scipy import signal
from simple_pid import PID

class MHR_Model:
    def __init__(self, observed_data, time_points):
        """
        Modelo Hierárquico Ressonante (MHR) para análise e predição de sistemas complexos.

        Parameters:
        -----------
        observed_data : ndarray
            Matriz de dados observados [n_levels, n_times] em diferentes níveis hierárquicos.
        time_points : ndarray
            Vetor de tempos correspondentes aos dados observados.
        """
        self.data = np.array(observed_data, dtype=np.float64)
        self.time = np.array(time_points, dtype=np.float64)
        self.levels = self.data.shape[0]
        self.params = None  # Parâmetros dinâmicos do modelo
        self.systems = None  # Funções de transferência estimadas
        self.pids = None    # Controladores PID por nível
        self.precision = 1e-16  # Precisão mínima para float64
        self._build_model_from_data()

    def _build_model_from_data(self):
        """
        Constrói o modelo dinâmico a partir dos dados observados, estimando interações e ressonâncias.
        Usa regressão para estimar coeficientes de um modelo de segunda ordem por nível.
        """
        dt = self.time[1] - self.time[0]
        dominant_freqs = []

        # Estimar frequências dominantes via FFT (auxiliar)
        for level in range(self.levels):
            fft_data = np.fft.fft(self.data[level])
            freqs = np.fft.fftfreq(len(self.time), d=dt)
            peak_idx = np.argmax(np.abs(fft_data))
            dominant_freq = np.abs(freqs[peak_idx]) if freqs[peak_idx] != 0 else 1e-3
            dominant_freqs.append(dominant_freq)

        # Estimar coeficientes dinâmicos por nível (substitui signal.estimate_transfer_function)
        self.params = np.zeros((self.levels, 4), dtype=np.float64)  # [self, prev, next, non-linear]
        for i in range(self.levels):
            # Simples modelo de segunda ordem: d^2x/dt^2 + a*dx/dt + b*x = 0
            dx_dt = np.gradient(self.data[i], dt)
            d2x_dt2 = np.gradient(dx_dt, dt)
            A = np.vstack([dx_dt, self.data[i]]).T
            b = -d2x_dt2
            coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)  # Regressão linear
            self.params[i, 0] = coeffs[1]  # Termo proporcional (b)
            self.params[i, 1] = 1e-2 if i > 0 else 0  # Acoplamento anterior
            self.params[i, 2] = 1e-2 if i < self.levels - 1 else 0  # Acoplamento seguinte
            self.params[i, 3] = 1e-3  # Termo não linear inicial

        # Inicializar PIDs com sintonia dinâmica
        self.pids = []
        for level in range(self.levels):
            scale_factor = np.max(np.abs(self.data[level])) or 1.0  # Evitar divisão por zero
            kp = 1.0 / (dominant_freqs[level] * scale_factor)
            ki = 0.1 / (dominant_freqs[level] * scale_factor)
            kd = 0.01 * dominant_freqs[level]
            pid = PID(kp, ki, kd, setpoint=0)
            pid.output_limits = (-scale_factor * 1e3, scale_factor * 1e3)
            pid.sample_time = dt / 10  # Amostragem adaptada ao intervalo de tempo
            self.pids.append(pid)

    def dynamics(self, state, t, controls):
        """
        Define a dinâmica do sistema com interações lineares e não lineares.

        Parameters:
        -----------
        state : ndarray
            Estado atual do sistema em cada nível.
        t : float
            Tempo atual (necessário para odeint).
        controls : ndarray
            Controles aplicados pelo PID.

        Returns:
        --------
        dA_dt : ndarray
            Derivadas do estado em cada nível.
        """
        dA_dt = np.zeros(self.levels, dtype=np.float64)
        for i in range(self.levels):
            interaction = (
                self.params[i, 0] * state[i] +  # Termo linear próprio
                (self.params[i, 1] * state[i - 1] if i > 0 else 0) +  # Acoplamento anterior
                (self.params[i, 2] * state[i + 1] if i < self.levels - 1 else 0) +  # Acoplamento seguinte
                self.params[i, 3] * state[i]**2  # Termo não linear (quadrático)
            )
            dA_dt[i] = interaction + controls[i]

            # Limites dinâmicos para estabilidade numérica
            min_val = np.max(np.abs(self.data[i])) * self.precision
            max_val = np.max(np.abs(self.data[i])) * 1e3
            if np.abs(dA_dt[i]) < min_val:
                dA_dt[i] = np.sign(dA_dt[i]) * min_val
            elif np.abs(dA_dt[i]) > max_val:
                dA_dt[i] = np.sign(dA_dt[i]) * max_val

        return dA_dt

    def fit_to_data(self, iterations=10):
        """
        Ajusta o modelo aos dados observados usando PID adaptativo.

        Parameters:
        -----------
        iterations : int
            Número de iterações de ajuste.
        """
        simulated = self.data[:, 0].copy()
        dt = self.time[1] - self.time[0]
        atol = [self.precision * (10 ** (i * 3)) for i in range(self.levels)]

        for _ in range(iterations):
            controls = np.zeros(self.levels, dtype=np.float64)
            for t in range(1, len(self.time)):
                for i in range(self.levels):
                    error = self.data[i, t] - simulated[i]
                    controls[i] = self.pids[i](error)
                simulated = odeint(self.dynamics, simulated,
                                 [self.time[t - 1], self.time[t]],
                                 args=(controls,),
                                 atol=atol[i], rtol=self.precision)[-1]

            # Atualizar parâmetros com ajuste seguro
            for i in range(self.levels):
                delta = np.clip(np.mean(controls[i]) * 1e-3,
                              -np.abs(self.params[i, 0]) * 1e-3,
                              np.abs(self.params[i, 0]) * 1e-3)
                self.params[i, 0] = np.round(self.params[i, 0] + delta, decimals=16)

    def predict(self, future_time_points):
        """
        Faz previsões futuras usando o modelo ajustado.

        Parameters:
        -----------
        future_time_points : ndarray
            Pontos de tempo para os quais prever o comportamento.

        Returns:
        --------
        simulated : ndarray
            Estados previstos em cada nível.
        """
        controls = np.zeros(self.levels, dtype=np.float64)
        atol = [self.precision * (10 ** (i * 3)) for i in range(self.levels)]
        simulated = odeint(self.dynamics, self.data[:, -1], future_time_points,
                         args=(controls,), atol=atol, rtol=self.precision)
        return simulated

    def emergent_behavior(self, simulated_data):
        """
        Calcula o comportamento emergente como média dos níveis.

        Parameters:
        -----------
        simulated_data : ndarray
            Dados simulados para análise.

        Returns:
        --------
        emergent : ndarray
            Comportamento emergente por nível.
        """
        return np.round(np.sum(simulated_data, axis=1) / self.levels, decimals=16)

    def validate(self):
        """
        Calcula o erro médio quadrático (MSE) entre dados observados e simulados.

        Returns:
        --------
        mse : ndarray
            MSE por nível hierárquico.
        """
        simulated = self.predict(self.time)
        mse = np.mean((self.data - simulated.T) ** 2, axis=1)
        return mse

# Exemplo prático de uso com testes de escalas extremas
if __name__ == "__main__":
    # Dados de teste com escalas extremas
    time = np.linspace(0, 10, 1000, dtype=np.float64)
    observed_data = np.array([
        1e-16 * np.sin(0.5 * time) + np.random.normal(0, 1e-18, len(time)),  # Muito pequeno
        1e6 * np.sin(0.3 * time) + np.random.normal(0, 1e3, len(time)),      # Intermediário
        1e308 * np.sin(0.1 * time) + np.random.normal(0, 1e305, len(time))   # Muito grande
    ], dtype=np.float64)

    # Inicializar e ajustar o modelo
    mhr = MHR_Model(observed_data, time)
    mhr.fit_to_data(iterations=5)

    # Prever comportamento futuro
    future_time = np.linspace(10, 20, 1000, dtype=np.float64)
    predicted = mhr.predict(future_time)
    emergent = mhr.emergent_behavior(predicted)
    mse = mhr.validate()

    # Exibir resultados
    print("Previsões futuras (primeiros 5 pontos):", predicted[:5])
    print("Comportamento emergente (primeiros 5 pontos):", emergent[:5])
    print("Erro médio quadrático por nível:", mse)