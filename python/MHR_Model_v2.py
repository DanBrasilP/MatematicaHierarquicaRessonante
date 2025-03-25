import numpy as np
import json
import os
from scipy.integrate import odeint
from scipy.fft import fft
from scipy.stats import entropy
from simple_pid import PID
from sympy import symbols, Function, laplace_transform, simplify
from sympy.abc import t, s
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

LOG_GLOBAL = "log_mhr_global.json"

class MHR_Model:
    """
    Modelo Holográfico Ressonante (MHR) - Implementação computacional da Teoria das Tríades Ressonantes (TTR).

    Esta classe não representa um sistema físico tradicional com massa, forças ou leis fixas.
    Em vez disso, o MHR modela uma estrutura ressonante vetorial (tríade) que interage com
    um meio adaptativo, vivo e polarizável.

    O aprendizado não visa prever dados diretamente, mas observar como o meio responde
    à coerência interna da tríade. A dinâmica é orientada por auto-organização,
    coerência fásica e feedback adaptativo do meio. O PID atua como um agente epistêmico
    que automatiza o processo de investigação científica da TTR.
    """
    
    def __init__(self, observed_data, time_points, energy=None, temperature=None,
                 mass=None, quantized_levels=None, initial_params=None,
                 energy_axis=None, inherited_coupling=None, inherit_params=True,
                 use_pid=True, use_quantum=True, variable_constants=False):        
        """
        Inicializa o modelo MHR (Matemática Hierárquica Ressonante), configurando níveis observacionais,
        parâmetros físicos e controle adaptativo.

        Parâmetros:
            observed_data: matriz [n_levels x t] com os dados observados por nível.
            time_points: vetor de tempo correspondente aos dados observados.
            energy: vetor opcional de energia por nível.
            temperature: vetor opcional de temperatura por nível.
            mass: vetor opcional de massa por nível.
            quantized_levels: lista de listas com níveis quantizados permitidos por nível.
            initial_params: matriz opcional de parâmetros para iniciar o modelo.
            energy_axis: vetor opcional com distribuição de energia contínua.
            inherited_coupling: matriz de acoplamento fornecida externamente.
            inherit_params: se True, tenta carregar os parâmetros salvos de execução anterior.
            use_pid: ativa ou desativa o controlador PID por nível.
            use_quantum: ativa ou desativa o modo de cálculo quântico.
            variable_constants: permite variação dinâmica das constantes físicas.
        """

        self.data = np.array(observed_data, dtype=np.float64)
        self.time = np.array(time_points, dtype=np.float64)
        self.levels = self.data.shape[0]
        self.mass = mass if mass is not None else np.ones(self.levels)
        self.energy = energy if energy is not None else np.ones(self.levels)
        self.temperature = temperature if temperature is not None else np.ones(self.levels)
        self.quantized_levels = quantized_levels
        self.energy_axis = np.array(energy_axis, dtype=np.float64) if energy_axis is not None else None
        self.use_pid = use_pid
        self.use_quantum = use_quantum
        self.variable_constants = variable_constants

        self.medium_response = np.ones(self.levels)

        self.constants = {
            "h": np.full(self.levels, 6.62607015e-34),
            "c": np.full(self.levels, 299792458.0),
            "kb": np.full(self.levels, 1.380649e-23),
            "G": np.full(self.levels, 6.67430e-11)
        }

        self.precision = 1e-16
        self.dt = self.time[1] - self.time[0]
        self.params = None

        if inherit_params:
            self.params = self._load_global_params()

        if self.params is None or initial_params is not None:
            self.params = np.array(initial_params, dtype=np.float64) if initial_params is not None else None
            self._build_model_from_data()
        else:
            self._adjust_for_new_levels()

        self._initialize_coupling(inherited_coupling)

        if self.use_pid:
            self._initialize_pids()

    def _load_global_params(self):
        """
        Carrega os parâmetros do log global salvo (última execução do modelo), se existente.

        Retorna:
            matriz numpy com os últimos parâmetros aprendidos, ou None se não existir.
        """

        if os.path.exists(LOG_GLOBAL):
            with open(LOG_GLOBAL, "r") as f:
                try:
                    logs = json.load(f)
                    return np.array(logs[-1]["parametros"], dtype=np.float64)
                except:
                    return None
        return None

    def _build_model_from_data(self):
        """
        Constrói a matriz de parâmetros iniciais do modelo com base nos dados observados,
        utilizando frequência dominante e escala relativa de amplitude por nível.

        Essa função é usada quando não existem parâmetros anteriores salvos.
        """

        self.params = np.zeros((self.levels, 4))

        for i in range(self.levels):
            dominant_freq = 1.0 / self.dt
            scale_factor = np.max(np.abs(self.data[i])) or 1.0
            self.params[i, 0] = dominant_freq
            self.params[i, 1] = 0.1 * dominant_freq
            self.params[i, 2] = 0.01 * dominant_freq
            self.params[i, 3] = scale_factor * 0.001 

    def _adjust_for_new_levels(self):
        """
        Adiciona automaticamente novos níveis à matriz de parâmetros caso os dados tenham
        mais níveis que os parâmetros carregados. Os novos parâmetros são inferidos
        pela média dos últimos níveis existentes ou iniciados aleatoriamente.
        """

        while self.params.shape[0] < self.levels:
            new_idx = self.params.shape[0]
            new_params = np.mean(self.params[-2:], axis=0) if new_idx >= 2 else np.random.uniform(0.1, 0.5, self.params.shape[1])
            self.params = np.vstack([self.params, new_params])

    def _initialize_pids(self):
        """
        Inicializa os controladores PID para cada nível com base nos parâmetros
        de frequência e amplitude do sinal. As saídas dos PIDs são usadas para
        modular a resposta do sistema durante o ajuste ao dado real.
        """

        self.pids = []
        for level in range(self.levels):
            freq = self.params[level, 0]
            amp = np.max(np.abs(self.data[level])) or 1.0
            freq = freq if freq > self.precision else 1e-3
            amp = amp if amp > self.precision else 1.0
            kp = 1.0 / (freq * amp)
            ki = 0.1 / (freq * amp)
            kd = 0.01 * freq
            pid = PID(kp, ki, kd, setpoint=0)
            pid.output_limits = (-amp * 1e3, amp * 1e3)
            pid.sample_time = self.dt
            self.pids.append(pid)

    def _initialize_coupling(self, inherited_coupling=None):
        """
        Inicializa a matriz de acoplamento entre os níveis do sistema,
        combinando uma matriz base com acoplamento por fase e por frequência.
        A matriz resultante expressa a intensidade de interação entre os níveis.
        """

        base_coupling = np.array(inherited_coupling, dtype=np.float64) if inherited_coupling is not None else np.random.uniform(-0.1, 0.1, (self.levels, self.levels))
        if base_coupling.shape != (self.levels, self.levels):
            base_coupling = np.random.uniform(-0.1, 0.1, (self.levels, self.levels))
        np.fill_diagonal(base_coupling, 1)
        self.phase_matrix = self._compute_phase_coupling()
        self.freq_matrix = self._compute_freq_coupling()
        self.coupling_matrix = (base_coupling + self.phase_matrix + self.freq_matrix) / 3

    def _compute_freq_coupling(self):
        """
        Calcula a matriz de acoplamento baseada na diferença das frequências dominantes
        dos níveis (análise de Fourier). Quanto menor a diferença, maior a coerência entre níveis.

        Retorna:
            matriz [n_levels x n_levels] com valores normalizados de coerência por frequência.
        """

        n = self.levels
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    fi = np.argmax(np.abs(fft(self.data[i])))
                    fj = np.argmax(np.abs(fft(self.data[j])))
                    diff = abs(fi - fj)
                    C[i, j] = 1 / (1 + diff)
        np.fill_diagonal(C, 1)
        return C

    def _compute_phase_coupling(self):
        """
        Calcula a matriz de acoplamento baseada na correlação de fase temporal entre níveis.
        Utiliza correlação cruzada para estimar defasagens relativas e converter em valores de coerência.

        Retorna:
            matriz [n_levels x n_levels] com valores normalizados de coerência por fase.
        """

        n = self.levels
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    cross = np.correlate(self.data[i], self.data[j], mode='full')
                    lag = np.argmax(cross) - len(self.data[i]) + 1
                    C[i, j] = 1 / (1 + abs(lag))
        np.fill_diagonal(C, 1)
        return C

    def _update_constants_from_medium(self, i, state):
        """
        Atualiza dinamicamente as constantes físicas (h, c, kb, G) com base na influência
        média da atividade no nível analisado. Simula uma resposta do meio a estados oscilatórios intensos.
        """

        if self.variable_constants:
            influence = np.tanh(np.linalg.norm(state) / (np.max(np.abs(self.data[i])) + 1e-9))
            self.constants["h"][i] *= 1 + 0.01 * influence
            self.constants["c"][i] *= 1 + 0.01 * influence
            self.constants["kb"][i] *= 1 + 0.01 * influence
            self.constants["G"][i] *= 1 + 0.01 * influence

    def _energy_dynamic(self, i, state):
        """
        Calcula a energia dinâmica do nível i com base em sua amplitude e
        coerência vetorial com vizinhos, sem depender diretamente da massa.
        A energia reflete a contribuição ressonante local da tríade.
        """

        center = state[i]
        left = state[i - 1] if i > 0 else 0
        right = state[i + 1] if i < self.levels - 1 else 0

        # Coerência circular com vizinhos
        coherence_left = np.cos(center - left)
        coherence_right = np.cos(center - right)
        coherence = (coherence_left + coherence_right) / 2

        # Energia dinâmica vetorial baseada em coerência e amplitude
        amplitude = np.abs(center)
        energy = (amplitude**2) * (1 + coherence)

        # Normalização opcional para manter energia em faixa estável
        energy = np.clip(energy, 0.0, 1e6)

        return energy

    def _quantize(self, val, levels):
        """
        Dada uma lista de níveis quantizados permitidos, retorna o mais próximo do valor fornecido.

        Parâmetros:
            val: valor contínuo.
            levels: lista de níveis quantizados.

        Retorna:
            valor quantizado mais próximo.
        """

        diffs = np.abs(np.array(levels) - val)
        return levels[np.argmin(diffs)]

    def dynamics(self, state, t, controls=None):
        """
        Dinâmica ressonante baseada em osciladores vetoriais ortogonais (X, Y, Z),
        incluindo aceleração (segunda ordem), amortecimento, coerência circular e
        resposta do meio adaptativo. Não depende diretamente da massa.
        """

        # Vetor para derivadas de segunda ordem (aceleração)
        d2A_dt2 = np.zeros(self.levels, dtype=np.float64)

        # Coleta derivadas de primeira ordem (velocidade) se disponíveis no estado
        if len(state) == self.levels * 2:
            A = state[:self.levels]
            dA_dt = state[self.levels:]
        else:
            A = state
            dA_dt = np.zeros(self.levels)

        for i in range(self.levels):
            omega_0 = self.params[i, 0]       # frequência natural base
            gamma = self.params[i, 1]         # amortecimento
            kappa = self.params[i, 2]         # acoplamento fásico
            amp = self.params[i, 3]           # amplitude de excitação
            control = controls[i] if controls is not None else 0

            # Vizinhos para cálculo de coerência vetorial circular
            left = A[i - 1] if i > 0 else 0
            right = A[i + 1] if i < self.levels - 1 else 0

            # Coerência circular média com os vizinhos
            coherence = (np.cos(A[i] - left) + np.cos(A[i] - right)) / 2

            # Modulação adaptativa da frequência pelo meio responsivo
            omega_eff = omega_0 * (1 + 0.1 * coherence) * self.medium_response[i]

            # Força restauradora + não-linearidade cúbica + acoplamento coerente
            restoring_force = -omega_eff**2 * A[i]
            nonlinearity = amp * np.sin(A[i])**3
            coherence_force = kappa * coherence

            # Segunda derivada da oscilação (aceleração)
            d2A_dt2[i] = restoring_force - gamma * dA_dt[i] + nonlinearity + coherence_force + control

        # Combina derivadas para integrador de sistemas de segunda ordem
        return np.concatenate((dA_dt, d2A_dt2))

    def fit_to_data(self, iterations=100, learning_rate=0.01, tolerance=1e-6, stagnation_limit=10):
        """
        O MHR investiga o meio vivo através da tríade vetorial ressonante.
        O PID atua como operador científico, ajustando os graus de liberdade
        (frequência e amplitude) para observar como o meio responde à coerência.
        Executa uma jornada de investigação do meio. O PID atua como um cientista automatizado,
        ajustando os graus de liberdade da tríade (frequência, amplitude) para observar
        como o meio responde em termos de coerência vetorial.
        Não visa reduzir erro apenas, mas maximizar o acoplamento ressonante com o meio vivo.
        """

        best_error = float('inf')
        stagnation_counter = 0
        log_trace = {
            "mse": [],
            "coerencia_media": [],
            "resposta_meio": []
        }

        for epoch in range(iterations):
            simulated = self.predict(future_steps=len(self.time))
            error = self.data - simulated
            mse = np.mean(error ** 2)

            # Coletar coerência vetorial média
            coerencias = [self.compute_vector_coherence(i) for i in range(self.levels)]
            coerencia_media = np.mean(coerencias)

            # Atualizar log da jornada
            log_trace["mse"].append(mse)
            log_trace["coerencia_media"].append(coerencia_media)
            log_trace["resposta_meio"].append(self.medium_response.tolist())

            # Critério de parada por estagnação
            if abs(best_error - mse) < tolerance:
                stagnation_counter += 1
                if stagnation_counter >= stagnation_limit:
                    print(f"🔹 Parada por estagnação no epoch {epoch}")
                    break
            else:
                stagnation_counter = 0
                best_error = mse

            # PID atua como cientista: adapta e observa
            for i in range(self.levels):
                coherence = coerencias[i]
                grad = np.sign(error[i].mean())

                # Ajuste: experimento de campo
                self.params[i, 0] += learning_rate * grad * 0.1 * coherence  # frequência
                self.params[i, 3] -= learning_rate * grad * (1 + coherence)  # amplitude

            # O meio responde às mudanças
            self.adjust_medium_properties()
            self.normalize_medium_response()

        # Registrar log da jornada de aprendizagem do meio
        self._save_global_log()
        self._log_pid_trace(log_trace)


    def predict(self, future_steps=100):
        """
        Projeta a evolução futura da tríade com base no estado atual e no histórico de interação com o meio.
        O objetivo não é prever dados observacionais, mas modelar a continuidade da coerência
        em um ambiente adaptativo.
        """


        # Estado inicial: posição + velocidade (segunda ordem)
        A0 = self.data[:, -1]
        V0 = np.gradient(self.data[:, -1], self.dt)
        initial_state = np.concatenate((A0, V0))

        future_time_points = np.linspace(
            self.time[-1], self.time[-1] + self.dt * future_steps, future_steps
        )

        controls = np.zeros(self.levels)

        predicted = odeint(
            self.dynamics,
            initial_state,
            future_time_points,
            args=(controls,),
            atol=1e-6,
            rtol=1e-6
        )

        # Retorna apenas as posições simuladas
        return predicted[:, :self.levels].T

    def emergent_behavior(self, simulated_data):
        """
        Calcula o comportamento emergente do sistema como a média vetorial dos níveis,
        representando a unificação ressonante das oscilações.

        Parâmetros:
            simulated_data: matriz simulada [n_levels x t].

        Retorna:
            vetor com a média por tempo (sinal emergente).
        """

        return np.mean(simulated_data, axis=0) if simulated_data.ndim > 1 else simulated_data

    def validate(self, simulated=None):
        """
        Avalia a qualidade da interação tríade ↔ meio ao longo do experimento.
        Não valida somente a previsão de dados em si, mas o grau de coerência, 
        resposta adaptativa, estabilidade lagrangiana e rupturas na ressonância.
        Útil para verificar o nível de organização que emergiu durante a simulação.

        Valida a simulação comparando com os dados reais observados. Retorna:
        - Erro médio quadrático (por nível)
        - Entropia espectral
        - Lagrangiana média
        - Resposta do meio
        - Ruptura de coerência (via FFT)
        """

        if simulated is None:
            simulated = self.predict(future_steps=len(self.time))

        mse = np.mean((self.data - simulated) ** 2, axis=1)

        entropias = [
            entropy(np.abs(np.fft.fft(self.data[i]))[:len(self.data[i]) // 2])
            for i in range(self.levels)
        ]

        lagrangiana = self.infer_lagrangian()
        resposta_meio = self.adjust_medium_properties()

        rupturas = [
            np.sum(np.abs(fft(self.data[i]) - fft(simulated[i])))
            for i in range(self.levels)
        ]

        self._save_global_log()

        return {
            "mse": mse,
            "entropia": entropias,
            "lagrangiana": lagrangiana,
            "resposta_meio": resposta_meio,
            "rupturas_coerencia": rupturas
        }

    def compute_tetrahedral_unity(self):
        """
        Mede a unidade tetraédrica ressonante entre três níveis base e um emergente.
        A unidade é interpretada como coerência circular entre os três vetores ortogonais
        e o quarto ponto emergente (ápice da tríade).
        """

        if self.levels < 4:
            raise ValueError("É necessário pelo menos 4 níveis para formar uma unidade tetraédrica.")

        unidades = []

        for i in range(self.levels - 3):
            base1 = self.data[i]
            base2 = self.data[i + 1]
            base3 = self.data[i + 2]
            apex = self.data[i + 3]

            # Coerência entre os 3 vetores base (X, Y, Z)
            base_coherence = np.mean([
                np.cos(base1 - base2),
                np.cos(base1 - base3),
                np.cos(base2 - base3)
            ])

            # Coerência do ápice com o plano base
            apex_coherence = np.mean([
                np.cos(apex - base1),
                np.cos(apex - base2),
                np.cos(apex - base3)
            ])

            unidade = (base_coherence + apex_coherence) / 2
            unidades.append(unidade)

        return unidades

    def _save_global_log(self):
        """
        Salva o estado atual da interação com o meio (parâmetros + resposta adaptativa),
        preservando o histórico de aprendizado entre execuções.

        Este log representa a memória epistêmica da tríade com o universo modelado.
        """
        log_path = "log_mhr.json"
        log_data = {
            "params": self.params.tolist(),
            "medium_response": self.medium_response.tolist()
        }

        try:
            with open(log_path, "w") as f:
                json.dump(log_data, f, indent=4)
            print("🔹 Log de aprendizado salvo com sucesso.")
        except Exception as e:
            print(f"⚠️ Erro ao salvar log: {e}")
    
    def load_global_log(self):
        """
        Carrega os parâmetros e a resposta do meio salvos anteriormente,
        permitindo continuidade do aprendizado ressonante entre execuções.
        """
        log_path = "log_mhr.json"

        if not os.path.exists(log_path):
            print("⚠️ Nenhum log global encontrado. Usando parâmetros atuais.")
            return

        try:
            with open(log_path, "r") as f:
                log_data = json.load(f)

            if "params" in log_data and "medium_response" in log_data:
                self.params = np.array(log_data["params"])
                self.medium_response = np.array(log_data["medium_response"])
                print("🔹 Log de aprendizado carregado com sucesso.")
            else:
                print("⚠️ Log encontrado, mas incompleto. Parâmetros não atualizados.")

        except Exception as e:
            print(f"⚠️ Erro ao carregar log: {e}")
    
    def _log_pid_trace(self, trace):
        """
        Registra a jornada do PID como agente de investigação científica.
        Captura como a tríade se organizou em relação ao meio durante o processo de escuta e ajuste.
        """
        path = "pid_learning_trace.json"

        try:
            with open(path, "w") as f:
                json.dump(trace, f, indent=4)
            print("📘 Jornada de aprendizagem do meio registrada.")
        except Exception as e:
            print(f"⚠️ Falha ao registrar trace do PID: {e}")
    
    def report_learning_journey(self, path="pid_learning_trace.json"):
        """
        Gera um relatório visual e analítico da jornada de aprendizagem conduzida pelo PID,
        mostrando como o modelo investigou o meio ao longo das iterações.
        """
        try:
            with open(path, "r") as f:
                trace = json.load(f)
        except Exception as e:
            print(f"⚠️ Falha ao carregar trace de aprendizagem: {e}")
            return

        iterations = len(trace["mse"])
        x = np.arange(iterations)

        fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

        # MSE ao longo do tempo
        axs[0].plot(x, trace["mse"], label="MSE", color="tab:red")
        axs[0].set_ylabel("Erro Médio Quadrático")
        axs[0].set_title("📉 Erro durante a investigação do meio")
        axs[0].legend()

        # Coerência média
        axs[1].plot(x, trace["coerencia_media"], label="Coerência Vetorial Média", color="tab:blue")
        axs[1].set_ylabel("Coerência")
        axs[1].set_title("🧭 Coerência vetorial ao longo das iterações")
        axs[1].legend()

        # Resposta do meio (curva média por iteração)
        mean_response = [np.mean(r) for r in trace["resposta_meio"]]
        axs[2].plot(x, mean_response, label="Resposta Média do Meio", color="tab:green")
        axs[2].set_xlabel("Iteração")
        axs[2].set_ylabel("Resposta do Meio")
        axs[2].set_title("🌐 Adaptação do meio à tríade")
        axs[2].legend()

        plt.tight_layout()
        plt.show()

    def infer_lagrangian(self):
        """
        Calcula a Lagrangiana média para cada nível:
        L = T - V, com:
        - T (energia cinética): baseada em derivada temporal da fase (velocidade vetorial)
        - V (energia potencial): associada à deformação angular (fase), sem massa
        """

        L = []
        dt = self.dt

        for i in range(self.levels):
            phase = self.data[i]
            dphase_dt = np.gradient(phase, dt)

            # Energia cinética como quadrado da velocidade vetorial
            kinetic = 0.5 * dphase_dt**2

            # Potencial como tensão fásica (quanto mais afastado do equilíbrio circular)
            potential = 0.5 * np.sin(phase)**2

            lagrangian = kinetic - potential
            L.append(np.mean(lagrangian))

        return L
    
    def compute_hamiltonian(self):
        """
        Calcula a Hamiltoniana clássica da tríade, considerando:
        H = T + V, com:
        - T = 0.5 * (dphi/dt)^2  (cinética fásica)
        - V = 0.5 * sin(phi)^2   (potencial angular)
        Retorna a energia total média por nível.
        """

        dt = self.dt
        H = []

        for i in range(self.levels):
            phi = self.data[i]
            dphi_dt = np.gradient(phi, dt)

            T = 0.5 * dphi_dt**2
            V = 0.5 * np.sin(phi)**2

            H_total = T + V
            H.append(np.mean(H_total))

        return H
    
    def check_energy_conservation(self):
        """
        Verifica se há conservação da energia total ao longo do tempo para cada nível.
        Calcula a variação percentual entre os extremos.
        """

        dt = self.dt
        energy_fluctuation = []

        for i in range(self.levels):
            phi = self.data[i]
            dphi_dt = np.gradient(phi, dt)

            T = 0.5 * dphi_dt**2
            V = 0.5 * np.sin(phi)**2
            E = T + V

            delta = (np.max(E) - np.min(E)) / np.mean(E)
            energy_fluctuation.append(delta)

        return energy_fluctuation  # valores próximos de 0 indicam conservação

    def generate_topology_map(self):
        """
        Gera um mapa topológico simplificado baseado em coerência fásica entre níveis.
        """
        n = self.levels
        topo = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                coherence = np.corrcoef(self.data[i], self.data[j])[0, 1]
                topo[i, j] = coherence
        return topo

    def simulate_multi_triad_interaction(self, num_tríades=3):
        """
        Simula a interferência entre múltiplas tríades com pequenas variações de fase.
        """
        simulated = []
        for i in range(num_tríades):
            phase_shift = (2 * np.pi / num_tríades) * i
            shifted = self.data * np.cos(phase_shift)
            simulated.append(shifted)
        return np.sum(simulated, axis=0)

    def map_energy_density(self):
        """
        Calcula a densidade de energia ressonante por nível ao longo do tempo.
        Baseia-se na variação temporal (cinética) e coerência vetorial local.
        """

        dt = self.dt
        energy_density = np.zeros_like(self.data)

        for i in range(self.levels):
            dA_dt = np.gradient(self.data[i], dt)

            # Coerência com vizinhos (circular)
            coherence = 1.0
            if i > 0:
                coherence += np.cos(self.data[i] - self.data[i - 1])
            if i < self.levels - 1:
                coherence += np.cos(self.data[i] - self.data[i + 1])

            # Energia ressonante local (proporcional à variação e acoplamento fásico)
            energy_density[i] = (dA_dt ** 2) * (coherence / 3)

        return energy_density

    def evaluate_symmetry_breaking(self):
        """
        Avalia mudanças abruptas nas relações de fase e frequência entre níveis.
        """
        symmetry_breaks = []
        for i in range(self.levels - 1):
            delta = np.abs(np.fft.fft(self.data[i]) - np.fft.fft(self.data[i+1]))
            symmetry_breaks.append(np.sum(delta))
        return symmetry_breaks
    
    def investigate_medium(self, freq_range=(0.1, 3.0), amp_range=(0.1, 2.0), steps=20):
        """
        Explora como o meio responde a diferentes combinações de frequência base e amplitude
        nos osciladores da tríade. A resposta é medida em termos de coerência vetorial média
        e variação na resposta do meio. Cada variação é aplicada a todos os níveis simultaneamente.

        Esse é um experimento epistêmico: não visa ajuste, mas observação de como o meio ressoa.
        """
        freq_values = np.linspace(freq_range[0], freq_range[1], steps)
        amp_values = np.linspace(amp_range[0], amp_range[1], steps)

        heatmap = np.zeros((steps, steps))

        for i, freq in enumerate(freq_values):
            for j, amp in enumerate(amp_values):
                # Aplicar configuração temporária nos parâmetros
                for k in range(self.levels):
                    self.params[k, 0] = freq
                    self.params[k, 3] = amp

                self.adjust_medium_properties()
                self.normalize_medium_response()

                # Medir coerência média como resposta do meio
                coerencias = [self.compute_vector_coherence(k) for k in range(self.levels)]
                coerencia_media = np.mean(coerencias)

                heatmap[j, i] = coerencia_media  # linha = amp, coluna = freq

        # Visualização
        plt.figure(figsize=(8, 6))
        plt.imshow(heatmap, extent=[*freq_range, *amp_range], origin='lower', aspect='auto', cmap='plasma')
        plt.colorbar(label="Coerência média do meio")
        plt.xlabel("Frequência base")
        plt.ylabel("Amplitude")
        plt.title("🧪 Resposta do meio à tríade ressonante")
        plt.show()

    def adjust_medium_properties(self):
        """
        Ajusta a resposta do meio de acordo com a coerência vetorial circular de cada nível.
        Quanto maior a coerência, mais "resonante" e responsivo é o meio local.
        """

        coherence = self.generate_topology_map()

        for i in range(self.levels):
            avg_coherence = np.mean(coherence[i])
            # O meio responde mais intensamente a zonas com alta coerência
            self.medium_response[i] = 1.0 + 0.1 * (avg_coherence - 0.5)

        return self.medium_response

    def symbolic_laplace_analysis(self, level=0):
        """
        Gera a representação simbólica da equação diferencial para um nível via Transformada de Laplace.
        """
        A = Function(f"A_{level}")(t)
        omega, gamma, amp = symbols('omega gamma amp')
        eq = -gamma * A + omega * A + amp * A**2
        laplace_expr = laplace_transform(eq, t, s, noconds=True)
        return simplify(laplace_expr)
    
    def adapt_pids(self):
        """
        Ajusta dinamicamente os parâmetros dos PIDs com base na coerência entre níveis.
        """
        topology = self.generate_topology_map()
        for i, pid in enumerate(self.pids):
            coherence = np.mean(topology[i])
            # Escala entre 0.5 e 1.5 dos parâmetros originais
            scale = 0.5 + coherence
            pid.Kp *= scale
            pid.Ki *= scale
            pid.Kd *= scale

    def compute_signal_entropy(self):
        """
        Calcula a entropia de Shannon para os sinais de cada nível.
        """
        entropies = []
        for i in range(self.levels):
            hist, _ = np.histogram(self.data[i], bins=50, density=True)
            hist = hist + 1e-9  # evitar log(0)
            entropies.append(entropy(hist))
        return entropies

    def compute_vector_coherence(self, i):
        """
        Calcula a coerência vetorial circular de um nível com seus vizinhos.
        Mede quão ressonante é a tríade formada entre os três eixos.
        """
        center = self.data[i]
        left = self.data[i - 1] if i > 0 else np.zeros_like(center)
        right = self.data[i + 1] if i < self.levels - 1 else np.zeros_like(center)

        coherence_left = np.cos(center - left)
        coherence_right = np.cos(center - right)

        return np.mean((coherence_left + coherence_right) / 2)

    def compute_emergent_mass(self, i):
        """
        Estima a massa como tempo de persistência da coerência vetorial local.
        Massa é emergente: tríades que mantêm coerência por mais tempo geram mais resposta do meio.
        """
        coherence_trace = []
        window = max(3, len(self.data[i]) // 20)

        for t in range(len(self.data[i]) - window):
            seg_center = self.data[i][t:t+window]
            seg_left = self.data[i-1][t:t+window] if i > 0 else np.zeros(window)
            seg_right = self.data[i+1][t:t+window] if i < self.levels - 1 else np.zeros(window)

            c1 = np.mean(np.cos(seg_center - seg_left))
            c2 = np.mean(np.cos(seg_center - seg_right))
            coherence_trace.append((c1 + c2) / 2)

        threshold = 0.9  # define coerência estável
        durations = [1 for c in coherence_trace if c > threshold]
        emergent_mass = len(durations) * self.dt
        return emergent_mass

    def compute_coherence_flux(self):
        """
        Calcula o fluxo de coerência vetorial entre todos os níveis.
        Indica regiões de transição de fase ou perda de unidade ressonante.
        """
        flux = np.zeros(self.levels)

        for i in range(1, self.levels - 1):
            pre = self.data[i - 1]
            curr = self.data[i]
            post = self.data[i + 1]

            coherence_left = np.mean(np.cos(curr - pre))
            coherence_right = np.mean(np.cos(curr - post))

            flux[i] = np.abs(coherence_left - coherence_right)

        return flux

    def visualize_triad_structure(self):
        """
        (Opcional) Prepara dados para visualização de uma tríade ressonante.
        Representa os 3 eixos (X, Y, Z) com suas fases e coerência relativa.
        """
        if self.levels < 3:
            print("Pelo menos 3 níveis são necessários para representar a tríade.")
            return

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = self.data[0]
        y = self.data[1]
        z = self.data[2]
        t = self.time

        ax.plot(x, y, z, label='Tríade Ressonante')
        ax.set_title("Osciladores Vetoriais em Três Eixos")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def normalize_medium_response(self):
        """
        Garante que a resposta do meio (densidade adaptativa) permaneça dentro de uma faixa controlada.
        Pode ser usada após longas simulações para manter a estabilidade do meio.
        """
        self.medium_response = np.clip(self.medium_response, 0.5, 2.0)
        return self.medium_response