import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# MASSA-MOLA (SEM AMORTECIMENTO)
# ============================================================
# EDO do problema:
#   x'' + (k/m)*x = 0
#
# Neste arquivo resolvemos a mesma EDO de duas formas:
# 1) Método de Euler
# 2) Método de Runge-Kutta de 4ª ordem (RK4)
#
# Para cada método:
# - mostramos saída numérica
# - mostramos animação da mola oscilando
# - mostramos gráfico de posição x(t)


# --------------------------
# BLOCO 1: PARÂMETROS
# --------------------------
# Parâmetros físicos
m = 1.0          # massa (kg)
k = 1.0          # constante elástica (N/m)

# Condições iniciais
x0 = 1.0         # posição inicial (m)
v0 = 0.0         # velocidade inicial (m/s)

# Parâmetros numéricos
dt = 0.02
t_final = 10.0
t = np.arange(0.0, t_final + dt, dt)
N = len(t)


# --------------------------
# BLOCO 2: MÉTODO DE EULER
# --------------------------
def simular_euler():
    # Vetores para guardar posição e velocidade
    x = np.zeros(N)
    v = np.zeros(N)

    # Condições iniciais
    x[0] = x0
    v[0] = v0

    # Integração temporal com Euler
    for i in range(N - 1):
        a = -(k / m) * x[i]      # aceleração da EDO
        v[i + 1] = v[i] + a * dt
        x[i + 1] = x[i] + v[i] * dt

    return x, v


# --------------------------
# BLOCO 3: MÉTODO RK4
# --------------------------
def derivadas(x, v):
    # Sistema equivalente de 1ª ordem
    dx_dt = v
    dv_dt = -(k / m) * x
    return dx_dt, dv_dt


def simular_rk4():
    # Vetores para guardar posição e velocidade
    x = np.zeros(N)
    v = np.zeros(N)

    # Condições iniciais
    x[0] = x0
    v[0] = v0

    # Integração temporal com RK4
    for i in range(N - 1):
        xi = x[i]
        vi = v[i]

        k1_x, k1_v = derivadas(xi, vi)
        k2_x, k2_v = derivadas(xi + 0.5 * dt * k1_x, vi + 0.5 * dt * k1_v)
        k3_x, k3_v = derivadas(xi + 0.5 * dt * k2_x, vi + 0.5 * dt * k2_v)
        k4_x, k4_v = derivadas(xi + dt * k3_x, vi + dt * k3_v)

        x[i + 1] = xi + (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        v[i + 1] = vi + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

    return x, v


# --------------------------
# BLOCO 4: SAÍDA CRUA
# --------------------------
def mostrar_saida_numerica(nome_metodo, x, v):
    print(f"\nSaída numérica - {nome_metodo}")
    print(" i   t(s)      x(m)        v(m/s)")
    for i in range(10):
        print(f"{i:2d}  {t[i]:6.3f}   {x[i]:10.6f}   {v[i]:10.6f}")


# --------------------------
# BLOCO 5: DESENHO DA MOLA
# --------------------------
def desenhar_mola(x_massa, x_parede=-1.2, espiras=12, amplitude=0.08, pontos=100):
    # Posição final da mola: um pouco antes do centro da massa
    x_final = x_massa - 0.08

    # Evita mola "invertida" quando a massa aproxima muito da parede
    if x_final <= x_parede + 0.05:
        x_final = x_parede + 0.05

    # Eixo x da mola
    xs = np.linspace(x_parede, x_final, pontos)

    # Forma de espiras usando seno
    fase = np.linspace(0.0, 2.0 * np.pi * espiras, pontos)
    ys = amplitude * np.sin(fase)

    # Força ponta inicial e final em y=0 para encostar reto na parede/massa
    ys[0] = 0.0
    ys[-1] = 0.0

    return xs, ys


# --------------------------
# BLOCO 6: ANIMAÇÃO + GRÁFICO
# --------------------------
def animar_mola(nome_metodo, x):
    # Cria figura com dois painéis
    fig, (ax_mola, ax_xt) = plt.subplots(
        2,
        1,
        figsize=(8, 7),
        gridspec_kw={"height_ratios": [1.3, 1.0]},
    )

    # ----- Painel superior: mola oscilando -----
    ax_mola.set_title(f"{nome_metodo} - Mola oscilando")
    ax_mola.set_xlabel("x (m)")
    ax_mola.set_ylabel("y")

    # Limites do painel espacial
    xmin = min(-1.3, np.min(x) - 0.3)
    xmax = max(1.3, np.max(x) + 0.3)
    ax_mola.set_xlim(xmin, xmax)
    ax_mola.set_ylim(-0.4, 0.4)
    ax_mola.grid(alpha=0.3)

    # Parede fixa onde a mola está presa
    ax_mola.axvline(-1.2, color="gray", lw=3)

    # Artistas que serão atualizados na animação
    linha_mola, = ax_mola.plot([], [], color="tab:blue", lw=2)
    massa, = ax_mola.plot([], [], "o", color="tab:red", ms=14)

    # ----- Painel inferior: gráfico de posição -----
    ax_xt.set_title(f"{nome_metodo} - Posição x(t)")
    ax_xt.set_xlabel("tempo (s)")
    ax_xt.set_ylabel("x (m)")
    ax_xt.set_xlim(0.0, t_final)

    margem = 0.1 * max(abs(np.min(x)), abs(np.max(x)), 1e-6)
    ax_xt.set_ylim(np.min(x) - margem, np.max(x) + margem)
    ax_xt.grid(alpha=0.3)

    linha_xt, = ax_xt.plot([], [], color="tab:orange", lw=2)

    # Texto com a EDO no rodapé da figura
    fig.subplots_adjust(bottom=0.14, hspace=0.4)
    fig.text(0.02, 0.04, "EDO: x'' + (k/m)*x = 0", fontsize=10)

    # Função de inicialização da animação
    def init():
        linha_mola.set_data([], [])
        massa.set_data([], [])
        linha_xt.set_data([], [])
        return linha_mola, massa, linha_xt

    # Função chamada a cada frame
    def update(i):
        # Calcula o desenho da mola para a posição atual da massa
        xs, ys = desenhar_mola(x[i])

        # Atualiza mola e massa
        linha_mola.set_data(xs, ys)
        massa.set_data([x[i]], [0.0])

        # Atualiza o gráfico x(t) até o instante atual
        linha_xt.set_data(t[: i + 1], x[: i + 1])

        return linha_mola, massa, linha_xt

    # Cria e mostra animação
    ani = FuncAnimation(fig, update, frames=N, init_func=init, interval=12, blit=False)
    plt.show()
    return ani


# --------------------------
# BLOCO 7: EXECUÇÃO
# --------------------------
# 1) Primeiro Euler
# 2) Depois RK4

# Euler
x_euler, v_euler = simular_euler()
mostrar_saida_numerica("Euler", x_euler, v_euler)
ani_euler = animar_mola("Euler", x_euler)

# RK4
x_rk4, v_rk4 = simular_rk4()
mostrar_saida_numerica("Runge-Kutta 4", x_rk4, v_rk4)
ani_rk4 = animar_mola("Runge-Kutta 4", x_rk4)
