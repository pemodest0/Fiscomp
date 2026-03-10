import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# MASSA-MOLA (SEM AMORTECIMENTO)
# ============================================================
# EDO do problema:
#   x'' + (k/m)*x = 0
#
# Comparamos dois integradores:
# 1) Euler-Cromer
# 2) Runge-Kutta de 4a ordem (RK4)
#
# A saida final mostra apenas a simulacao numerica e o plot comparativo.


# --------------------------
# BLOCO 1: PARAMETROS
# --------------------------
m = 1.0
k = 1.0

x0 = 1.0
v0 = 0.0

dt = 0.02
t_final = 10.0
t = np.arange(0.0, t_final + dt, dt)
N = len(t)


# --------------------------
# BLOCO 2: EULER-CROMER
# --------------------------
def simular_euler():
    x = np.zeros(N)
    v = np.zeros(N)

    x[0] = x0
    v[0] = v0

    for i in range(N - 1):
        a = -(k / m) * x[i]
        v[i + 1] = v[i] + a * dt
        x[i + 1] = x[i] + v[i + 1] * dt

    return x, v


# --------------------------
# BLOCO 3: RK4
# --------------------------
def derivadas(x, v):
    return v, -(k / m) * x


def simular_rk4():
    x = np.zeros(N)
    v = np.zeros(N)

    x[0] = x0
    v[0] = v0

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
# BLOCO 4: SAIDA CRUA
# --------------------------
def mostrar_saida_numerica(nome_metodo, x):
    print(f"\nSaida numerica - {nome_metodo}")
    print(" i   t(s)      x(m)        x/x0")
    for i in range(10):
        print(f"{i:2d}  {t[i]:6.3f}   {x[i]:10.6f}   {x[i] / x0:11.6f}")


# --------------------------
# BLOCO 5: PLOT COMPARATIVO
# --------------------------
def plotar_comparacao(x_euler, x_rk4):
    x_euler_n = x_euler / x0
    x_rk4_n = x_rk4 / x0
    amp = 1.1 * max(np.max(np.abs(x_euler_n)), np.max(np.abs(x_rk4_n)), 1.0)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(t, x_euler_n, color="tab:orange", lw=2, label="Euler-Cromer")
    ax.plot(t, x_rk4_n, color="tab:green", lw=2, label="RK4")
    ax.set_title("Sistema massa-mola: comparacao dos metodos")
    ax.set_xlabel("tempo (s)")
    ax.set_ylabel("x(t)/x0")
    ax.set_xlim(0.0, t_final)
    ax.set_ylim(-amp, amp)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


# --------------------------
# BLOCO 6: EXECUCAO
# --------------------------
x_euler, v_euler = simular_euler()
x_rk4, v_rk4 = simular_rk4()

mostrar_saida_numerica("Euler-Cromer", x_euler)
mostrar_saida_numerica("Runge-Kutta 4", x_rk4)
plotar_comparacao(x_euler, x_rk4)
