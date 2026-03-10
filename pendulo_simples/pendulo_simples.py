import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PENDULO SIMPLES
# ============================================================
# EDO do problema:
#   theta'' + (g/L)*sin(theta) = 0
#
# Comparamos dois integradores:
# 1) Euler-Cromer
# 2) Runge-Kutta de 4a ordem (RK4)
#
# A saida final mostra apenas a simulacao numerica e o plot comparativo.


# --------------------------
# BLOCO 1: PARAMETROS
# --------------------------
g = 9.81
L = 1.0

theta0 = np.deg2rad(20.0)
omega0 = 0.0

dt = 0.01
t_final = 10.0
t = np.arange(0.0, t_final + dt, dt)
N = len(t)


# --------------------------
# BLOCO 2: EULER-CROMER
# --------------------------
def simular_euler():
    theta = np.zeros(N)
    omega = np.zeros(N)

    theta[0] = theta0
    omega[0] = omega0

    for i in range(N - 1):
        alpha = -(g / L) * np.sin(theta[i])
        omega[i + 1] = omega[i] + alpha * dt
        theta[i + 1] = theta[i] + omega[i + 1] * dt

    return theta, omega


# --------------------------
# BLOCO 3: RK4
# --------------------------
def derivadas(theta, omega):
    return omega, -(g / L) * np.sin(theta)


def simular_rk4():
    theta = np.zeros(N)
    omega = np.zeros(N)

    theta[0] = theta0
    omega[0] = omega0

    for i in range(N - 1):
        th = theta[i]
        om = omega[i]

        k1_th, k1_om = derivadas(th, om)
        k2_th, k2_om = derivadas(th + 0.5 * dt * k1_th, om + 0.5 * dt * k1_om)
        k3_th, k3_om = derivadas(th + 0.5 * dt * k2_th, om + 0.5 * dt * k2_om)
        k4_th, k4_om = derivadas(th + dt * k3_th, om + dt * k3_om)

        theta[i + 1] = th + (dt / 6.0) * (k1_th + 2 * k2_th + 2 * k3_th + k4_th)
        omega[i + 1] = om + (dt / 6.0) * (k1_om + 2 * k2_om + 2 * k3_om + k4_om)

    return theta, omega


# --------------------------
# BLOCO 4: SAIDA CRUA
# --------------------------
def mostrar_saida_numerica(nome_metodo, theta):
    print(f"\nSaida numerica - {nome_metodo}")
    print(" i   t(s)    theta(rad)    theta/theta0")
    for i in range(10):
        print(f"{i:2d}  {t[i]:6.3f}   {theta[i]:10.6f}   {theta[i] / theta0:11.6f}")


# --------------------------
# BLOCO 5: PLOT COMPARATIVO
# --------------------------
def plotar_comparacao(theta_euler, theta_rk4):
    theta_euler_n = theta_euler / theta0
    theta_rk4_n = theta_rk4 / theta0
    amp = 1.1 * max(np.max(np.abs(theta_euler_n)), np.max(np.abs(theta_rk4_n)), 1.0)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(t, theta_euler_n, color="tab:orange", lw=2, label="Euler-Cromer")
    ax.plot(t, theta_rk4_n, color="tab:green", lw=2, label="RK4")
    ax.set_title("Pendulo simples: comparacao dos metodos")
    ax.set_xlabel("tempo (s)")
    ax.set_ylabel("theta(t)/theta0")
    ax.set_xlim(0.0, t_final)
    ax.set_ylim(-amp, amp)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


# --------------------------
# BLOCO 6: EXECUCAO
# --------------------------
theta_euler, omega_euler = simular_euler()
theta_rk4, omega_rk4 = simular_rk4()

mostrar_saida_numerica("Euler-Cromer", theta_euler)
mostrar_saida_numerica("Runge-Kutta 4", theta_rk4)
plotar_comparacao(theta_euler, theta_rk4)
