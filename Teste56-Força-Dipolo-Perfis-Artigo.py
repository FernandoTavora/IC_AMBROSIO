import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn

def simular_forca_dipolo():
# Parâmetros
    lam = 1064e-9
    k = 2 * np.pi / lam
    M = 1.2 
    d = 50e-9 
    r = d / 2
    eta = 1.0

# Abaixo teremos 3 opções de perfis a serem analisados, escolher apenas 1 e comentar os outros:

    # Perfil gaussiano:
    # Zmax = 30e-6; Q = 0.85 * k; n_exp = 2
    # def F_func(z): return np.exp(-8 * (z/Zmax)**n_exp + 1j * Q * z)
    # def dF_dz_func(z): return F_func(z) * (-8 * n_exp * (z**(n_exp-1) / Zmax**n_exp) + 1j * Q)

# Perfil super-gaussiano:
    # Zmax = 40e-6;
    # Q = 0.8 * k;
    # n_exp = 8
    # def F_func(z): return np.exp(-8 * (z / Zmax) ** n_exp + 1j * Q * z)
    # def dF_dz_func(z): return F_func(z) * (-8 * n_exp * (z ** (n_exp - 1) / Zmax ** n_exp) + 1j * Q)

# Perfil senoidal modulado por envelope super-gaussiano:
    Zmax = 30e-6; Q = 0.8 * k; n_exp = 8; freq_mod = 2 * np.pi / Zmax
    def F_func(z): return np.exp(-8 * (z/Zmax)**n_exp + 1j * Q * z) * np.cos(freq_mod * z)
    def dF_dz_func(z):
        termo_exp = np.exp(-8 * (z/Zmax)**n_exp + 1j * Q * z)
        deriv_exp = termo_exp * (-8 * n_exp * (z**(n_exp-1) / Zmax**n_exp) + 1j * Q)
        return deriv_exp * np.cos(freq_mod * z) - termo_exp * freq_mod * np.sin(freq_mod * z)

# Perfil do centro nulo modulado por envelope super-gaussiano:
#     Zmax = 30e-6;
#     Q = 0.8 * k;
#     n_exp = 8
#     def F_func(z):
#         # O termo (z^2) garante o zero central
#         return (z ** 2 / Zmax ** 2) * np.exp(-8 * (z / Zmax) ** n_exp + 1j * Q * z)
#     def dF_dz_func(z):
#         termo_env = np.exp(-8 * (z / Zmax) ** n_exp + 1j * Q * z)
#         deriv_env = termo_env * (-8 * n_exp * (z ** (n_exp - 1) / Zmax ** n_exp) + 1j * Q)
#         return (2 * z / Zmax ** 2) * termo_env + (z ** 2 / Zmax ** 2) * deriv_env

# Polarizabilidade (alpha = -i(3/2k^3)a1)
    x = k * r;
    y = M * x

    # Funções de Riccati-Bessel usadas no artigo (p/ n=1)/ teoria de Mie
    psi_x = x * spherical_jn(1, x)
    psi_y = y * spherical_jn(1, y)
    xi_x = x * (spherical_jn(1, x) - 1j * spherical_yn(1, x))

    # Derivadas
    dpsi_x = spherical_jn(1, x) + x * (spherical_jn(0, x) - 2 * spherical_jn(1, x) / x)
    dpsi_y = spherical_jn(1, y) + y * (spherical_jn(0, y) - 2 * spherical_jn(1, y) / y)
    dxi_x = dpsi_x - 1j * (spherical_yn(1, x) + x * (spherical_yn(0, x) - 2 * spherical_yn(1, x) / x))

    a1 = (psi_x * dpsi_y - M * psi_y * dpsi_x) / (xi_x * dpsi_y - M * psi_y * dxi_x)
    alpha = -1j * (3 / (2 * k ** 3)) * a1

# z0 é o deslocamento do feixe e a partícula está em 0.
    z0_axis = np.linspace(-40e-6, 40e-6, 1000)

    # O feixe translada e a partícula fica na origem, logo é avaliado em z = -z0
    Z_eval = -z0_axis

    # Criação dos pontos de amostragem para "montar" a frozen wave com o perfil morfológico
    n_min, n_max = -400, 400
    n_vec = np.arange(n_min, n_max + 1)
    z_samples = n_vec * np.pi / k
    F_samples = F_func(z_samples)

    # Matriz do campo e derivada
    Psi_real = np.zeros_like(Z_eval, dtype=complex)
    dPsi_dz_real = np.zeros_like(Z_eval, dtype=complex)

    # Somatorio das FWs continuas com a solução sinc
    for i, zl in enumerate(z_samples):
        F_val = F_samples[i]

        # Argumento da função sinc
        u = (k / np.pi) * (Z_eval - zl)
        sinc_val = np.sinc(u)

        # Derivada da função sinc para calculo da força
        pi_u = np.pi * u
        dsinc_du = np.zeros_like(u)
        nz = np.abs(u) > 1e-12  # Evita divisão por zero no centro do sinc

        dsinc_du[nz] = (pi_u[nz] * np.cos(pi_u[nz]) - np.sin(pi_u[nz])) / (np.pi * u[nz] ** 2)
        dsinc_dz = dsinc_du * (k / np.pi)

        # Acumula no campo total e na derivada
        Psi_real += F_val * sinc_val
        dPsi_dz_real += F_val * dsinc_dz

    # Aplicação da formula de força de dipolo considerando Ex = Psi
    F_z = (2 * np.pi / eta) * np.real(alpha * Psi_real * np.conj(dPsi_dz_real))
    F_z_nm2 = F_z * 1e18

    # Para plotagem da intensidade, avaliamos em +z0 para exibir o feixe de frente
    Intensidade_F = np.abs(F_func(z0_axis)) ** 2

    # Reconstruindo a intensidade visual para a FW
    Psi_visual = np.zeros_like(z0_axis, dtype=complex)
    for i, zl in enumerate(z_samples):
        Psi_visual += F_samples[i] * np.sinc((k / np.pi) * (z0_axis - zl))
    Intensidade_Psi = np.abs(Psi_visual) ** 2

#Plotagem
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # Grafico (a): Intensidades
    ax1.plot(z0_axis * 1e6, Intensidade_F, color='black', linewidth=1.5, label=r'$|F(z)|^2$')
    ax1.plot(z0_axis * 1e6, Intensidade_Psi, color='blue', linewidth=1.5, linestyle='--',
             label=r'$|\Psi(0,z)|^2$')
    ax1.set_title("(a) Perfil de intensidade longitudinal", fontweight='bold', fontsize=13)
    ax1.set_ylabel("Magnitude")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    # Grafico (b): Força de dipolo
    ax2.plot(z0_axis * 1e6, F_z_nm2, color='black', linewidth=1.5)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_title("(b) Força Longitudinal $F_z$ (Dipolo)", fontweight='bold', fontsize=13)
    ax2.set_xlabel(r"$z_0$ ($\mu m$)")
    ax2.set_ylabel(r"$F_z$ ($nm^2$)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

simular_forca_dipolo()
