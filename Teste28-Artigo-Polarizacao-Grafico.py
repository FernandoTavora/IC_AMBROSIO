import numpy as np
import matplotlib.pyplot as plt
from scipy.special import airy

# 1. PARÂMETROS FÍSICOS GERAIS
lam = 0.532          # Comprimento de onda em micrometros
k = 2 * np.pi / lam  # Número de onda
K = 2 * k            # Largura de banda

# 2. FUNÇÕES MATEMÁTICAS AUXILIARES (Derivadas do sinc não normalizado)
def sinc_math(u):
    return np.where(u == 0, 1.0, np.sin(u) / u)

# g1(u) representa a derivada d(sinc(u))/du dividida por u
# g1(u) = (u*cos(u) - sin(u)) / u^3
def g1(u):
    ans = np.zeros_like(u)
    mask = np.abs(u) > 1e-4
    u_m = u[mask]
    ans[mask] = (u_m * np.cos(u_m) - np.sin(u_m)) / (u_m**3)
    # Expansão de Taylor para u -> 0 para evitar singularidade
    ans[~mask] = -1.0/3.0 + (u[~mask]**2)/30.0
    return ans

# g2(u) representa a parte não trivial da derivada cruzada para E_rho
# g2(u) = (3*sin(u) - 3*u*cos(u) - u^2*sin(u)) / u^5
def g2(u):
    ans = np.zeros_like(u)
    mask = np.abs(u) > 1e-4
    u_m = u[mask]
    ans[mask] = (3*np.sin(u_m) - 3*u_m*np.cos(u_m) - (u_m**2)*np.sin(u_m)) / (u_m**5)
    # Expansão de Taylor para u -> 0
    ans[~mask] = 1.0/15.0 - (u[~mask]**2)/210.0
    return ans

# 3. MALHA ESPACIAL E ESPECTRAL
z = np.linspace(-15, 15, 600)          # Eixo Z (micrometros)
kz_k = np.linspace(-1, 1, 600)         # Eixo kz/k (espectro normalizado)
kz = kz_k * k                          # Eixo kz

# Configuração da Figura
fig, axs = plt.subplots(2, 3, figsize=(16, 9))
color_target = '#81C1C1' # Azul contínuo
color_field = '#C54B5C'  # Vermelho tracejado

# 4. CASO A: POLARIZAÇÃO LINEAR (Eqs. 2, 6)
Z_A = 12.61
Q_A = 0.7 * k
N_A = 52 # 105 termos no total (de -52 a +52)
n_A = np.arange(-N_A, N_A + 1)
zn_A = -n_A * np.pi / k

def F_A(z_val):
    Ai, _, _, _ = airy(3 * np.pi * (z_val - 0.8 * Z_A) / Z_A)
    return np.exp(1j * Q_A * z_val) * np.exp(-0.5 * (z_val / Z_A)**8) * Ai

An_A = F_A(zn_A)

# Espectro S(k_z)
S_A = np.sum(An_A[:, None] * np.exp(1j * n_A[:, None] * np.pi * kz / k), axis=0)

# Reconstrução do Campo Ex em rho=0
Ex = np.zeros_like(z, dtype=complex)
for i, n in enumerate(n_A):
    u = k * z + n * np.pi
    Ex += An_A[i] * sinc_math(u)

target_A = np.abs(F_A(z))**2
actual_A = np.abs(Ex)**2

# Plot (a)
axs[0, 0].plot(kz_k, np.abs(S_A), color=color_target, lw=1)
axs[0, 0].set_title("(a)")
axs[0, 0].set_xlabel("$k_z / k$")
axs[0, 0].set_ylabel("$|S(k_z)|$")
axs[0, 0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
axs[0, 0].set_xlim(-1, 1)

# Plot (d)
axs[1, 0].plot(z, target_A, color=color_target, lw=2)
axs[1, 0].plot(z, actual_A, color=color_field, linestyle='--', lw=1.5)
axs[1, 0].set_title("(d)")
axs[1, 0].set_xlabel("$z \\ (\\mu m)$")
axs[1, 0].set_ylabel("$|E_x(\\rho=0, z, t) / E_0|^2$")

# 5. CASO B: POLARIZAÇÃO AZIMUTAL (Eqs. 8, 11)
Z_B = 10
Q_B = 0.75 * k
N_B = 38 # 77 termos no total
rho1 = 0.24
n_B = np.arange(-N_B, N_B + 1)
zn_B = -n_B * np.pi / k

def F_B(z_val):
    return np.exp(1j * Q_B * z_val) * (np.exp(-0.5 * ((z_val + Z_B/2)/(Z_B/4))**8) +
                                       np.exp(-0.5 * ((z_val - Z_B/2)/(Z_B/4))**8))

An_B = F_B(zn_B)

# Espectro S'(k_z) = sqrt(k^2 - k_z^2) * S(k_z)
S_B = np.sum(An_B[:, None] * np.exp(1j * n_B[:, None] * np.pi * kz / k), axis=0)
Sp_B = k * np.sqrt(np.clip(1 - kz_k**2, 0, None)) * S_B

# Reconstrução do Campo E_phi em rho=rho_1
Ephi = np.zeros_like(z, dtype=complex)
for i, n in enumerate(n_B):
    u = np.sqrt(k**2 * rho1**2 + (k * z + n * np.pi)**2)
    Ephi += An_B[i] * k * rho1 * g1(u)

target_B = np.abs(F_B(z))**2
target_B /= np.max(target_B) # Normalização como no artigo
actual_B = np.abs(Ephi)**2
actual_B /= np.max(actual_B)

# Plot (b)
axs[0, 1].plot(kz_k, np.abs(Sp_B), color=color_target, lw=1)
axs[0, 1].set_title("(b)")
axs[0, 1].set_xlabel("$k_z / k$")
axs[0, 1].set_ylabel("$|S'(k_z)|$")
axs[0, 1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
axs[0, 1].set_xlim(-1, 1)

# Plot (e)
axs[1, 1].plot(z, target_B, color=color_target, lw=2)
axs[1, 1].plot(z, actual_B, color=color_field, linestyle='--', lw=1.5)
axs[1, 1].set_title("(e)")
axs[1, 1].set_xlabel("$z \\ (\\mu m)$")
axs[1, 1].set_ylabel("$|E_\\phi(\\rho=\\rho_1, z, t) / E_0|^2$")

# 6. CASO C: POLARIZAÇÃO RADIAL (Eqs. 12, 15)
Z_C = 8
Q_C = 0.75 * k
N_C = 36 # 73 termos no total
rho0 = 0.24
n_C = np.arange(-N_C, N_C + 1)
zn_C = -n_C * np.pi / k

def F_C(z_val):
    return np.exp(1j * Q_C * z_val) * np.exp(-0.5 * (z_val / Z_C)**8) * np.cos(7 * np.pi * z_val / (2 * Z_C))

An_C = F_C(zn_C)

# Espectro S'(k_z)
S_C = np.sum(An_C[:, None] * np.exp(1j * n_C[:, None] * np.pi * kz / k), axis=0)
Sp_C = k * np.sqrt(np.clip(1 - kz_k**2, 0, None)) * S_C

# Reconstrução do Campo E_rho em rho=rho_0
Erho = np.zeros_like(z, dtype=complex)
for i, n in enumerate(n_C):
    u = np.sqrt(k**2 * rho0**2 + (k * z + n * np.pi)**2)
    # Derivada segunda analítica implementada em g2(u)
    Erho += An_C[i] * 1j * k * rho0 * (k * z + n * np.pi) * g2(u)

target_C = np.abs(F_C(z))**2
target_C /= np.max(target_C)
actual_C = np.abs(Erho)**2
actual_C /= np.max(actual_C)

# Plot (c)
axs[0, 2].plot(kz_k, np.abs(Sp_C), color=color_target, lw=1)
axs[0, 2].set_title("(c)")
axs[0, 2].set_xlabel("$k_z / k$")
axs[0, 2].set_ylabel("$|S'(k_z)|$")
axs[0, 2].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
axs[0, 2].set_xlim(-1, 1)

# Plot (f)
axs[1, 2].plot(z, target_C, color=color_target, lw=2)
axs[1, 2].plot(z, actual_C, color=color_field, linestyle='--', lw=1.5)
axs[1, 2].set_title("(f)")
axs[1, 2].set_xlabel("$z \\ (\\mu m)$")
axs[1, 2].set_ylabel("$|E_\\rho(\\rho=\\rho_0, z, t) / E_0|^2$")

# 7. VISUALIZAÇÃO
plt.tight_layout()
plt.show()
