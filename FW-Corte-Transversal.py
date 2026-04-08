import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0, j1

def avaliar_conf_transversal():
    lambda_0 = 632.8e-9
    k = 2 * np.pi / lambda_0
    Q = 0.80 * k
    k_rho_Q = np.sqrt(k ** 2 - Q ** 2)

    # Avaliando em uma janela de -6 a +6 micrômetros
    x = np.linspace(-6e-6, 6e-6, 2000)
    rho = np.abs(x)

    cos_phi = np.sign(x)
    cos_phi[cos_phi == 0] = 1.0

    # No modelo de Mackinnon, o perfil transversal é descrito por J0
    E_x = j0(k_rho_Q * rho)

    # O campo Ez surge do gradiente de Ex (da lei de gauss), descrito por J1
    fator_vetorial_Ez = 1j * (k_rho_Q / Q)
    E_z = fator_vetorial_Ez * j1(k_rho_Q * rho) * cos_phi

    I_x = np.abs(E_x) ** 2
    I_z = np.abs(E_z) ** 2
    I_tot = I_x + I_z

    raio_confinamento = 2.405 / k_rho_Q
    diametro_confinamento = 2 * raio_confinamento

    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14})
    plt.figure(figsize=(12, 7))

    # Plot de intensidades
    plt.plot(x * 1e6, I_x, label=r'Componente Principal $|E_x|^2 \propto |J_0|^2$', color='darkorange', lw=2.5)
    plt.plot(x * 1e6, I_z, label=r'Componente Secundária $|E_z|^2 \propto |J_1|^2$', color='royalblue', lw=2.5)
    plt.plot(x * 1e6, I_tot, label=r'Intensidade Total $|E_{tot}|^2$', color='black', linestyle='--', alpha=0.5, lw=1.5)

    plt.axvline(raio_confinamento * 1e6, color='red', linestyle=':', lw=2,
                label='Primeiro Zero (Fim do Lóbulo Central)')
    plt.axvline(-raio_confinamento * 1e6, color='red', linestyle=':', lw=2)

    plt.fill_between(x * 1e6, 0, I_x, where=(np.abs(x) <= raio_confinamento), color='bisque', alpha=0.5)

    plt.title(f'Corte Transversal de uma Única FW Contínua Vetorial ($Q = {Q / k:.2f}k$)')
    plt.xlabel(r'Eixo Transversal $x$ ($\mu m$)')
    plt.ylabel('Intensidade Normalizada')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.4)
    plt.xlim(-6, 6)
    plt.ylim(-0.05, 1.1)

    plt.tight_layout()
    plt.show()

avaliar_conf_transversal()
