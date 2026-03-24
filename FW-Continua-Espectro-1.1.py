import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
import os


def analisar_espectro_fw_continua_corrigido(caminho_img):
    if not os.path.exists(caminho_img):
        print(f"Erro: Arquivo não encontrado em {caminho_img}")
        return

    # 1. PARÂMETROS FÍSICOS
    L = 100e-6
    R = 30e-6
    lambda_0 = 632.8e-9
    k = 2 * np.pi / lambda_0
    K_banda = 2 * k
    Q = 0.80 * k
    k_rho_Q = np.sqrt(k ** 2 - Q ** 2)

    # 2. A MALHA MATEMÁTICA E O LIMITE DE RESOLUÇÃO
    n_min = int(-L * k / np.pi)
    n_vec = np.arange(n_min, 0, 1)
    z_samples = -2 * np.pi * n_vec / K_banda
    z_samples_sorted = np.sort(z_samples)
    nz_exato = len(z_samples_sorted)

    # Espaçamento mínimo
    delta_x_min = 4.81 / k_rho_Q
    nx_maximo_permitido = int(R / delta_x_min)
    nx = nx_maximo_permitido

    # 3. DIGITALIZAÇÃO SUAVIZADA
    img = Image.open(caminho_img).convert('L')
    img = img.resize((nz_exato, nx))
    img_data = np.array(img)
    norm_img = img_data / 255.0

    if np.mean(norm_img) > 0.5:
        F_matrix = 1.0 - norm_img
    else:
        F_matrix = norm_img

    F_matrix = np.flipud(F_matrix)

    # sigma=0.8 para manter estrita coerência com o script principal da FW
    F_matrix = gaussian_filter(F_matrix, sigma=0.8)

    x_axis_vis = np.linspace(-R / 2, R / 2, nx)

    # 4. A MALHA ESPECTRAL
    num_beta = 600
    kz_k_axis = np.linspace(-1.2, 1.2, num_beta)
    beta = kz_k_axis * k

    S_2D = np.zeros((nx, num_beta), dtype=complex)
    phase_matrix = np.exp(1j * beta[:, None] * z_samples_sorted[None, :])

    for p in range(nx):
        f_envelope = F_matrix[p, :]
        if np.sum(f_envelope) < 0.01: continue
        A_n = f_envelope * np.exp(-1j * Q * z_samples_sorted)
        S_2D[p, :] = phase_matrix @ A_n


    # 5. VISUALIZAÇÃO ESPECTRAL
    Intensidade_Espectral_2D = np.abs(S_2D)

    Espectro_Medio = np.mean(Intensidade_Espectral_2D, axis=0)
    Espectro_Maximo = np.max(Intensidade_Espectral_2D, axis=0)

    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    extent_real = [0, L * 1e6, -R / 2 * 1e6, R / 2 * 1e6]
    im0 = axes[0].imshow(F_matrix, extent=extent_real, aspect='auto', cmap='gray', origin='lower')
    axes[0].set_title(r'O Alvo Físico $F(x, z)$ (Com Suavização Gaussiana)')
    axes[0].set_xlabel(r'$z$ ($\mu m$)')
    axes[0].set_ylabel(r'$x$ ($\mu m$)')

    axes[1].plot(kz_k_axis, Espectro_Medio, color='darkblue', lw=2.5, label='Espectro Médio (Folha de Luz)')

    axes[1].fill_between(kz_k_axis, 0, Espectro_Maximo, color='royalblue', alpha=0.3,
                         label=f'Envelope Máximo (Limites de todas as {nx} FWs)')

    axes[1].set_title(r'Espectro Longitudinal $|S(k_z)|$')
    axes[1].set_xlabel(r'$k_z / k$')
    axes[1].set_ylabel(r'$|S(k_z)|$')

    axes[1].axvline(x=0.80, color='orange', linestyle='-', lw=2, alpha=0.9, label='Portadora $Q=0.8k$')
    axes[1].axvline(x=1.0, color='red', linestyle='--', lw=2, alpha=0.8, label='Limite Físico ($k_z=k$)')
    axes[1].axvline(x=-1.0, color='red', linestyle='--', lw=2, alpha=0.8)

    axes[1].legend(loc='upper left', fontsize=11)
    axes[1].set_xlim(-1.0, 1.0)
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()


# 6. Execução
caminho = r"F=MA.png"
analisar_espectro_fw_continua_corrigido(caminho)