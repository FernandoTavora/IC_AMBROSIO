import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.special import j1
from scipy.ndimage import gaussian_filter
import os

def simular_fw_continua_vetorial_corrigida(caminho_img):
    if not os.path.exists(caminho_img):
        print(f"Erro: Arquivo não encontrado em {caminho_img}")
        return

    # PARÂMETROS FÍSICOS
    L = 100e-6
    R = 30e-6
    lambda_0 = 632.8e-9
    k = 2 * np.pi / lambda_0
    K_banda = 2 * k
    Q = 0.80 * k
    k_rho_Q = np.sqrt(k ** 2 - Q ** 2)

    # MALHA MATEMÁTICA E O LIMITE DE RESOLUÇÃO

    n_min = int(-L * k / np.pi)
    n_vec = np.arange(n_min, 0, 1)

    #Amostragem
    z_samples = -2 * np.pi * n_vec / K_banda
    z_samples_sorted = np.sort(z_samples)
    nz_exato = len(z_samples_sorted)

    # Cálculo do espaçamento mínimo obrigatório para não haver encavalamento destrutivo
    delta_x_min = 4.81 / k_rho_Q

    # O número máximo de FWs é a janela R dividida pelo espaçamento mínimo
    nx_maximo_permitido = int(R / delta_x_min)

    # O nx no limite físico calculado
    nx = nx_maximo_permitido

    # DIGITALIZAÇÃO 1:1
    img = Image.open(caminho_img).convert('L')
    img = img.resize((nz_exato, nx)) 
    img_data = np.array(img)

    norm_img = img_data / 255.0
    if np.mean(norm_img) > 0.5:
        F_matrix = 1.0 - norm_img
    else:
        F_matrix = norm_img

    F_matrix = np.flipud(F_matrix)

    # Aplicação do Filtro Gaussiano para impedir a queda brusca de uma zona ilumidada para escura (para não requisitar frequências infinitas)
    F_matrix = gaussian_filter(F_matrix, sigma=0.8)
    F_matrix = F_matrix / np.max(F_matrix)

    # MALHA VISUAL
    resolucao_visual = 500
    z_axis_vis = np.linspace(0, L, resolucao_visual)
    x_axis_vis = np.linspace(-R / 2, R / 2, nx)

    #Definindo feixes de Mackinnon
    arg_sinc = (k / np.pi) * (z_axis_vis[:, None] - z_samples_sorted[None, :])
    kernel_mackinnon = np.sinc(arg_sinc)

    E_x = np.zeros((nx, resolucao_visual), dtype=complex)
    E_z = np.zeros((nx, resolucao_visual), dtype=complex)

    # PROCESSAMENTO VETORIAL
    for p in range(nx):
        f_envelope = F_matrix[p, :]
        if np.sum(f_envelope) < 0.01: continue

        A_n = f_envelope * np.exp(-1j * Q * z_samples_sorted)

        #Produto matricial que substitui os somatórios
        psi_line = kernel_mackinnon @ A_n
        E_x[p, :] = psi_line

        rho = np.abs(x_axis_vis[p])
        cos_phi = np.sign(x_axis_vis[p]) if x_axis_vis[p] != 0 else 1.0

        fator_vetorial_Ez = 1j * (k_rho_Q / Q) * j1(k_rho_Q * rho) * cos_phi
        E_z[p, :] = fator_vetorial_Ez * psi_line

    # VISUALIZAÇÃO
    Intensidade_Ex = np.abs(E_x) ** 2
    Intensidade_Ez = np.abs(E_z) ** 2
    Intensidade_Total = Intensidade_Ex + Intensidade_Ez

    extent = [0, L * 1e6, -R / 2 * 1e6, R / 2 * 1e6]

    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    im0 = axes[0].imshow(Intensidade_Total, extent=extent, aspect='auto', cmap='inferno', origin='lower')
    axes[0].set_title(r'Intensidade Elétrica Total ($|E_x|^2 + |E_z|^2$)')
    axes[0].set_ylabel(r'$x$ ($\mu m$)')
    cbar0 = fig.colorbar(im0, ax=axes[0])
    cbar0.set_label(r'Intensidade', size=14)

    im1 = axes[1].imshow(Intensidade_Ex, extent=extent, aspect='auto', cmap='inferno', origin='lower')
    axes[1].set_title(r'Componente Transversal Principal ($|E_x|^2$)')
    axes[1].set_ylabel(r'$x$ ($\mu m$)')
    cbar1 = fig.colorbar(im1, ax=axes[1])
    cbar1.set_label(r'Intensidade', size=14)

    im2 = axes[2].imshow(Intensidade_Ez, extent=extent, aspect='auto', cmap='inferno', origin='lower')
    axes[2].set_title(r'Componente Longitudinal Secundária ($|E_z|^2$)')
    axes[2].set_xlabel(r'$z$ ($\mu m$)')
    axes[2].set_ylabel(r'$x$ ($\mu m$)')
    cbar2 = fig.colorbar(im2, ax=axes[2])
    cbar2.set_label(r'Intensidade Relativa', size=14)

    plt.tight_layout()
    plt.show()

# EXECUÇÃO
caminho = r"F=MA.png"
simular_fw_continua_vetorial_corrigida(caminho)
