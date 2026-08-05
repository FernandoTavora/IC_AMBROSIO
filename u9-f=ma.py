import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from scipy.special import j1, jn_zeros
from scipy.ndimage import gaussian_filter
import os

def simular_cvfw_3D(img_path):

    L = 100e-6
    R = 30e-6
    lambda_0 = 632.8e-9
    k = 2 * np.pi / lambda_0
    k_band = 2 * k
    Q = 0.80 * k
    k_rho_Q = np.sqrt(k ** 2 - Q ** 2)

    n_folhas = 4

    alvo_y = 35e-6
    argumento_alvo = alvo_y * k_rho_Q

    # Calcula os zeros e trava no anel escuro de bessel mais próximo
    zeros_j0 = jn_zeros(0, 100)
    indice_zero_mais_proximo = np.argmin(np.abs(zeros_j0 - argumento_alvo))
    zero_exato = zeros_j0[indice_zero_mais_proximo]

    espaco_y = zero_exato / k_rho_Q
    print(f"Ajuste espacial para {espaco_y * 1e6:.2f} µm (anel escuro #{indice_zero_mais_proximo + 1})")

    # Distribuição para as folhas centradas
    limite_y = (n_folhas - 1) / 2 * espaco_y
    folhas_y = np.linspace(-limite_y, limite_y, n_folhas)

    # Malha matemática
    n_min = int(-L * k / np.pi)
    n_vec = np.arange(n_min, 0, 1)

    z_samples = np.sort(-2 * np.pi * n_vec / k_band)
    nz_exact = len(z_samples)

    dx_min = 4.81 / k_rho_Q
    nx = int(R / dx_min)

    img = Image.open(img_path).convert('L').resize((nz_exact, nx))
    img_data = np.array(img) / 255.0

    # Inverte as cores caso o fundo seja claro
    if np.mean(img_data) > 0.5:
        img_data = 1.0 - img_data

    f_matrix = np.flipud(img_data)
    f_matrix = gaussian_filter(f_matrix, sigma=0.8)
    f_matrix /= np.max(f_matrix)

    # Malha amostrada
    nx_vis, nz_vis = 200, 500
    x_axis = np.linspace(-R / 2, R / 2, nx_vis)
    z_axis = np.linspace(0, L, nz_vis)

    Z_grid, X_grid = np.meshgrid(z_axis, x_axis)
    x0_fws = np.linspace(-R / 2, R / 2, nx)

    # Largura da envoltória Gaussiana (para apodização transversal)
    w_0 = 40e-6

    # Folhas de luz com crosstalk real e apodização
    intensidade_planos = []

    for plano_y in folhas_y:
        Ex_plano = np.zeros_like(Z_grid, dtype=complex)
        Ez_plano = np.zeros_like(Z_grid, dtype=complex)

        for yp in folhas_y:
            for p in range(nx):
                f_envelope = f_matrix[p, :]
                if np.sum(f_envelope) < 0.01:
                    continue
                xp = x0_fws[p]

                A_n = f_envelope * np.exp(-1j * Q * z_samples)

                rho_g = np.sqrt((X_grid - xp) ** 2 + (plano_y - yp) ** 2)

                apodizacao = np.exp(-(rho_g ** 2) / (w_0 ** 2))

                arg_sinc = (k / np.pi) * np.sqrt(rho_g[..., None] ** 2 + (Z_grid[..., None] - z_samples) ** 2)

                # Aplica a apodização
                psi_3D = (np.sinc(arg_sinc) @ A_n) * apodizacao
                Ex_plano += psi_3D

                # Evita divisão por zero
                cos_phi = (X_grid - xp) / (rho_g + 1e-12)
                Ez_plano += 1j * (k_rho_Q / Q) * j1(k_rho_Q * rho_g) * cos_phi * psi_3D

        i_total = np.abs(Ex_plano) ** 2 + np.abs(Ez_plano) ** 2
        intensidade_planos.append(i_total)

#Cortes 2D

    fig2d, axes2d = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes2d = axes2d.flatten()

    for i, y_plane in enumerate(folhas_y):
        im = axes2d[i].imshow(
            intensidade_planos[i],
            extent=[0, L * 1e6, -R / 2 * 1e6, R / 2 * 1e6],
            aspect='auto',
            cmap='inferno',
            origin='lower'
        )
        axes2d[i].set_title(f"Folha {i + 1} (y = {y_plane * 1e6:.1f} $\\mu m$)")
        axes2d[i].set_ylabel(r"$x$ ($\mu m$)")
        if i >= 2:
            axes2d[i].set_xlabel(r"$z$ ($\mu m$)")

    plt.suptitle("", fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.show()

#Plotagem
    i_max_global = np.max(intensidade_planos)
    plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i, y_plane in enumerate(folhas_y):
        i_norm = intensidade_planos[i] / i_max_global
        cores = plt.cm.inferno(i_norm)

        # Limiar de transparência ajustado para manter a solidez e revelar a difração
        threshold = 0.18
        cores[..., 3] = np.where(i_norm > threshold, 0.85, 0.0)

        Y_surf = np.full_like(Z_grid, y_plane * 1e6)

        # Eixos em sua orientação original (folhas "em pé")
        ax.plot_surface(
            Z_grid * 1e6, Y_surf, X_grid * 1e6,
            facecolors=cores,
            rstride=1, cstride=1,
            linewidth=0, antialiased=True, shade=False
        )

    ax.set_title("", fontweight='bold', fontsize=16)
    ax.set_xlabel(r'$z$ ($\mu m$)')
    ax.set_ylabel(r'$y$ ($\mu m$)')
    ax.set_zlabel(r'$x$ ($\mu m$)')

    ax.set_xlim([0, L * 1e6])
    ax.set_ylim([(-limite_y - espaco_y) * 1e6, (limite_y + espaco_y) * 1e6])
    ax.set_zlim([-R / 2 * 1e6, R / 2 * 1e6])

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.view_init(elev=20, azim=-45)
    plt.tight_layout()
    plt.show()

simular_cvfw_3D("F=MA.png")