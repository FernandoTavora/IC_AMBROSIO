import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from scipy.special import j1
from scipy.ndimage import gaussian_filter
from pathlib import Path

def simular_cvfw_3D(img_path):
    L = 100e-6
    R = 30e-6
    lambda_0 = 632.8e-9
    k = 2 * np.pi / lambda_0
    k_band = 2 * k
    Q = 0.80 * k
    k_rho_Q = np.sqrt(k ** 2 - Q ** 2)

    # Configuração das fls
    n_folhas = 4
    espaco_y = 35e-6
    folhas_y = np.linspace(-1.5 * espaco_y, 1.5 * espaco_y, n_folhas)

# Mesmo processo feito nos outros códigos
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
    nx_vis, nz_vis = 300, 750
    x_axis = np.linspace(-R / 2, R / 2, nx_vis)
    z_axis = np.linspace(0, L, nz_vis)

    Z_grid, X_grid = np.meshgrid(z_axis, x_axis)
    x0_fws = np.linspace(-R / 2, R / 2, nx)

    # Folhas de luz isoladas (Sem efeito crosstalk)
    intensidade_planos= []

    for plano_y in folhas_y:
        Ex_plano = np.zeros_like(Z_grid, dtype=complex)
        Ez_plano = np.zeros_like(Z_grid, dtype=complex)

        # O plano emissor (yp) agora é apenas o próprio
        # plano de observação.
        yp = plano_y

        for p in range(nx):
            f_envelope = f_matrix[p, :]
            if np.sum(f_envelope) < 0.01:
                continue
            xp = x0_fws[p]
            A_n = f_envelope * np.exp(-1j * Q * z_samples)

            # Como plano_y == yp, a diferença (plano_y - yp) será sempre ZERO.
            # Estav sendo calculado a propagação puramente no eixo central da folha
            rho_g = np.sqrt((X_grid - xp) ** 2 + (plano_y - yp) ** 2)
            arg_sinc = (k / np.pi) * np.sqrt(rho_g[..., None] ** 2 + (Z_grid[..., None] - z_samples) ** 2)

            psi_3D = np.sinc(arg_sinc) @ A_n
            Ex_plano += psi_3D

            cos_phi = (X_grid - xp) / (rho_g + 1e-12)
            Ez_plano += 1j * (k_rho_Q / Q) * j1(k_rho_Q * rho_g) * cos_phi * psi_3D

        i_total = np.abs(Ex_plano) ** 2 + np.abs(Ez_plano) ** 2
        intensidade_planos.append(i_total)

# Subtração de planos para avaliar se as intensidades são as mesmas
    # Subtrai a fl 2 da fl 1
    dif_1_2 = np.max(np.abs(intensidade_planos[0] - intensidade_planos[1]))
    # Subtrai a fl 4 da fl 1
    dif_1_4 = np.max(np.abs(intensidade_planos[0] - intensidade_planos[3]))

    print(f"Diferença de intensidade entre fl 1 e fl 2: {dif_1_2}")
    print(f"Diferença de intensidade entre fl 1 e fl 4: {dif_1_4}")

# Renderização 3D
    i_max_global = np.max(intensidade_planos)
    plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i, y_plane in enumerate(folhas_y):
        i_norm = intensidade_planos[i] / i_max_global
        cores = plt.cm.inferno(i_norm)

        # esconde pixels com intensidade muito baixa
        threshold = 0.20
        cores[..., 3] = np.where(i_norm > threshold, 0.9, 0.0)

        Y_surf = np.full_like(Z_grid, y_plane * 1e6)

        ax.plot_surface(
            Z_grid * 1e6, Y_surf, X_grid * 1e6,
            facecolors=cores,
            rstride=1, cstride=1,
            linewidth=0, antialiased=True, shade=False
        )

    # Ajuste de camera
    ax.set_title("CSFW", fontweight='bold', fontsize=16)
    ax.set_xlabel(r'$z$ ($\mu m$)')
    ax.set_ylabel(r'$y$ ($\mu m$)')
    ax.set_zlabel(r'$x$ ($\mu m$)')

    ax.set_xlim([0, L * 1e6])
    ax.set_ylim([-2 * espaco_y * 1e6, 2 * espaco_y * 1e6])
    ax.set_zlim([-R / 2 * 1e6, R / 2 * 1e6])

    # Remoção do preenchimento dos planos de fundo para destaque das FLs
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

# Ajuste para visao em perspectiva
    ax.view_init(elev=20, azim=-45)
    plt.tight_layout()
    plt.show()

# Ajuste para visao frontal (plano xz)
    #ax.view_init(elev=0, azim=-90)
    #plt.tight_layout()
    #plt.show()

simular_cvfw_3D("F=MA.png")