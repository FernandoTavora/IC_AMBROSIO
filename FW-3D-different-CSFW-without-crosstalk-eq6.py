import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from scipy.special import j0, j1
from scipy.ndimage import gaussian_filter
from pathlib import Path
import os

def simular_cvfw_3D_multiplas_eq6(lista_img_paths):
    # Verificação de arquivos
    for path in lista_img_paths:
        if not os.path.exists(path):
            print(f"Erro: Arquivo não encontrado em {path}")
            return

    L = 100e-6
    R = 30e-6
    lambda_0 = 632.8e-9
    k = 2 * np.pi / lambda_0
    k_band = 2 * k
    Q = 0.80 * k
    k_rho_Q = np.sqrt(k ** 2 - Q ** 2)

    n_folhas = len(lista_img_paths)
    espaco_y = 35e-6

    limite_y = (n_folhas - 1) / 2 * espaco_y
    folhas_y = np.linspace(-limite_y, limite_y, n_folhas)

    n_min = int(-L * k / np.pi)
    n_vec = np.arange(n_min, 0, 1)

    z_samples = np.sort(-2 * np.pi * n_vec / k_band)
    nz_exact = len(z_samples)

    dx_min = 4.81 / k_rho_Q
    nx = int(R / dx_min)

    lista_f_matrices = []
    for img_path in lista_img_paths:
        img = Image.open(img_path).convert('L').resize((nz_exact, nx))
        img_data = np.array(img) / 255.0

        if img_data[0, 0] > 0.5:
            img_data = 1.0 - img_data

        f_mat = np.flipud(img_data)
        f_mat = gaussian_filter(f_mat, sigma=0.8)
        f_mat /= np.max(f_mat)
        lista_f_matrices.append(f_mat)

    nx_vis, nz_vis = 200, 500
    x_axis = np.linspace(-R / 2, R / 2, nx_vis)
    z_axis = np.linspace(0, L, nz_vis)

    Z_grid, X_grid = np.meshgrid(z_axis, x_axis)
    x0_fws = np.linspace(-R / 2, R / 2, nx)

    intensidade_planos = []

    N_beta = 400
    beta_array = np.linspace(-k, k, N_beta)
    d_beta = beta_array[1] - beta_array[0]

    exp_matrix = np.exp(1j * beta_array[:, None] * z_samples[None, :])

    # Folhas de luz
    for idx_plano, plano_y in enumerate(folhas_y):  # Plano de observação atual
        Ex_plano = np.zeros_like(Z_grid, dtype=complex)
        Ez_plano = np.zeros_like(Z_grid, dtype=complex)

        yp = plano_y

        f_matrix_atual = lista_f_matrices[idx_plano]

        for p in range(nx):
            f_envelope = f_matrix_atual[p, :]
            if np.sum(f_envelope) < 0.01:
                continue

            xp = x0_fws[p]
            A_n = f_envelope * np.exp(-1j * Q * z_samples)

            rho_g = np.sqrt((X_grid - xp) ** 2 + (plano_y - yp) ** 2)
            cos_phi = (X_grid - xp) / (rho_g + 1e-12)

            S_beta = exp_matrix @ A_n

            psi_3D = np.zeros_like(Z_grid, dtype=complex)
            ez_3D = np.zeros_like(Z_grid, dtype=complex)

            for idx_b, beta in enumerate(beta_array):
                if np.abs(S_beta[idx_b]) < 1e-6:
                    continue

                k_rho = np.sqrt(k ** 2 - beta ** 2)
                beta_safe = beta if np.abs(beta) > 1e-12 else 1e-12

                integrand_x = S_beta[idx_b] * j0(k_rho * rho_g) * np.exp(-1j * beta * Z_grid)
                psi_3D += integrand_x

                integrand_z = 1j * (k_rho / beta_safe) * j1(k_rho * rho_g) * cos_phi * integrand_x
                ez_3D += integrand_z

            psi_3D *= d_beta / (2 * k)
            ez_3D *= d_beta / (2 * k)

            Ex_plano += psi_3D
            Ez_plano += ez_3D

        i_total = np.abs(Ex_plano) ** 2 + np.abs(Ez_plano) ** 2
        intensidade_planos.append(i_total)
        print(f"Fatia {idx_plano + 1}/{n_folhas} processada.")

    # Renderização 3D
    i_max_global = np.max(intensidade_planos)

    plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i, y_plane in enumerate(folhas_y):
        i_norm = intensidade_planos[i] / i_max_global
        cores = plt.cm.inferno(i_norm)

        threshold = 0.20
        cores[..., 3] = np.where(i_norm > threshold, 0.9, 0.0)

        Y_surf = np.full_like(Z_grid, y_plane * 1e6)

        ax.plot_surface(
            Z_grid * 1e6, X_grid * 1e6, Y_surf,
            facecolors=cores,
            rstride=1, cstride=1,
            linewidth=0, antialiased=True, shade=False
        )

    # Ajuste de câmera
    ax.set_title("", fontweight='bold', fontsize=16)

    ax.set_xlabel(r'$z$ ($\mu m$)')
    ax.set_ylabel(r'$x$ ($\mu m$)')
    ax.set_zlabel(r'$y$ ($\mu m$)')

    ax.set_xlim([0, L * 1e6])
    ax.set_ylim([-R / 2 * 1e6, R / 2 * 1e6])
    ax.set_zlim([(-limite_y - espaco_y) * 1e6, (limite_y + espaco_y) * 1e6])

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    # Ajuste de angulo de visão
    ax.view_init(elev=20, azim=-45)
    plt.tight_layout()
    plt.show()

minhas_imagens = [
    "fatia_1.png",
    "fatia_2.png",
    "fatia_3.png",
    "fatia_4.png",
    "fatia_5.png"
]

simular_cvfw_3D_multiplas_eq6(minhas_imagens)