import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.special import j1
from scipy.ndimage import gaussian_filter

def fw_vec_sim(img_path):
    # Parâmetros
    L, R = 100e-6, 30e-6
    lam0 = 632.8e-9
    k = 2 * np.pi / lam0
    Q = 0.8 * k
    kpQ = np.sqrt(k ** 2 - Q ** 2)

    # Amostragem
    n_min = int(-L * k / np.pi)
    n_vec = np.arange(n_min, 0)
    zs = np.sort(-2 * np.pi * n_vec / (2 * k))
    nz = len(zs)

    # limite de difração
    dx_min = 4.81 / kpQ
    nx = int(R / dx_min)

    # Tratamento da imagem
    img = Image.open(img_path).convert('L').resize((nz, nx))
    f_mat = np.array(img) / 255.0

    # Inversão de cores se o fundo for branco
    if f_mat.mean() > 0.5:
        f_mat = 1.0 - f_mat

    # inverter para que a imagem seja reconstruida de forma "invertida"
    f_mat = np.flipud(f_mat)

    # Suavização para evitar frequencias infinitas
    f_mat = gaussian_filter(f_mat, sigma=0.8)
    f_mat /= f_mat.max()

    # Grid visual
    z_vis = np.linspace(0, L, 500)
    x_vis = np.linspace(-R / 2, R / 2, nx)

    # Kernel Sinc (feixe de mackinnon)
    sinc_ker = np.sinc((k / np.pi) * (z_vis[:, None] - zs[None, :]))

    Ex = np.zeros((nx, 500), dtype=complex)
    Ez = np.zeros((nx, 500), dtype=complex)

    # Loop de integração forma vetorial
    for p in range(nx):
        env = f_mat[p, :]
        if env.sum() < 0.05: continue

        an = env * np.exp(-1j * Q * zs)
        psi = sinc_ker @ an
        Ex[p, :] = psi

        rho = np.abs(x_vis[p])
        cp = np.sign(x_vis[p]) if x_vis[p] != 0 else 1.0

        ez_factor = 1j * (kpQ / Q) * j1(kpQ * rho) * cp
        Ez[p, :] = ez_factor * psi

    # Plot
    I_tot = np.abs(Ex) ** 2 + np.abs(Ez) ** 2
    ext = [0, L * 1e6, -R / 2 * 1e6, R / 2 * 1e6]

    plt.figure(figsize=(10, 8))

    plt.subplot(311)
    plt.imshow(I_tot, extent=ext, aspect='auto', cmap='inferno', origin='lower')
    plt.title("Intensidade Total")
    plt.colorbar(label="I")

    plt.subplot(312)
    plt.imshow(np.abs(Ex) ** 2, extent=ext, aspect='auto', cmap='inferno', origin='lower')
    plt.title("Componente Transversal |Ex|^2")
    plt.colorbar()

    plt.subplot(313)
    plt.imshow(np.abs(Ez) ** 2, extent=ext, aspect='auto', cmap='inferno', origin='lower')
    plt.title("Componente Longitudinal |Ez|^2")
    plt.xlabel(r"z ($\mu$m)")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

fw_vec_sim("F=MA.png")
