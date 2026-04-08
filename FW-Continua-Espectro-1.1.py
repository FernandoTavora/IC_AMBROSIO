import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter


def analise_espectro_cvfw(img_path):
    # Mesmos dados do código de frozen wave continua
    L, R = 100e-6, 30e-6
    lam = 632.8e-9
    k = 2 * np.pi / lam
    Q = 0.80 * k
    kpQ = np.sqrt(k ** 2 - Q ** 2)

    n_min = int(-L * k / np.pi)
    n_vec = np.arange(n_min, 0)
    zs = np.sort(-2 * np.pi * n_vec / (2 * k))
    nz = len(zs)

    dx_min = 4.81 / kpQ
    nx = int(R / dx_min)

    img = Image.open(img_path).convert('L').resize((nz, nx))
    f_mat = np.array(img) / 255.0

    if f_mat.mean() > 0.5:
        f_mat = 1.0 - f_mat

    # Correção de orientação: flip vertical para bater com origin='lower'
    f_mat = np.flipud(f_mat)
    f_mat = gaussian_filter(f_mat, sigma=0.8)

    # Calculo do espectro
    nb = 600
    kzk = np.linspace(-1.2, 1.2, nb)
    beta = kzk * k

    # Matriz espectral
    S2D = np.zeros((nx, nb), dtype=complex)
    # Matriz de fase para transformada discreta
    P = np.exp(1j * beta[:, None] * zs[None, :])

    for p in range(nx):
        env = f_mat[p, :]
        if env.sum() < 0.01: continue

        # Coeficientes an da fw
        an = env * np.exp(-1j * Q * zs)
        S2D[p, :] = P @ an

    # Médias para plot
    S_abs = np.abs(S2D)
    spec_avg = np.mean(S_abs, axis=0)
    spec_max = np.max(S_abs, axis=0)

# Plot
    plt.figure(figsize=(12, 8))

    # Alvo Fisico
    plt.subplot(211)
    ext = [0, L * 1e6, -R / 2 * 1e6, R / 2 * 1e6]
    plt.imshow(f_mat, extent=ext, aspect='auto', cmap='gray', origin='lower')
    plt.title(r"Alvo Físico $F(x, z)$")
    plt.ylabel(r"x ($\mu$m)")

    # Espectro
    plt.subplot(212)
    plt.plot(kzk, spec_avg, 'navy', lw=2, label='Espectro Médio')
    plt.fill_between(kzk, 0, spec_max, color='royalblue', alpha=0.2, label='Envelope Máximo')

    # Linhas para referencia
    plt.axvline(0.8, color='orange', ls='-', label=r'Portadora $Q=0.8k$')
    plt.axvline(1.0, color='red', ls='--', alpha=0.5, label='Limite Evanescente')
    plt.axvline(-1.0, color='red', ls='--', alpha=0.5)

    plt.title(r"Espectro Longitudinal $|S(k_z)|$")
    plt.xlabel(r"$k_z / k$")
    plt.ylabel("Magnitude")
    plt.xlim(-1, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.show()

analise_espectro_cvfw("F=MA.png")
