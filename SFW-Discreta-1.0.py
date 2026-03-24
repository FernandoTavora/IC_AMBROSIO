import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def simular_sfw_discreta(caminho_img):
    if not os.path.exists(caminho_img):
        print(f"Erro: Arquivo não encontrado em {caminho_img}")
        return

    # 1. DIGITALIZAÇÃO
    img = Image.open(caminho_img).convert('L')

    # Mantendo a resoluçao 276x75 (q nem no artigo)
    img = img.resize((276, 75))
    img_data = np.array(img)

    F_matrix = np.flipud(img_data < 200).astype(float) # fundo Preto

    # 2. PARÂMETROS FÍSICOS OTIMIZADOS (baseado na Figura 5 do artigo)
    L = 0.06  # 6 cm
    R = 0.0163  # 1.63 cm
    lambda_0 = 632.8e-9
    k = 2 * np.pi / lambda_0

    # Aumento do N para implica melhor definição longitudinal.
    # Alem de q, o Q deve ser reduzido levemente (menos paraxial) para acomodar os feixes.
    N = 50
    Q = 0.995 * k

    num_x, num_z = F_matrix.shape
    z_axis = np.linspace(0, L, num_z)
    x_axis = np.linspace(-R / 2, R / 2, num_x) 

    # Matriz resultado
    psi_sfw = np.zeros((num_x, num_z), dtype=complex)

    # 3. SUPERPOSIÇÃO
    for p in range(num_x):  #for pq permite que sejam um feixe por vez (cada linha)/ superposiçao
        F_z = F_matrix[p, :]

        # Se a linha for toda vazia (preta), pular para economizar processamento
        if np.sum(F_z) == 0:
            continue

        for q in range(-N, N + 1):
            beta_q = Q + (2 * np.pi * q / L)

            # Condiçao fisica: beta_q deve ser real e propagante
            if 0 < beta_q <= k:
                # Coeficiente A_q (eq. 6 do artigo)
                term = np.exp(1j * (2 * np.pi * q / L) * z_axis)
                A_q = (1 / L) * np.trapezoid(F_z * term, z_axis)

                # Superposiçap no plano da imagem (eq. 4)
                psi_sfw[p, :] += A_q * np.exp(-1j * beta_q * z_axis)

    intensidade = np.abs(psi_sfw) ** 2

    # 4. VISUALIZAÇÃO 2D (mapa de calor)
    plt.figure(figsize=(10, 6))

    extent = [0, L * 100, -R / 2 * 100, R / 2 * 100]

    # origin='lower' coloca o indice [0,0] da matriz no canto inferior esquerdo
    # Como foi usado flipud na entrada, a imagem ficara correta
    plt.imshow(intensidade, extent=extent, aspect='auto', cmap='hot', origin='lower')

    plt.colorbar(label=r'Intensidade $|\Psi_{SFW}|^2$')
    plt.title(f'Surface Frozen Wave (N={N}, Q={Q / k:.3f}k)\n')
    plt.xlabel('Eixo Longitudinal z (cm)')
    plt.ylabel('Eixo Transversal x (cm)')
    plt.tight_layout()
    plt.show()


# 5. EXECUÇÃO
caminho = r"C:\Users\ferna\PycharmProjects\IC-AMBROSIO\eesc_sfw.png"
simular_sfw_discreta(caminho)
