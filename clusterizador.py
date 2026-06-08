from PIL import Image
import numpy as np
from sklearn.cluster import KMeans


def rgb_para_hsi(imagem):
    imagem = imagem.astype(np.float32) / 255.0

    R = imagem[:, :, 0]
    G = imagem[:, :, 1]
    B = imagem[:, :, 2]

    I = (R + G + B) / 3

    minimo = np.minimum(np.minimum(R, G), B)
    S = 1 - (minimo / (I + 1e-6))

    numerador = 0.5 * ((R - G) + (R - B))
    denominador = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-6

    angulo = np.arccos(np.clip(numerador / denominador, -1, 1))

    H = np.where(B <= G, angulo, 2 * np.pi - angulo)
    H = H / (2 * np.pi)

    return np.dstack((H, S, I))


def hsi_para_rgb(hsi):
    H = hsi[:, :, 0] * 2 * np.pi
    S = hsi[:, :, 1]
    I = hsi[:, :, 2]

    R = np.zeros_like(H)
    G = np.zeros_like(H)
    B = np.zeros_like(H)

    setor1 = (H < 2 * np.pi / 3)

    B[setor1] = I[setor1] * (1 - S[setor1])

    R[setor1] = I[setor1] * (
        1 +
        (S[setor1] * np.cos(H[setor1])) /
        (np.cos(np.pi / 3 - H[setor1]) + 1e-6)
    )

    G[setor1] = 3 * I[setor1] - R[setor1] - B[setor1]

    setor2 = (H >= 2 * np.pi / 3) & (H < 4 * np.pi / 3)

    h2 = H[setor2] - 2 * np.pi / 3

    R[setor2] = I[setor2] * (1 - S[setor2])

    G[setor2] = I[setor2] * (
        1 +
        (S[setor2] * np.cos(h2)) /
        (np.cos(np.pi / 3 - h2) + 1e-6)
    )

    B[setor2] = 3 * I[setor2] - R[setor2] - G[setor2]

    setor3 = H >= 4 * np.pi / 3

    h3 = H[setor3] - 4 * np.pi / 3

    G[setor3] = I[setor3] * (1 - S[setor3])

    B[setor3] = I[setor3] * (
        1 +
        (S[setor3] * np.cos(h3)) /
        (np.cos(np.pi / 3 - h3) + 1e-6)
    )

    R[setor3] = 3 * I[setor3] - G[setor3] - B[setor3]

    rgb = np.dstack((R, G, B))
    rgb = np.clip(rgb, 0, 1)

    return (rgb * 255).astype(np.uint8)


# ==================================================
# PROGRAMA PRINCIPAL
# ==================================================

imagem = Image.open("img_original.jpeg").convert("RGB")

dados_rgb = np.array(imagem)

print("Imagem carregada.")
print("Tamanho:", dados_rgb.shape)

# RGB -> HSI
dados_hsi = rgb_para_hsi(dados_rgb)

print("Conversão RGB -> HSI concluída.")

# ==========================================
# Clusterização
# ==========================================

pixels = dados_hsi.reshape(-1, 3)

NUM_CLUSTERS = 8

kmeans = KMeans(
    n_clusters=NUM_CLUSTERS,
    random_state=42,
    n_init=10
)

rotulos = kmeans.fit_predict(pixels)

centros = kmeans.cluster_centers_

print("Clusterização concluída.")

# ==========================================
# Ordenar clusters pela frequência
# ==========================================

frequencia = np.bincount(rotulos)

ordem = np.argsort(frequencia)[::-1]

print("\nRanking das cores:")

for pos, cluster in enumerate(ordem):
    print(
        f"{pos+1}º lugar -> "
        f"Cluster {cluster} "
        f"({frequencia[cluster]} pixels)"
    )

# ==========================================
# Gerar imagens progressivas
# ==========================================

for quantidade in range(1, NUM_CLUSTERS + 1):

    print(f"\nGerando imagem_{quantidade}.png")

    clusters_ativos = ordem[:quantidade]

    nova_hsi = np.zeros_like(pixels)

    mascara = np.isin(rotulos, clusters_ativos)

    nova_hsi[mascara] = centros[rotulos[mascara]]

    nova_hsi = nova_hsi.reshape(dados_hsi.shape)

    # HSI -> RGB
    resultado_rgb = hsi_para_rgb(nova_hsi)

    resultado = Image.fromarray(resultado_rgb)

    nome_arquivo = f"imagem_{quantidade}.png"

    resultado.save(nome_arquivo)

print("\nProcessamento concluído!")
print("Arquivos gerados:")

for i in range(1, NUM_CLUSTERS + 1):
    print(f"imagem_{i}.png")