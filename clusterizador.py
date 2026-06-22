from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans
import os


# ==================================================
# CONFIGURAÇÕES
# ==================================================

CAMINHO_IMAGEM = "img_original.jpeg"

# Faixa para o método do cotovelo
K_MIN = 2
K_MAX = 12

# Valores de K para geração de imagens progressivas
# (inspirado no projeto de referência)
KS_PROGRESSIVOS = [1, 2, 3, 4, 5, 6, 8, 12, 16, 32, 64, 128]

PASTA_SAIDA = "resultados"


# ==================================================
# CONVERSÕES
# ==================================================

def rgb_para_hsi(imagem):
    imagem = imagem.astype(np.float64) / 255.0

    R = imagem[:, :, 0]
    G = imagem[:, :, 1]
    B = imagem[:, :, 2]

    I = (R + G + B) / 3.0

    minimo = np.minimum(np.minimum(R, G), B)
    S = np.where(I > 0, 1 - (minimo / (I + 1e-10)), 0)

    numerador = 0.5 * ((R - G) + (R - B))
    denominador = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-10

    angulo = np.arccos(np.clip(numerador / denominador, -1, 1))

    H = np.where(B <= G, angulo, 2 * np.pi - angulo)
    H = H / (2 * np.pi)

    return np.stack([H, S, I], axis=2)


def hsi_para_rgb(hsi):
    H = hsi[:, :, 0] * 2 * np.pi
    S = hsi[:, :, 1]
    I = hsi[:, :, 2]

    R = np.zeros_like(I)
    G = np.zeros_like(I)
    B = np.zeros_like(I)

    # Setor RG: 0 <= H < 120°
    idx = (H >= 0) & (H < 2 * np.pi / 3)
    B[idx] = I[idx] * (1 - S[idx])
    R[idx] = I[idx] * (1 + S[idx] * np.cos(H[idx]) / (np.cos(np.pi / 3 - H[idx]) + 1e-10))
    G[idx] = 3 * I[idx] - (R[idx] + B[idx])

    # Setor GB: 120° <= H < 240°
    idx = (H >= 2 * np.pi / 3) & (H < 4 * np.pi / 3)
    H2 = H[idx] - 2 * np.pi / 3
    R[idx] = I[idx] * (1 - S[idx])
    G[idx] = I[idx] * (1 + S[idx] * np.cos(H2) / (np.cos(np.pi / 3 - H2) + 1e-10))
    B[idx] = 3 * I[idx] - (R[idx] + G[idx])

    # Setor BR: 240° <= H < 360°
    idx = (H >= 4 * np.pi / 3) & (H < 2 * np.pi)
    H3 = H[idx] - 4 * np.pi / 3
    G[idx] = I[idx] * (1 - S[idx])
    B[idx] = I[idx] * (1 + S[idx] * np.cos(H3) / (np.cos(np.pi / 3 - H3) + 1e-10))
    R[idx] = 3 * I[idx] - (G[idx] + B[idx])

    rgb = np.clip(np.stack([R, G, B], axis=2) * 255, 0, 255).astype(np.uint8)
    return rgb


# ==================================================
# MÉTODO DO COTOVELO
# ==================================================

def calcular_cotovelo(pixels, k_min, k_max):
    print(f"\nCalculando cotovelo de K={k_min} até K={k_max}...")

    ks = list(range(k_min, k_max + 1))
    inercias = []

    for k in ks:
        print(f"  Testando K={k}...", end="\r")
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(pixels)
        inercias.append(km.inertia_)

    print(f"  Cotovelo calculado! ({len(ks)} valores testados)      ")
    return ks, inercias


def detectar_cotovelo(ks, inercias):
    pontos = np.array(list(zip(ks, inercias)), dtype=np.float64)

    # Normaliza para colocar X e Y na mesma escala
    pontos[:, 0] = (pontos[:, 0] - pontos[:, 0].min()) / (pontos[:, 0].max() - pontos[:, 0].min())
    pontos[:, 1] = (pontos[:, 1] - pontos[:, 1].min()) / (pontos[:, 1].max() - pontos[:, 1].min())

    # Linha do primeiro ao último ponto
    inicio = pontos[0]
    fim = pontos[-1]
    direcao = fim - inicio

    # Distância de cada ponto à linha
    distancias = []
    for ponto in pontos:
        t = np.dot(ponto - inicio, direcao) / np.dot(direcao, direcao)
        projecao = inicio + t * direcao
        distancias.append(np.linalg.norm(ponto - projecao))

    idx_cotovelo = np.argmax(distancias)
    return ks[idx_cotovelo]


def plotar_cotovelo(ks, inercias, k_ideal, pasta):
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(ks, inercias, marker='o', linewidth=2,
            color='steelblue', markersize=6, label='Inércia')

    ax.axvline(x=k_ideal, color='tomato', linestyle='--', linewidth=1.8,
               label=f'Cotovelo detectado: K={k_ideal}')

    ax.scatter([k_ideal], [inercias[ks.index(k_ideal)]],
               color='tomato', s=120, zorder=5)

    ax.set_xlabel('Número de clusters (K)', fontsize=12)
    ax.set_ylabel('Inércia (soma das distâncias²)', fontsize=12)
    ax.set_title('Método do Cotovelo — Escolha do K ideal', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    caminho = os.path.join(pasta, "grafico_cotovelo.png")
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close()
    print(f"Gráfico do cotovelo salvo: {caminho}")
    return caminho


# ==================================================
# CLUSTERIZAÇÃO E GERAÇÃO DE IMAGEM
# ==================================================

def clusterizar(pixels, n_clusters):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rotulos = km.fit_predict(pixels)
    centros = km.cluster_centers_
    return rotulos, centros


def gerar_imagem_quantizada(pixels, rotulos, centros, shape):
    pixels_quantizados = centros[rotulos]
    hsi_quantizado = pixels_quantizados.reshape(shape)
    return hsi_para_rgb(hsi_quantizado)


# ==================================================
# GRÁFICO: COMPARATIVO DE KS PROGRESSIVOS
# ==================================================

def plotar_comparativo(imagem_original, resultados_por_k, pasta):
    total = len(resultados_por_k) + 1   # +1 para a original
    colunas = 4
    linhas = (total + colunas - 1) // colunas

    fig, eixos = plt.subplots(linhas, colunas,
                              figsize=(colunas * 4, linhas * 3.5))
    eixos = eixos.flatten()

    # Original
    eixos[0].imshow(imagem_original)
    eixos[0].set_title("Original", fontsize=11, fontweight='bold')
    eixos[0].axis('off')

    for i, (k, img) in enumerate(resultados_por_k, start=1):
        eixos[i].imshow(img)
        eixos[i].set_title(f"K = {k}", fontsize=11)
        eixos[i].axis('off')

    # Esconde eixos extras
    for j in range(total, len(eixos)):
        eixos[j].axis('off')

    fig.suptitle("Clusterização progressiva — RGB → HSI → K-Means → RGB",
                 fontsize=13, fontweight='bold', y=1.01)

    caminho = os.path.join(pasta, "comparativo_ks.png")
    plt.tight_layout()
    plt.savefig(caminho, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Painel comparativo salvo: {caminho}")
    return caminho


# ==================================================
# GRÁFICO: PALETA DE CORES DOS CLUSTERS
# ==================================================

def plotar_paleta(centros_hsi, rotulos, k, pasta):
    frequencia = np.bincount(rotulos)
    ordem = np.argsort(frequencia)[::-1]

    # Monta um array HSI de 1×k para converter para RGB
    paleta_hsi = np.zeros((1, k, 3))
    for i, c in enumerate(ordem):
        paleta_hsi[0, i] = centros_hsi[c]
    paleta_rgb = hsi_para_rgb(paleta_hsi)

    fig, ax = plt.subplots(figsize=(max(k * 0.6, 6), 2.2))

    for i in range(k):
        cor = paleta_rgb[0, i] / 255.0
        proporcao = frequencia[ordem[i]] / len(rotulos) * 100
        ax.bar(i, 1, color=cor, edgecolor='white', linewidth=0.5)
        ax.text(i, -0.15, f"{proporcao:.1f}%",
                ha='center', va='top', fontsize=7, color='#333333')

    ax.set_xlim(-0.5, k - 0.5)
    ax.set_ylim(-0.3, 1.05)
    ax.axis('off')
    ax.set_title(f"Paleta de cores (K={k}) — ordenada por frequência",
                 fontsize=11, fontweight='bold')

    caminho = os.path.join(pasta, f"paleta_k{k}.png")
    plt.tight_layout()
    plt.savefig(caminho, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Paleta K={k} salva: {caminho}")
    return caminho


# ==================================================
# PROGRAMA PRINCIPAL
# ==================================================

if __name__ == "__main__":

    os.makedirs(PASTA_SAIDA, exist_ok=True)

    # ── 1. Carrega a imagem ──────────────────────────────────────
    imagem = Image.open(CAMINHO_IMAGEM).convert("RGB")
    dados_rgb = np.array(imagem)
    print(f"Imagem carregada: {dados_rgb.shape}")

    # ── 2. RGB → HSI ─────────────────────────────────────────────
    dados_hsi = rgb_para_hsi(dados_rgb)
    pixels = dados_hsi.reshape(-1, 3)
    print("Conversão RGB → HSI concluída.")

    # ── 3. Método do cotovelo ─────────────────────────────────────
    ks, inercias = calcular_cotovelo(pixels, K_MIN, K_MAX)
    k_ideal = detectar_cotovelo(ks, inercias)
    print(f"\nK ideal detectado pelo cotovelo: {k_ideal}")
    plotar_cotovelo(ks, inercias, k_ideal, PASTA_SAIDA)

    # ── 4. Imagens progressivas (vários Ks) ──────────────────────
    print(f"\nGerando imagens para Ks: {KS_PROGRESSIVOS}")

    resultados_por_k = []

    for k in KS_PROGRESSIVOS:
        print(f"  K={k}...", end="\r")
        rotulos, centros = clusterizar(pixels, k)
        img_rgb = gerar_imagem_quantizada(pixels, rotulos, centros, dados_hsi.shape)
        resultados_por_k.append((k, img_rgb))

        # Salva individualmente
        nome = os.path.join(PASTA_SAIDA, f"imagem_k{k}.png")
        Image.fromarray(img_rgb).save(nome)

    print(f"  Imagens individuais salvas em '{PASTA_SAIDA}/'      ")

    # ── 5. Painel comparativo ─────────────────────────────────────
    plotar_comparativo(dados_rgb, resultados_por_k, PASTA_SAIDA)

    # ── 6. Paleta do K ideal ──────────────────────────────────────
    print(f"\nGerando paleta para K ideal ({k_ideal})...")
    rotulos_ideal, centros_ideal = clusterizar(pixels, k_ideal)
    plotar_paleta(centros_ideal, rotulos_ideal, k_ideal, PASTA_SAIDA)

    # ── 7. Resumo final ───────────────────────────────────────────
    print("\n" + "=" * 50)
    print("PROCESSAMENTO CONCLUÍDO")
    print("=" * 50)
    print(f"K ideal (cotovelo): {k_ideal}")
    print(f"\nArquivos gerados em '{PASTA_SAIDA}/':")
    for arquivo in sorted(os.listdir(PASTA_SAIDA)):
        print(f"  {arquivo}")