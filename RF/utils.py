"""
==========================================================
UTILS.PY
Funções auxiliares para o projeto Random Forest Fake News
==========================================================
"""

from pathlib import Path
import pandas as pd


# ==========================================================
# LEITURA DO DATASET
# ==========================================================

from pathlib import Path
import pandas as pd


def carregar_dataset(caminho_dataset, coluna_texto="Notícia", rotulo_excel=1):
    textos = []
    classes = []
    arquivos = []

    caminho = Path(caminho_dataset)

    print("=" * 60)
    print("LENDO DATASET")
    print("=" * 60)

    # ======================================================
    # CASO 1: ARQUIVO EXCEL
    # ======================================================

    if caminho.is_file() and caminho.suffix in [".xlsx", ".xls"]:

        print(f"Lendo arquivo Excel: {caminho}")

        df = pd.read_excel(caminho)

        if coluna_texto not in df.columns:
            raise ValueError(
                f"A coluna '{coluna_texto}' não foi encontrada no Excel. "
                f"Colunas disponíveis: {list(df.columns)}"
            )

        for i, texto in enumerate(df[coluna_texto].dropna()):

            textos.append(str(texto))
            classes.append(rotulo_excel)
            arquivos.append(f"linha_{i + 1}")

        print()
        print(f"Total de notícias : {len(textos)}")
        print(f"Rótulo aplicado   : {rotulo_excel}")
        print()

        return textos, classes, arquivos

    # ======================================================
    # CASO 2: PASTA COM FAKE / TRUE
    # ======================================================

    pasta_fake = caminho / "fake"
    pasta_true = caminho / "true"

    if not pasta_fake.exists() or not pasta_true.exists():
        raise FileNotFoundError(
            "Estrutura inválida. Use uma pasta contendo 'fake' e 'true' "
            "ou informe um arquivo Excel .xlsx."
        )

    print(f"Lendo: {pasta_fake}")

    for arquivo in sorted(pasta_fake.glob("*.txt")):

        with open(arquivo, "r", encoding="utf-8") as f:
            textos.append(f.read())
            classes.append(1)
            arquivos.append(arquivo.name)

    print(f"Lendo: {pasta_true}")

    for arquivo in sorted(pasta_true.glob("*.txt")):

        with open(arquivo, "r", encoding="utf-8") as f:
            textos.append(f.read())
            classes.append(0)
            arquivos.append(arquivo.name)

    print()
    print(f"Total de notícias : {len(textos)}")
    print(f"Fake              : {classes.count(1)}")
    print(f"Verdadeiras       : {classes.count(0)}")
    print()

    return textos, classes, arquivos

# ==========================================================
# ESTATÍSTICAS DO DATASET
# ==========================================================

def estatisticas_dataset(textos):

    tamanhos = [len(t) for t in textos]

    print("=" * 60)
    print("ESTATÍSTICAS DO DATASET")
    print("=" * 60)

    print(f"Menor notícia : {min(tamanhos)} caracteres")
    print(f"Maior notícia : {max(tamanhos)} caracteres")
    print(f"Média         : {sum(tamanhos)/len(tamanhos):.2f} caracteres")
    print()

    return tamanhos


# ==========================================================
# SALVAR DATAFRAME
# ==========================================================

def salvar_csv(df, caminho):

    df.to_csv(

        caminho,

        index=False,

        encoding="utf-8"

    )

    print(f"Arquivo salvo: {caminho}")


# ==========================================================
# RELATÓRIO TXT
# ==========================================================

def salvar_relatorio(caminho, texto):

    with open(

        caminho,

        "w",

        encoding="utf-8"

    ) as f:

        f.write(texto)

    print(f"Relatório salvo: {caminho}")


# ==========================================================
# MOSTRAR RESULTADOS
# ==========================================================

def imprimir_metricas(

    accuracy,

    precision,

    recall,

    f1,

    auc

):

    print("=" * 60)
    print("MÉTRICAS")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")

    print()


# ==========================================================
# SALVAR IMPORTÂNCIA DAS FEATURES
# ==========================================================

def salvar_importancias(

    feature_names,

    importancias,

    quantidade=100,

    caminho="resultados/importancia_features.csv"

):

    dados = []

    indices = importancias.argsort()[::-1]

    for i in indices[:quantidade]:

        dados.append({

            "Palavra": feature_names[i],

            "Importância": importancias[i]

        })

    df = pd.DataFrame(dados)

    salvar_csv(df, caminho)