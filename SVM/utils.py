from pathlib import Path
import pandas as pd


def carregar_dataset(caminho_dataset):
    textos = []
    classes = []
    arquivos = []

    caminho = Path(caminho_dataset)

    pasta_fake = caminho / "fake"
    pasta_true = caminho / "true"

    if not pasta_fake.exists() or not pasta_true.exists():
        raise FileNotFoundError(
            f"Estrutura inválida em {caminho}. "
            "Esperado: fake/ e true/"
        )

    print("=" * 60)
    print("LENDO DATASET")
    print("=" * 60)

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


def carregar_excel_externo(arquivo_fake, arquivo_true):
    textos = []
    classes = []
    arquivos = []

    df_fake = pd.read_excel(arquivo_fake)
    df_true = pd.read_excel(arquivo_true)

    coluna_numero_fake = df_fake.columns[0]
    coluna_texto_fake = df_fake.columns[1]

    coluna_numero_true = df_true.columns[0]
    coluna_texto_true = df_true.columns[1]

    df_fake = df_fake.dropna(subset=[coluna_texto_fake])
    df_true = df_true.dropna(subset=[coluna_texto_true])

    for _, linha in df_fake.iterrows():
        textos.append(str(linha[coluna_texto_fake]))
        classes.append(1)
        arquivos.append(f"fake_ia_{linha[coluna_numero_fake]}")

    for _, linha in df_true.iterrows():
        textos.append(str(linha[coluna_texto_true]))
        classes.append(0)
        arquivos.append(f"true_{linha[coluna_numero_true]}")

    print("=" * 60)
    print("DATASET EXTERNO")
    print("=" * 60)
    print(f"Total de notícias : {len(textos)}")
    print(f"Fake IA           : {classes.count(1)}")
    print(f"Verdadeiras       : {classes.count(0)}")
    print()

    return textos, classes, arquivos


def estatisticas_dataset(textos):
    tamanhos = [len(t) for t in textos]

    print("=" * 60)
    print("ESTATÍSTICAS DO DATASET")
    print("=" * 60)
    print(f"Menor notícia : {min(tamanhos)} caracteres")
    print(f"Maior notícia : {max(tamanhos)} caracteres")
    print(f"Média         : {sum(tamanhos) / len(tamanhos):.2f} caracteres")
    print()

    return tamanhos


def salvar_relatorio(caminho, texto):
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(texto)

    print(f"Relatório salvo: {caminho}")