"""
==========================================================
VALIDAÇÃO DO RANDOM FOREST
==========================================================
"""

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

from RF.utils import (
    carregar_dataset,
    estatisticas_dataset,
    salvar_relatorio,
    imprimir_metricas
)

# ==========================================================
# PASTAS
# ==========================================================

VALID_FOLDER = "dataset/validacao"

RESULT_FOLDER = "resultados"

MODEL_FOLDER = "modelos"

os.makedirs(RESULT_FOLDER, exist_ok=True)

# ==========================================================
# CARREGAR MODELO
# ==========================================================

print("="*60)
print("CARREGANDO MODELO")
print("="*60)

rf = joblib.load(
    os.path.join(
        MODEL_FOLDER,
        "random_forest.pkl"
    )
)

vectorizer = joblib.load(
    os.path.join(
        MODEL_FOLDER,
        "vectorizer.pkl"
    )
)

print("Modelo carregado com sucesso.\n")

# ==========================================================
# CARREGAR DATASET
# ==========================================================

textos, classes, arquivos = carregar_dataset(
    VALID_FOLDER
)

estatisticas_dataset(textos)

# ==========================================================
# TF-IDF
# ==========================================================

print("="*60)
print("VETORIZANDO")
print("="*60)

X_valid = vectorizer.transform(textos)

y_valid = np.array(classes)

print("Shape:", X_valid.shape)
print()

# ==========================================================
# CLASSIFICAÇÃO
# ==========================================================

print("="*60)
print("CLASSIFICANDO")
print("="*60)

predicoes = rf.predict(X_valid)

probabilidades = rf.predict_proba(X_valid)

print("Classificação concluída.\n")

import pandas as pd

print("=" * 60)
print("ANÁLISE DAS NOTÍCIAS")
print("=" * 60)

dados = []

for i in range(len(textos)):

    classe_real = "Fake" if y_valid[i] == 1 else "Verdadeira"

    classe_prevista = "Fake" if predicoes[i] == 1 else "Verdadeira"

    prob_fake = probabilidades[i][1] * 100

    confianca = max(probabilidades[i]) * 100

    acertou = classe_real == classe_prevista

    dados.append({

        "Arquivo": arquivos[i],

        "Classe Real": classe_real,

        "Classe Prevista": classe_prevista,

        "Probabilidade Fake (%)": round(prob_fake,2),

        "Confiança (%)": round(confianca,2),

        "Resultado": "✔" if acertou else "✘"

    })

    print()

    print("="*40)

    print(arquivos[i])

    print("="*40)

    print("Classe Real:",classe_real)

    print("Classe Prevista:",classe_prevista)

    print(f"Probabilidade Fake: {prob_fake:.2f}%")

    print(f"Confiança: {confianca:.2f}%")

    print("Resultado:", "✔ ACERTO" if acertou else "✘ ERRO")

df = pd.DataFrame(dados)

df.to_csv(
    "resultados/classificacoes.csv",
    index=False,
    sep=";",
    encoding="utf-8-sig"
)

print()

print("CSV salvo.")

print()

print("="*60)

print("ERROS DA FLORESTA")

print("="*60)

erros = df[df["Resultado"]=="✘"]

print(erros)

erros.to_csv(

    "resultados/erros.csv",

    index=False,

    encoding="utf-8"

)

print()

print("="*60)

print("ESTATÍSTICAS")

print("="*60)

print("Maior confiança :",df["Confiança (%)"].max())

print("Menor confiança :",df["Confiança (%)"].min())

print("Confiança média :",df["Confiança (%)"].mean())

print()

print("Maior probabilidade Fake :",df["Probabilidade Fake (%)"].max())

print("Menor probabilidade Fake :",df["Probabilidade Fake (%)"].min())

print("Probabilidade média :",df["Probabilidade Fake (%)"].mean())

# ==========================================================
# DECISION PATH - CAMINHO PERCORRIDO EM UMA ÁRVORE
# ==========================================================

def explicar_noticia(indice_noticia, indice_arvore=0):

    print()
    print("=" * 70)
    print("EXPLICAÇÃO DA CLASSIFICAÇÃO")
    print("=" * 70)

    vetor = X_valid[indice_noticia]

    texto_original = textos[indice_noticia]

    classe_real = y_valid[indice_noticia]

    classe_prevista = predicoes[indice_noticia]

    prob_fake = probabilidades[indice_noticia][1] * 100

    print(f"Arquivo: {arquivos[indice_noticia]}")
    print(f"Classe real: {'Fake' if classe_real == 1 else 'Verdadeira'}")
    print(f"Classe prevista: {'Fake' if classe_prevista == 1 else 'Verdadeira'}")
    print(f"Probabilidade de Fake: {prob_fake:.2f}%")

    # votação da floresta
    votos = []

    for arvore in rf.estimators_:

        voto = arvore.predict(vetor)[0]

        votos.append(voto)

    votos_fake = sum(votos)

    votos_true = len(votos) - votos_fake

    print()
    print("VOTAÇÃO DA FLORESTA")
    print(f"Fake: {votos_fake}")
    print(f"Verdadeira: {votos_true}")

    # árvore escolhida para explicação
    arvore = rf.estimators_[indice_arvore]

    node_indicator = arvore.decision_path(vetor)

    leaf_id = arvore.apply(vetor)

    feature_names = vectorizer.get_feature_names_out()

    caminho = node_indicator.indices

    print()
    print(f"CAMINHO NA ÁRVORE {indice_arvore + 1}")
    print("-" * 70)

    linhas_relatorio = []

    linhas_relatorio.append("EXPLICAÇÃO DA CLASSIFICAÇÃO\n")
    linhas_relatorio.append(f"Arquivo: {arquivos[indice_noticia]}\n")
    linhas_relatorio.append(f"Classe real: {'Fake' if classe_real == 1 else 'Verdadeira'}\n")
    linhas_relatorio.append(f"Classe prevista: {'Fake' if classe_prevista == 1 else 'Verdadeira'}\n")
    linhas_relatorio.append(f"Probabilidade de Fake: {prob_fake:.2f}%\n\n")
    linhas_relatorio.append("VOTAÇÃO DA FLORESTA\n")
    linhas_relatorio.append(f"Fake: {votos_fake}\n")
    linhas_relatorio.append(f"Verdadeira: {votos_true}\n\n")
    linhas_relatorio.append(f"CAMINHO NA ÁRVORE {indice_arvore + 1}\n")

    for node_id in caminho:

        if node_id == leaf_id[0]:

            valores = arvore.tree_.value[node_id][0]

            classe_folha = np.argmax(valores)

            texto_classe = "Fake" if classe_folha == 1 else "Verdadeira"

            print(f"Folha final: {node_id}")
            print(f"Classificação da folha: {texto_classe}")

            linhas_relatorio.append(f"\nFolha final: {node_id}\n")
            linhas_relatorio.append(f"Classificação da folha: {texto_classe}\n")

            continue

        feature_index = arvore.tree_.feature[node_id]

        threshold = arvore.tree_.threshold[node_id]

        nome_feature = feature_names[feature_index]

        valor_tfidf = vetor[0, feature_index]

        if valor_tfidf <= threshold:

            direcao = "ESQUERDA"

            comparacao = "<="

        else:

            direcao = "DIREITA"

            comparacao = ">"

        print()
        print(f"Nó: {node_id}")
        print(f"Palavra/atributo: {nome_feature}")
        print(f"Valor TF-IDF: {valor_tfidf:.6f}")
        print(f"Threshold: {threshold:.6f}")
        print(f"Decisão: {valor_tfidf:.6f} {comparacao} {threshold:.6f}")
        print(f"Caminho seguido: {direcao}")

        linhas_relatorio.append("\n")
        linhas_relatorio.append(f"Nó: {node_id}\n")
        linhas_relatorio.append(f"Palavra/atributo: {nome_feature}\n")
        linhas_relatorio.append(f"Valor TF-IDF: {valor_tfidf:.6f}\n")
        linhas_relatorio.append(f"Threshold: {threshold:.6f}\n")
        linhas_relatorio.append(f"Decisão: {valor_tfidf:.6f} {comparacao} {threshold:.6f}\n")
        linhas_relatorio.append(f"Caminho seguido: {direcao}\n")

    with open(
        f"resultados/explicacao_noticia_{indice_noticia + 1}.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.writelines(linhas_relatorio)

    print()
    print(f"Relatório salvo em resultados/explicacao_noticia_{indice_noticia + 1}.txt")


# ==========================================================
# GERAR EXPLICAÇÕES DE ALGUMAS NOTÍCIAS
# ==========================================================

explicar_noticia(0, 0)

if len(textos) > 1:
    explicar_noticia(1, 1)

if len(textos) > 2:
    explicar_noticia(2, 2)