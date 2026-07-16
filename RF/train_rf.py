"""
==========================================================
RANDOM FOREST PARA DETECÇÃO DE FAKE NEWS
Treinamento Completo
Autor: Gabriel
==========================================================
"""

import os
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split


# ==========================================================
# CONFIGURAÇÕES DA FLORESTA
# ==========================================================

N_ESTIMATORS = 300

CRITERION = "gini"

MAX_DEPTH = None

MIN_SAMPLES_SPLIT = 2

MIN_SAMPLES_LEAF = 1

MAX_FEATURES = "sqrt"

BOOTSTRAP = True

OOB_SCORE = True

RANDOM_STATE = 42

N_JOBS = -1

VERBOSE = 1


# ==========================================================
# CAMINHOS
# ==========================================================

TRAIN_FAKE = "dataset/treino/fake"

TRAIN_TRUE = "dataset/treino/true"

MODEL_FOLDER = "modelos"

RESULT_FOLDER = "resultados"

TREE_FOLDER = "arvores"


os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(TREE_FOLDER, exist_ok=True)


# ==========================================================
# LEITURA DO DATASET
# ==========================================================

from RF.utils import carregar_dataset
from RF.utils import estatisticas_dataset
from RF.utils import salvar_csv
from RF.utils import salvar_relatorio
from RF.utils import salvar_importancias

textos, classes, arquivos = carregar_dataset("dataset/treino")

estatisticas_dataset(textos)


print("=" * 60)
print("CARREGANDO DATASET")
print("=" * 60)

from RF.utils import carregar_dataset

textos, classes, arquivos = carregar_dataset("dataset/treino")

print()

print("Quantidade de notícias:", len(textos))

print("Fake:", sum(classes))

print("True:", len(classes)-sum(classes))

print()


# ==========================================================
# TF-IDF
# ==========================================================

print("=" * 60)
print("VETORIZAÇÃO TF-IDF")
print("=" * 60)

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words=None,
    max_features=30000
)

X = vectorizer.fit_transform(textos)

y = np.array(classes)

print("Shape:", X.shape)

print()


# ==========================================================
# CONSTRUÇÃO DA FLORESTA
# ==========================================================

print("=" * 60)
print("CONFIGURAÇÃO DA FLORESTA")
print("=" * 60)

rf = RandomForestClassifier(

    n_estimators=N_ESTIMATORS,

    criterion=CRITERION,

    max_depth=MAX_DEPTH,

    min_samples_split=MIN_SAMPLES_SPLIT,

    min_samples_leaf=MIN_SAMPLES_LEAF,

    max_features=MAX_FEATURES,

    bootstrap=BOOTSTRAP,

    oob_score=OOB_SCORE,

    random_state=RANDOM_STATE,

    n_jobs=N_JOBS,

    verbose=VERBOSE
)

print(rf)

print()

print("Treinando...")

rf.fit(X, y)

print()

print("Treinamento concluído.")

print()

print("=" * 60)
print("CONFIGURAÇÕES UTILIZADAS")
print("=" * 60)

print("Árvores................:", rf.n_estimators)

print("Critério...............:", rf.criterion)

print("Bootstrap..............:", rf.bootstrap)

print("Max Depth..............:", rf.max_depth)

print("Min Samples Split......:", rf.min_samples_split)

print("Min Samples Leaf.......:", rf.min_samples_leaf)

print("Max Features...........:", rf.max_features)

print("Random State...........:", rf.random_state)

print("OOB Score..............:", rf.oob_score_)

print()


# ==========================================================
# SALVAR
# ==========================================================

joblib.dump(rf,
            "modelos/random_forest.pkl")

joblib.dump(vectorizer,
            "modelos/vectorizer.pkl")

print("Modelo salvo.")

# ==========================================================
# ESTATÍSTICAS DAS ÁRVORES
# ==========================================================

from sklearn.tree import export_graphviz
import graphviz

print()
print("=" * 60)
print("ANÁLISE DA FLORESTA")
print("=" * 60)

dados_arvores = []

for indice, arvore in enumerate(rf.estimators_):

    profundidade = arvore.tree_.max_depth

    nos = arvore.tree_.node_count

    folhas = arvore.tree_.n_leaves

    dados_arvores.append({

        "Árvore": indice + 1,

        "Profundidade": profundidade,

        "Nós": nos,

        "Folhas": folhas

    })

    print(f"Árvore {indice+1}")

    print(f"   Profundidade : {profundidade}")

    print(f"   Nós          : {nos}")

    print(f"   Folhas       : {folhas}")

    print()


df_arvores = pd.DataFrame(dados_arvores)

df_arvores.to_csv(
    "resultados/estatisticas_arvores.csv",
    index=False
)

print("Relatório salvo em resultados/estatisticas_arvores.csv")

# ==========================================================
# VALIDAÇÃO
# ==========================================================

print()
print("=" * 60)
print("VALIDAÇÃO")
print("=" * 60)

VALID_FAKE = "dataset/validacao/fake"
VALID_TRUE = "dataset/validacao/true"

textos_validacao, classes_validacao, arquivos = carregar_dataset(
    "dataset/validacao"
)

X_valid = vectorizer.transform(textos_validacao)

y_valid = np.array(classes_validacao)