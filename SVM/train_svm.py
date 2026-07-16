"""
==========================================================
SVM PARA DETECÇÃO DE FAKE NEWS
Treinamento
Autor: Gabriel
==========================================================
"""

import os
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

from utils import carregar_dataset
from utils import estatisticas_dataset
from utils import salvar_relatorio


BASE_DIR = Path(__file__).resolve().parent

DATASET_TREINO = BASE_DIR / "dataset" / "treino"
MODEL_FOLDER = BASE_DIR / "modelos"
RESULT_FOLDER = BASE_DIR / "resultados"

os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


print("=" * 60)
print("TREINAMENTO SVM")
print("=" * 60)

textos, classes, arquivos = carregar_dataset(DATASET_TREINO)

estatisticas_dataset(textos)

y = np.array(classes)


print("=" * 60)
print("VETORIZAÇÃO TF-IDF")
print("=" * 60)

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words=None,
    max_features=30000
)

X = vectorizer.fit_transform(textos)

print("Shape TF-IDF:", X.shape)
print()


print("=" * 60)
print("CONFIGURAÇÃO DO SVM")
print("=" * 60)

svm = LinearSVC(
    C=1.0,
    loss="squared_hinge",
    max_iter=5000,
    random_state=42
)

print(svm)
print()


print("=" * 60)
print("TREINANDO SVM")
print("=" * 60)

svm.fit(X, y)

print("Treinamento concluído.")
print()


print("=" * 60)
print("AVALIAÇÃO NO TREINO")
print("=" * 60)

pred_train = svm.predict(X)

accuracy_train = accuracy_score(y, pred_train)

relatorio_train = classification_report(
    y,
    pred_train,
    target_names=["Verdadeira", "Fake"]
)

print(f"Accuracy treino: {accuracy_train:.4f}")
print()
print(relatorio_train)


print("=" * 60)
print("PESOS DAS PALAVRAS")
print("=" * 60)

feature_names = vectorizer.get_feature_names_out()
coeficientes = svm.coef_[0]

indices_fake = np.argsort(coeficientes)[-30:][::-1]
indices_true = np.argsort(coeficientes)[:30]

dados_pesos = []

print("\nPalavras mais associadas à classe FAKE:\n")

for i in indices_fake:
    print(f"{feature_names[i]:25} {coeficientes[i]:.6f}")
    dados_pesos.append({
        "Classe associada": "Fake",
        "Palavra": feature_names[i],
        "Peso": coeficientes[i]
    })

print("\nPalavras mais associadas à classe VERDADEIRA:\n")

for i in indices_true:
    print(f"{feature_names[i]:25} {coeficientes[i]:.6f}")
    dados_pesos.append({
        "Classe associada": "Verdadeira",
        "Palavra": feature_names[i],
        "Peso": coeficientes[i]
    })


df_pesos = pd.DataFrame(dados_pesos)

df_pesos.to_csv(
    RESULT_FOLDER / "pesos_svm.csv",
    index=False,
    sep=";",
    encoding="utf-8-sig"
)


texto_relatorio = f"""
===============================
TREINAMENTO SVM
===============================

Total de notícias: {len(textos)}
Fake: {classes.count(1)}
Verdadeiras: {classes.count(0)}

Shape TF-IDF: {X.shape}

Configuração do SVM:
{svm}

Accuracy no treino: {accuracy_train:.4f}

Classification Report:

{relatorio_train}
"""

salvar_relatorio(
    RESULT_FOLDER / "resultado_treino_svm.txt",
    texto_relatorio
)


joblib.dump(svm, MODEL_FOLDER / "svm.pkl")
joblib.dump(vectorizer, MODEL_FOLDER / "vectorizer_svm.pkl")

print()
print("Modelo salvo em:", MODEL_FOLDER / "svm.pkl")
print("Vectorizer salvo em:", MODEL_FOLDER / "vectorizer_svm.pkl")