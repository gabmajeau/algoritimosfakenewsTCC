import os
import joblib
import numpy as np
import pandas as pd
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

# =========================
# CAMINHOS
# =========================

MODEL_FOLDER = "modelos"
RESULT_FOLDER = "resultados"

ARQUIVO_FAKE = "dataset/noticias_fake_ia.xlsx"
ARQUIVO_TRUE = "dataset/noticias_true.xlsx"

os.makedirs(RESULT_FOLDER, exist_ok=True)

# =========================
# CARREGAR MODELO
# =========================

print("=" * 60)
print("CARREGANDO MODELO RANDOM FOREST")
print("=" * 60)

modelo = joblib.load(os.path.join(MODEL_FOLDER, "random_forest.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_FOLDER, "vectorizer.pkl"))

print("Modelo carregado com sucesso.\n")

# =========================
# LER EXCELS
# =========================

print("=" * 60)
print("CARREGANDO DATASET EXTERNO")
print("=" * 60)

df_fake = pd.read_excel(ARQUIVO_FAKE)
df_true = pd.read_excel(ARQUIVO_TRUE)

# Coluna 1 = número
# Coluna 2 = texto da notícia
coluna_numero_fake = df_fake.columns[0]
coluna_texto_fake = df_fake.columns[1]

coluna_numero_true = df_true.columns[0]
coluna_texto_true = df_true.columns[1]

df_fake = df_fake.dropna(subset=[coluna_texto_fake])
df_true = df_true.dropna(subset=[coluna_texto_true])

textos = []
classes = []
arquivos = []

for _, linha in df_fake.iterrows():
    textos.append(str(linha[coluna_texto_fake]))
    classes.append(1)
    arquivos.append(f"fake_ia_{linha[coluna_numero_fake]}")

for _, linha in df_true.iterrows():
    textos.append(str(linha[coluna_texto_true]))
    classes.append(0)
    arquivos.append(f"true_{linha[coluna_numero_true]}")

y_test = np.array(classes)

print(f"Total de notícias: {len(textos)}")
print(f"Fake IA: {classes.count(1)}")
print(f"Verdadeiras: {classes.count(0)}")
print()

# =========================
# VETORIZAR
# =========================

X_test = vectorizer.transform(textos)

# =========================
# CLASSIFICAR
# =========================

print("=" * 60)
print("REALIZANDO TESTE EXTERNO")
print("=" * 60)

predicoes = modelo.predict(X_test)
probabilidades = modelo.predict_proba(X_test)

prob_fake = probabilidades[:, 1]

# =========================
# MÉTRICAS
# =========================

accuracy = accuracy_score(y_test, predicoes)
precision = precision_score(y_test, predicoes)
recall = recall_score(y_test, predicoes)
f1 = f1_score(y_test, predicoes)

fpr, tpr, _ = roc_curve(y_test, prob_fake)
roc_auc = auc(fpr, tpr)

print()
print("=" * 60)
print("MÉTRICAS DO TESTE EXTERNO")
print("=" * 60)
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"AUC      : {roc_auc:.4f}")
print()

relatorio = classification_report(
    y_test,
    predicoes,
    target_names=["Verdadeira", "Fake"]
)

print(relatorio)

with open("resultados/classification_report_teste_externo.txt", "w", encoding="utf-8") as f:
    f.write(relatorio)

# =========================
# MATRIZ DE CONFUSÃO
# =========================

cm = confusion_matrix(y_test, predicoes)

plt.figure(figsize=(6, 6))
plt.imshow(cm)
plt.title("Matriz de Confusão - Teste Externo")
plt.colorbar()

plt.xticks([0, 1], ["Verdadeira", "Fake"])
plt.yticks([0, 1], ["Verdadeira", "Fake"])

plt.xlabel("Previsto")
plt.ylabel("Real")

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=15
        )

plt.savefig(
    "resultados/confusion_matrix_teste_externo.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =========================
# ROC CURVE
# =========================

plt.figure(figsize=(7, 7))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Teste Externo")
plt.legend()

plt.savefig(
    "resultados/roc_curve_teste_externo.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =========================
# EXPORTAR CLASSIFICAÇÕES
# =========================

dados = []

for i in range(len(textos)):
    classe_real = "Fake" if y_test[i] == 1 else "Verdadeira"
    classe_prevista = "Fake" if predicoes[i] == 1 else "Verdadeira"

    dados.append({
        "Arquivo/ID": arquivos[i],
        "Classe Real": classe_real,
        "Classe Prevista": classe_prevista,
        "Probabilidade Fake (%)": round(prob_fake[i] * 100, 2),
        "Confiança (%)": round(max(probabilidades[i]) * 100, 2),
        "Resultado": "Acerto" if y_test[i] == predicoes[i] else "Erro",
        "Texto": textos[i]
    })

df_resultados = pd.DataFrame(dados)

df_resultados.to_excel(
    "resultados/classificacoes_teste_externo.xlsx",
    index=False
)

df_resultados.to_csv(
    "resultados/classificacoes_teste_externo.csv",
    index=False,
    sep=";",
    encoding="utf-8-sig"
)

# =========================
# RELATÓRIO FINAL
# =========================

total = len(y_test)
acertos = int((y_test == predicoes).sum())
erros = total - acertos

verdadeiras_corretas = int(cm[0][0])
verdadeiras_como_fake = int(cm[0][1])
fake_como_verdadeiras = int(cm[1][0])
fake_corretas = int(cm[1][1])

texto_relatorio = f"""
===============================
TESTE EXTERNO - RANDOM FOREST
===============================

Dataset externo:
- Notícias verdadeiras: {classes.count(0)}
- Notícias fake sintéticas IA: {classes.count(1)}
- Total: {total}

Acertos: {acertos}
Erros: {erros}

Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1-score: {f1:.4f}
AUC: {roc_auc:.4f}

Matriz de Confusão:

Verdadeiras classificadas corretamente: {verdadeiras_corretas}
Verdadeiras classificadas como fake: {verdadeiras_como_fake}
Fake classificadas como verdadeiras: {fake_como_verdadeiras}
Fake classificadas corretamente: {fake_corretas}

Arquivos gerados:

- resultados/classification_report_teste_externo.txt
- resultados/confusion_matrix_teste_externo.png
- resultados/roc_curve_teste_externo.png
- resultados/classificacoes_teste_externo.xlsx
- resultados/classificacoes_teste_externo.csv
"""

with open("resultados/resultado_teste_externo.txt", "w", encoding="utf-8") as f:
    f.write(texto_relatorio)

print(texto_relatorio)
print("Teste externo concluído.")