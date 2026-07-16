import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

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

from utils import carregar_excel_externo
from utils import salvar_relatorio


BASE_DIR = Path(__file__).resolve().parent

MODEL_FOLDER = BASE_DIR / "modelos"
RESULT_FOLDER = BASE_DIR / "resultados"

ARQUIVO_FAKE = BASE_DIR / "dataset" / "noticias_fake_ia.xlsx"
ARQUIVO_TRUE = BASE_DIR / "dataset" / "noticias_true.xlsx"

os.makedirs(RESULT_FOLDER, exist_ok=True)


print("=" * 60)
print("TESTE EXTERNO SVM")
print("=" * 60)

modelo = joblib.load(MODEL_FOLDER / "svm.pkl")
vectorizer = joblib.load(MODEL_FOLDER / "vectorizer_svm.pkl")

print("Modelo carregado com sucesso.\n")


textos, classes, arquivos = carregar_excel_externo(
    ARQUIVO_FAKE,
    ARQUIVO_TRUE
)

X_test = vectorizer.transform(textos)
y_test = np.array(classes)


print("=" * 60)
print("CLASSIFICANDO TESTE EXTERNO")
print("=" * 60)

predicoes = modelo.predict(X_test)

scores = modelo.decision_function(X_test)

prob_fake = 1 / (1 + np.exp(-scores))


accuracy = accuracy_score(y_test, predicoes)
precision = precision_score(y_test, predicoes)
recall = recall_score(y_test, predicoes)
f1 = f1_score(y_test, predicoes)

fpr, tpr, _ = roc_curve(y_test, prob_fake)
roc_auc = auc(fpr, tpr)


print()
print("=" * 60)
print("MÉTRICAS DO TESTE EXTERNO - SVM")
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

salvar_relatorio(
    RESULT_FOLDER / "classification_report_teste_externo_svm.txt",
    relatorio
)


cm = confusion_matrix(y_test, predicoes)

plt.figure(figsize=(6, 6))
plt.imshow(cm)
plt.title("Matriz de Confusão - Teste Externo SVM")
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
    RESULT_FOLDER / "confusion_matrix_teste_externo_svm.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


plt.figure(figsize=(7, 7))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Teste Externo SVM")
plt.legend()

plt.savefig(
    RESULT_FOLDER / "roc_curve_teste_externo_svm.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


dados = []

for i in range(len(textos)):
    classe_real = "Fake" if y_test[i] == 1 else "Verdadeira"
    classe_prevista = "Fake" if predicoes[i] == 1 else "Verdadeira"

    confianca = max(prob_fake[i], 1 - prob_fake[i]) * 100

    dados.append({
        "Arquivo/ID": arquivos[i],
        "Classe Real": classe_real,
        "Classe Prevista": classe_prevista,
        "Score SVM": round(scores[i], 6),
        "Distância Absoluta do Hiperplano": round(abs(scores[i]), 6),
        "Probabilidade Fake (%)": round(prob_fake[i] * 100, 2),
        "Confiança (%)": round(confianca, 2),
        "Resultado": "Acerto" if y_test[i] == predicoes[i] else "Erro",
        "Texto": textos[i]
    })


df_resultados = pd.DataFrame(dados)

df_resultados.to_excel(
    RESULT_FOLDER / "classificacoes_teste_externo_svm.xlsx",
    index=False
)

df_resultados.to_csv(
    RESULT_FOLDER / "classificacoes_teste_externo_svm.csv",
    index=False,
    sep=";",
    encoding="utf-8-sig"
)


total = len(y_test)
acertos = int((y_test == predicoes).sum())
erros = total - acertos

verdadeiras_corretas = int(cm[0][0])
verdadeiras_como_fake = int(cm[0][1])
fake_como_verdadeiras = int(cm[1][0])
fake_corretas = int(cm[1][1])


texto_relatorio = f"""
===============================
TESTE EXTERNO - SVM
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

- classification_report_teste_externo_svm.txt
- confusion_matrix_teste_externo_svm.png
- roc_curve_teste_externo_svm.png
- classificacoes_teste_externo_svm.xlsx
- classificacoes_teste_externo_svm.csv
"""

salvar_relatorio(
    RESULT_FOLDER / "resultado_teste_externo_svm.txt",
    texto_relatorio
)

print(texto_relatorio)
print("Teste externo SVM concluído.")