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

from RF.utils import carregar_dataset, salvar_relatorio, imprimir_metricas


TEST_FOLDER = "dataset/teste"
MODEL_FOLDER = "modelos"
RESULT_FOLDER = "resultados"

os.makedirs(RESULT_FOLDER, exist_ok=True)


print("=" * 60)
print("CARREGANDO MODELO FINAL")
print("=" * 60)

modelo = joblib.load(os.path.join(MODEL_FOLDER, "random_forest.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_FOLDER, "vectorizer.pkl"))

print("Modelo e vetorizador carregados.\n")


print("=" * 60)
print("CARREGANDO DATASET DE TESTE")
print("=" * 60)

textos, classes, arquivos = carregar_dataset(TEST_FOLDER)

X_test = vectorizer.transform(textos)
y_test = np.array(classes)


print("=" * 60)
print("REALIZANDO CLASSIFICAÇÃO FINAL")
print("=" * 60)

predicoes = modelo.predict(X_test)
probabilidades = modelo.predict_proba(X_test)

prob_fake = probabilidades[:, 1]


accuracy = accuracy_score(y_test, predicoes)
precision = precision_score(y_test, predicoes)
recall = recall_score(y_test, predicoes)
f1 = f1_score(y_test, predicoes)

fpr, tpr, _ = roc_curve(y_test, prob_fake)
roc_auc = auc(fpr, tpr)

imprimir_metricas(
    accuracy,
    precision,
    recall,
    f1,
    roc_auc
)


relatorio = classification_report(
    y_test,
    predicoes,
    target_names=["Verdadeira", "Fake"]
)

print(relatorio)

salvar_relatorio(
    "resultados/classification_report_teste.txt",
    relatorio
)


cm = confusion_matrix(y_test, predicoes)

plt.figure(figsize=(6, 6))
plt.imshow(cm)
plt.title("Matriz de Confusão - Teste Final")
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
    "resultados/confusion_matrix_teste.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


plt.figure(figsize=(7, 7))
plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.3f}"
)

plt.plot([0, 1], [0, 1], "--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Teste Final")
plt.legend()

plt.savefig(
    "resultados/roc_curve_teste.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


dados = []

for i in range(len(textos)):

    classe_real = "Fake" if y_test[i] == 1 else "Verdadeira"
    classe_prevista = "Fake" if predicoes[i] == 1 else "Verdadeira"

    dados.append({
        "Arquivo": arquivos[i],
        "Classe Real": classe_real,
        "Classe Prevista": classe_prevista,
        "Probabilidade Fake (%)": round(prob_fake[i] * 100, 2),
        "Confiança (%)": round(max(probabilidades[i]) * 100, 2),
        "Resultado": "Acerto" if y_test[i] == predicoes[i] else "Erro"
    })

df_resultados = pd.DataFrame(dados)

df_resultados.to_excel(
    "resultados/classificacoes_teste.xlsx",
    index=False
)

df_resultados.to_csv(
    "resultados/classificacoes_teste.csv",
    index=False,
    sep=";",
    encoding="utf-8-sig"
)


total = len(y_test)
acertos = int((y_test == predicoes).sum())
erros = total - acertos

falsos_positivos = int(cm[0][1])
falsos_negativos = int(cm[1][0])
verdadeiros_negativos = int(cm[0][0])
verdadeiros_positivos = int(cm[1][1])


texto_relatorio = f"""
===============================
TESTE FINAL - RANDOM FOREST
===============================

Total de notícias avaliadas: {total}

Acertos: {acertos}
Erros: {erros}

Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1-score: {f1:.4f}
AUC: {roc_auc:.4f}

Matriz de Confusão:

Verdadeiras classificadas corretamente: {verdadeiros_negativos}
Verdadeiras classificadas como fake: {falsos_positivos}
Fake classificadas como verdadeiras: {falsos_negativos}
Fake classificadas corretamente: {verdadeiros_positivos}

Arquivos gerados:

- resultados/classification_report_teste.txt
- resultados/confusion_matrix_teste.png
- resultados/roc_curve_teste.png
- resultados/classificacoes_teste.xlsx
- resultados/classificacoes_teste.csv
"""

salvar_relatorio(
    "resultados/resultado_teste.txt",
    texto_relatorio
)

print(texto_relatorio)
print("Teste final concluído.")