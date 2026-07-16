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

from utils import carregar_dataset
from utils import salvar_relatorio


BASE_DIR = Path(__file__).resolve().parent

VALID_FOLDER = BASE_DIR / "dataset" / "validacao"
MODEL_FOLDER = BASE_DIR / "modelos"
RESULT_FOLDER = BASE_DIR / "resultados"

os.makedirs(RESULT_FOLDER, exist_ok=True)


print("=" * 60)
print("VALIDAÇÃO SVM")
print("=" * 60)

modelo = joblib.load(MODEL_FOLDER / "svm.pkl")
vectorizer = joblib.load(MODEL_FOLDER / "vectorizer_svm.pkl")

print("Modelo carregado com sucesso.\n")


textos, classes, arquivos = carregar_dataset(VALID_FOLDER)

X_valid = vectorizer.transform(textos)
y_valid = np.array(classes)


print("=" * 60)
print("CLASSIFICANDO VALIDAÇÃO")
print("=" * 60)

predicoes = modelo.predict(X_valid)

scores = modelo.decision_function(X_valid)

prob_fake = 1 / (1 + np.exp(-scores))


accuracy = accuracy_score(y_valid, predicoes)
precision = precision_score(y_valid, predicoes)
recall = recall_score(y_valid, predicoes)
f1 = f1_score(y_valid, predicoes)

fpr, tpr, _ = roc_curve(y_valid, prob_fake)
roc_auc = auc(fpr, tpr)


print()
print("=" * 60)
print("MÉTRICAS DA VALIDAÇÃO - SVM")
print("=" * 60)
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"AUC      : {roc_auc:.4f}")
print()


relatorio = classification_report(
    y_valid,
    predicoes,
    target_names=["Verdadeira", "Fake"]
)

print(relatorio)

salvar_relatorio(
    RESULT_FOLDER / "classification_report_validacao_svm.txt",
    relatorio
)


# ==========================================================
# MATRIZ DE CONFUSÃO
# ==========================================================

cm = confusion_matrix(y_valid, predicoes)

plt.figure(figsize=(6, 6))
plt.imshow(cm)
plt.title("Matriz de Confusão - Validação SVM")
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
    RESULT_FOLDER / "confusion_matrix_validacao_svm.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# ROC CURVE
# ==========================================================

plt.figure(figsize=(7, 7))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Validação SVM")
plt.legend()

plt.savefig(
    RESULT_FOLDER / "roc_curve_validacao_svm.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# CLASSIFICAÇÕES INDIVIDUAIS
# ==========================================================

dados = []

for i in range(len(textos)):
    classe_real = "Fake" if y_valid[i] == 1 else "Verdadeira"
    classe_prevista = "Fake" if predicoes[i] == 1 else "Verdadeira"

    confianca = max(prob_fake[i], 1 - prob_fake[i]) * 100

    dados.append({
        "Arquivo": arquivos[i],
        "Classe Real": classe_real,
        "Classe Prevista": classe_prevista,
        "Score SVM": round(scores[i], 6),
        "Distância Absoluta do Hiperplano": round(abs(scores[i]), 6),
        "Probabilidade Fake (%)": round(prob_fake[i] * 100, 2),
        "Confiança (%)": round(confianca, 2),
        "Resultado": "Acerto" if y_valid[i] == predicoes[i] else "Erro"
    })


df_resultados = pd.DataFrame(dados)

df_resultados.to_excel(
    RESULT_FOLDER / "classificacoes_validacao_svm.xlsx",
    index=False
)

df_resultados.to_csv(
    RESULT_FOLDER / "classificacoes_validacao_svm.csv",
    index=False,
    sep=";",
    encoding="utf-8-sig"
)


# ==========================================================
# VETOR DE PESOS DO SVM
# ==========================================================

print("=" * 60)
print("GERANDO VETOR DE PESOS DO SVM")
print("=" * 60)

feature_names = vectorizer.get_feature_names_out()
pesos = modelo.coef_[0]
intercepto = modelo.intercept_[0]

dados_pesos = []

for palavra, peso in zip(feature_names, pesos):

    if peso > 0:
        classe_favorecida = "Fake"
    elif peso < 0:
        classe_favorecida = "Verdadeira"
    else:
        classe_favorecida = "Neutra"

    dados_pesos.append({
        "Palavra": palavra,
        "Peso": peso,
        "Classe favorecida": classe_favorecida
    })

df_pesos = pd.DataFrame(dados_pesos)

df_pesos.to_csv(
    RESULT_FOLDER / "vetor_pesos_svm.csv",
    index=False,
    sep=";",
    encoding="utf-8-sig"
)

df_pesos.to_excel(
    RESULT_FOLDER / "vetor_pesos_svm.xlsx",
    index=False
)


# ==========================================================
# PALAVRAS MAIS IMPORTANTES
# ==========================================================

print("=" * 60)
print("GERANDO PALAVRAS MAIS IMPORTANTES")
print("=" * 60)

TOP_N = 50

df_fake = df_pesos.sort_values("Peso", ascending=False).head(TOP_N)
df_true = df_pesos.sort_values("Peso", ascending=True).head(TOP_N)

df_fake.to_csv(
    RESULT_FOLDER / "palavras_mais_importantes_fake_svm.csv",
    index=False,
    sep=";",
    encoding="utf-8-sig"
)

df_true.to_csv(
    RESULT_FOLDER / "palavras_mais_importantes_true_svm.csv",
    index=False,
    sep=";",
    encoding="utf-8-sig"
)

df_fake.to_excel(
    RESULT_FOLDER / "palavras_mais_importantes_fake_svm.xlsx",
    index=False
)

df_true.to_excel(
    RESULT_FOLDER / "palavras_mais_importantes_true_svm.xlsx",
    index=False
)


# ==========================================================
# HIPERPLANO
# ==========================================================

print("=" * 60)
print("ANÁLISE DO HIPERPLANO")
print("=" * 60)

norma_vetor = np.linalg.norm(pesos)

pesos_positivos = int(np.sum(pesos > 0))
pesos_negativos = int(np.sum(pesos < 0))
pesos_zerados = int(np.sum(pesos == 0))

maior_peso = float(np.max(pesos))
menor_peso = float(np.min(pesos))
media_pesos = float(np.mean(pesos))
desvio_pesos = float(np.std(pesos))

texto_hiperplano = f"""
===============================
HIPERPLANO - SVM
===============================

Modelo: LinearSVC
Tipo de separação: Hiperplano linear

Intercepto: {intercepto:.6f}
Norma do vetor de pesos: {norma_vetor:.6f}

Número total de atributos: {len(pesos)}
Pesos positivos: {pesos_positivos}
Pesos negativos: {pesos_negativos}
Pesos zerados: {pesos_zerados}

Maior peso: {maior_peso:.6f}
Menor peso: {menor_peso:.6f}
Média dos pesos: {media_pesos:.6f}
Desvio padrão dos pesos: {desvio_pesos:.6f}

Interpretação:

Pesos positivos deslocam a classificação em direção à classe Fake.
Pesos negativos deslocam a classificação em direção à classe Verdadeira.
O score calculado para cada notícia representa sua distância assinada em relação ao hiperplano.
Valores positivos indicam tendência à classe Fake.
Valores negativos indicam tendência à classe Verdadeira.
Quanto maior a distância absoluta, maior a segurança geométrica da decisão.
"""

salvar_relatorio(
    RESULT_FOLDER / "hiperplano_svm.txt",
    texto_hiperplano
)


# ==========================================================
# AMOSTRAS MAIS PRÓXIMAS DO HIPERPLANO
# ==========================================================

print("=" * 60)
print("GERANDO AMOSTRAS MAIS PRÓXIMAS DO HIPERPLANO")
print("=" * 60)

distancias = np.abs(scores)

indices_proximos = np.argsort(distancias)[:50]

dados_proximos = []

for idx in indices_proximos:

    dados_proximos.append({
        "Arquivo": arquivos[idx],
        "Classe Real": "Fake" if y_valid[idx] == 1 else "Verdadeira",
        "Classe Prevista": "Fake" if predicoes[idx] == 1 else "Verdadeira",
        "Score SVM": scores[idx],
        "Distância Absoluta do Hiperplano": abs(scores[idx]),
        "Probabilidade Fake (%)": round(prob_fake[idx] * 100, 2),
        "Confiança (%)": round(max(prob_fake[idx], 1 - prob_fake[idx]) * 100, 2),
        "Resultado": "Acerto" if y_valid[idx] == predicoes[idx] else "Erro"
    })

df_proximos = pd.DataFrame(dados_proximos)

df_proximos.to_csv(
    RESULT_FOLDER / "amostras_proximas_hiperplano_svm.csv",
    index=False,
    sep=";",
    encoding="utf-8-sig"
)

df_proximos.to_excel(
    RESULT_FOLDER / "amostras_proximas_hiperplano_svm.xlsx",
    index=False
)


# ==========================================================
# RELATÓRIO FINAL DA VALIDAÇÃO
# ==========================================================

total = len(y_valid)
acertos = int((y_valid == predicoes).sum())
erros = total - acertos

verdadeiras_corretas = int(cm[0][0])
verdadeiras_como_fake = int(cm[0][1])
fake_como_verdadeiras = int(cm[1][0])
fake_corretas = int(cm[1][1])


texto_relatorio = f"""
===============================
VALIDAÇÃO - SVM
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

Verdadeiras classificadas corretamente: {verdadeiras_corretas}
Verdadeiras classificadas como fake: {verdadeiras_como_fake}
Fake classificadas como verdadeiras: {fake_como_verdadeiras}
Fake classificadas corretamente: {fake_corretas}

===============================
CONFIGURAÇÃO DO SVM
===============================

Modelo: LinearSVC
C: {modelo.C}
Loss: {modelo.loss}
Max Iterations: {modelo.max_iter}
Random State: {modelo.random_state}

===============================
HIPERPLANO
===============================

Intercepto: {intercepto:.6f}
Norma do vetor de pesos: {norma_vetor:.6f}
Número total de atributos: {len(pesos)}
Pesos positivos: {pesos_positivos}
Pesos negativos: {pesos_negativos}
Pesos zerados: {pesos_zerados}

===============================
ARQUIVOS GERADOS
===============================

- classification_report_validacao_svm.txt
- confusion_matrix_validacao_svm.png
- roc_curve_validacao_svm.png
- classificacoes_validacao_svm.xlsx
- classificacoes_validacao_svm.csv
- vetor_pesos_svm.xlsx
- vetor_pesos_svm.csv
- palavras_mais_importantes_fake_svm.xlsx
- palavras_mais_importantes_true_svm.xlsx
- hiperplano_svm.txt
- amostras_proximas_hiperplano_svm.xlsx
- amostras_proximas_hiperplano_svm.csv
"""

salvar_relatorio(
    RESULT_FOLDER / "resultado_validacao_svm.txt",
    texto_relatorio
)

print(texto_relatorio)
print("Validação SVM concluída.")