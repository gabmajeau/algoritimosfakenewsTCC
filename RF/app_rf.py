from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

modelo = joblib.load("modelos/random_forest.pkl")
vectorizer = joblib.load("modelos/vectorizer.pkl")


def nivel_confianca(conf):
    if conf < 60:
        return "BAIXO"
    elif conf < 80:
        return "MÉDIO"
    else:
        return "ALTO"


def explicar_random_forest(texto, texto_vetorizado, predicao, probabilidades):
    feature_names = vectorizer.get_feature_names_out()

    prob_fake = probabilidades[1] * 100
    prob_true = probabilidades[0] * 100
    confianca = max(probabilidades) * 100

    votos = []
    for arvore in modelo.estimators_:
        votos.append(arvore.predict(texto_vetorizado)[0])

    votos_fake = int(sum(votos))
    votos_true = len(votos) - votos_fake

    importancias = modelo.feature_importances_
    indices = np.argsort(importancias)[::-1][:15]

    palavras_importantes = []
    for i in indices:
        if importancias[i] > 0:
            palavras_importantes.append(
                f"- **{feature_names[i]}**: {importancias[i]:.6f}"
            )

    arvore = modelo.estimators_[0]
    caminho = arvore.decision_path(texto_vetorizado).indices
    folha = arvore.apply(texto_vetorizado)[0]

    caminhos = []

    for node_id in caminho:
        if node_id == folha:
            valores = arvore.tree_.value[node_id][0]
            classe_folha = np.argmax(valores)
            texto_classe = "Fake News" if classe_folha == 1 else "Notícia Verdadeira"

            caminhos.append(
                f"- **Folha final {node_id}** → classificação da árvore: **{texto_classe}**"
            )
            continue

        feature_index = arvore.tree_.feature[node_id]
        threshold = arvore.tree_.threshold[node_id]

        if feature_index >= 0:
            palavra = feature_names[feature_index]
            valor_tfidf = texto_vetorizado[0, feature_index]

            if valor_tfidf <= threshold:
                direcao = "esquerda"
                comparacao = "<="
            else:
                direcao = "direita"
                comparacao = ">"

            caminhos.append(
                f"- Nó **{node_id}**: palavra **{palavra}** | "
                f"TF-IDF = `{valor_tfidf:.6f}` {comparacao} threshold `{threshold:.6f}` "
                f"→ segue para a **{direcao}**"
            )

    markdown = f"""
## Classificação Técnica — Random Forest

### Configurações da floresta

- Número de árvores: **{modelo.n_estimators}**
- Critério de divisão: **{modelo.criterion}**
- Bootstrap: **{modelo.bootstrap}**
- Máxima profundidade: **{modelo.max_depth}**
- Mínimo de amostras para divisão: **{modelo.min_samples_split}**
- Mínimo de amostras por folha: **{modelo.min_samples_leaf}**
- Máximo de atributos por divisão: **{modelo.max_features}**

---

### Probabilidades

- Probabilidade de notícia verdadeira: **{prob_true:.2f}%**
- Probabilidade de fake news: **{prob_fake:.2f}%**
- Confiança da decisão: **{confianca:.2f}%**

---

### Votação das árvores

- Árvores que votaram em **Fake News**: **{votos_fake}**
- Árvores que votaram em **Notícia Verdadeira**: **{votos_true}**
- Total de árvores: **{len(modelo.estimators_)}**

---

### Palavras com maior importância na floresta

{chr(10).join(palavras_importantes)}

---

### Caminho percorrido na primeira árvore

{chr(10).join(caminhos)}

---

### Observação

A classificação é baseada em padrões estatísticos aprendidos durante o treinamento. O modelo não verifica fatos em tempo real e deve ser utilizado como ferramenta de apoio.
"""

    return markdown


@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    prob_fake = None
    confianca = None
    nivel = None
    explicacao = None

    if request.method == "POST":
        texto = request.form["noticia"]

        if len(texto) < 400:
            resultado = "Texto insuficiente para análise"
            prob_fake = "--"
            confianca = "--"
            nivel = "INSUFICIENTE"
            explicacao = "O texto possui menos de 400 caracteres e não apresenta contexto suficiente para análise."
        else:
            texto_vetorizado = vectorizer.transform([texto])

            predicao = modelo.predict(texto_vetorizado)[0]
            probabilidades = modelo.predict_proba(texto_vetorizado)[0]

            prob_fake = round(probabilidades[1] * 100, 2)
            confianca = round(max(probabilidades) * 100, 2)
            nivel = nivel_confianca(confianca)

            if predicao == 1:
                resultado = "Possível Fake News"
            else:
                resultado = "Possível Notícia Verdadeira"

            explicacao = explicar_random_forest(
                texto,
                texto_vetorizado,
                predicao,
                probabilidades
            )

    return render_template(
        "index_rf.html",
        resultado=resultado,
        prob_fake=prob_fake,
        confianca=confianca,
        nivel=nivel,
        explicacao=explicacao
    )


if __name__ == "__main__":
    app.run(debug=True)