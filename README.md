# Tech Challenge 1 - Entregáveis Grupo 18

Este repositório contém todos os artefatos, códigos e análises desenvolvidos para a entrega do Tech Challenge. Abaixo você encontra os links diretos para cada etapa do projeto e seus respectivos vídeos explicativos.

## 🔗 Links e Repositório
- **Repositório Git (Main):** [https://github.com/mvgv/techchallenge1](https://github.com/mvgv/techchallenge1)

---

## 🩺 Parte 1: Análise Exploratória e Predição de Diabetes
Neste subprojeto, construímos um pipeline completo de dados para o dataset Pima Indians, com foco em **recall da classe diabética** — métrica crítica para diagnóstico clínico, onde deixar passar um doente é mais grave que dar um alarme falso. Tratamos zeros biologicamente impossíveis com `SimpleImputer` (mediana), corrigimos o forte desbalanceamento (~2:1) com `SMOTE` aplicado apenas no treino, e comparamos cinco modelos sob o mesmo split estratificado 70/30: Random Forest baseline, Decision Tree, KNN normalizado e duas variantes Random Forest + PCA (3 e 5 componentes). O vencedor foi o **Random Forest + PCA(3)** (Recall = 0.84, F1 = 0.72, Accuracy = 0.77), demonstrando que mesmo numa base pequena o PCA pode atuar como regularizador quando há features de baixo poder preditivo.
- **Documentação e Código:** [Acessar a pasta diabetes-eda](./diabetes-eda/README.md)
- **Relatório do EDA:** [Acessar Notebook EDA](./diabetes-eda/diabetes_eda.ipynb)
- **Vídeo Explicativo Técnico:** [Assistir no YouTube (Placeholder)](https://youtube.com/watch?v=dummy_diabetes_video)

---

## 🫁 Parte 2: Detecção de Pneumonia
Neste subprojeto, focamos na etapa de detecção e classificação de imagens de Raio-X.
- **Documentação e Código:** [Acessar a pasta pneumonia](./pneumonia/README.md)

