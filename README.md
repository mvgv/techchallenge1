# Tech Challenge 1 - Entregáveis Grupo 18

Este repositório contém todos os artefatos, códigos e análises desenvolvidos para a entrega do Tech Challenge. Abaixo você encontra os links diretos para cada etapa do projeto e seus respectivos vídeos explicativos.

## 🔗 Links e Repositório
- **Repositório Git (Main):** [https://github.com/mvgv/techchallenge1](https://github.com/mvgv/techchallenge1)

---

## 🩺 Parte 1: Análise Exploratória e Predição de Diabetes
Neste subprojeto, construímos um pipeline completo de dados para o dataset Pima Indians, com foco em **recall da classe diabética** — métrica crítica para diagnóstico clínico, onde deixar passar um doente é mais grave que dar um alarme falso. Tratamos zeros biologicamente impossíveis com `SimpleImputer` (mediana), corrigimos o forte desbalanceamento (~2:1) com `SMOTE` aplicado apenas no treino, e comparamos cinco modelos sob o mesmo split estratificado 70/30: Random Forest baseline, Decision Tree, KNN normalizado e duas variantes Random Forest + PCA (3 e 5 componentes). O vencedor foi o **Random Forest + PCA(3)** (Recall = 0.84, F1 = 0.72, Accuracy = 0.77), demonstrando que mesmo numa base pequena o PCA pode atuar como regularizador quando há features de baixo poder preditivo.
- **Documentação e Código:** [Acessar a pasta diabetes-eda](./diabetes-eda/README.md)
- **Relatório do EDA (Notebook com prints, gráficos e análises):** [Acessar Notebook EDA](./diabetes-eda/diabetes_eda.ipynb)
- **Dataset:** [Pima Indians Diabetes — Kaggle (`mathchi/diabetes-data-set`)](https://www.kaggle.com/datasets/mathchi/diabetes-data-set) — download automatizado via `kagglehub` no notebook.
- **Vídeo Explicativo Técnico:** [Assistir no YouTube](https://youtu.be/o5I6UpWcI6A)

---

## 🫁 Parte 2: Detecção de Pneumonia
Neste subprojeto, focamos na etapa de detecção e classificação de imagens de Raio-X com uma CNN. O pipeline cobre carregamento dos dados (`ImageDataGenerator`), treino (`src/train.py`), avaliação (`src/evaluate.py`) e interpretabilidade via Grad-CAM (`src/gradcam.py`). Em ambiente Windows, o `requirements.txt` usa `tensorflow-directml` para aceleração via DirectML.
- **Documentação e Código:** [Acessar a pasta pneumonia-detection](./pneumonia-detection/README.md)
- **Dataset:** [Chest X-Ray Images (Pneumonia) — Kaggle (`paultimothymooney/chest-xray-pneumonia`)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — extrair em `pneumonia-detection/data/chest_xray/{train,val,test}`.

---

## ✅ Entregáveis da Fase 1 — Mapa de rastreabilidade

| Requisito | Onde está |
|---|---|
| Link do repositório Git | Seção *Links e Repositório* (topo) |
| Código-fonte completo | [`diabetes-eda/`](./diabetes-eda) e [`pneumonia-detection/`](./pneumonia-detection) |
| README.md com instruções de execução | [diabetes-eda/README.md](./diabetes-eda/README.md) · [pneumonia-detection/README.md](./pneumonia-detection/README.md) |
| Dockerfile / instruções de ambiente | Execução via `venv` + `requirements.txt` documentada nos READMEs internos (ambientes Python isolados por subprojeto) |
| Dataset (link de download) | Diabetes: Kaggle `mathchi/diabetes-data-set` · Pneumonia: Kaggle `paultimothymooney/chest-xray-pneumonia` |
| Resultados (prints, gráficos, análises) | [Notebook EDA Diabetes](./diabetes-eda/diabetes_eda.ipynb) com gráficos, classification reports e ROC dos 5 modelos |
| Relatório técnico — pré-processamento | Resumo nesta página + detalhes no notebook (imputação por mediana, split estratificado 70/30, SMOTE só no treino, normalização para KNN) |
| Relatório técnico — modelos e justificativa | Resumo nesta página + notebook (RF baseline, Decision Tree, KNN, RF+PCA(3), RF+PCA(5); critério: F1 e recall da classe 1) |
| Relatório técnico — resultados e interpretação | Tabela comparativa e leitura da curva ROC no notebook |
| Vídeo de demonstração (≤ 15 min, YouTube) | [https://youtu.be/o5I6UpWcI6A](https://youtu.be/o5I6UpWcI6A) |

