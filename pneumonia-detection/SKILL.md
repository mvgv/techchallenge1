# pneumonia-detection skill file

## Objetivo
Reduzir o consumo de tokens ao trabalhar com este projeto, mantendo respostas técnicas precisas e direcionadas.

## Escopo
- Diretório principal do projeto: `pneumonia-detection/`
- Arquivos principais: `src/train.py`, `src/model.py`, `src/data_loader.py`, `src/evaluate.py`, `src/gradcam.py`
- Dependências: `requirements.txt`

## Diretrizes de token efficiency
- Use respostas breves e objetivas.
- Não repita trechos de código inteiros quando apenas um bloco ou alteração for necessária.
- Prefira alterações pontuais e diffs em vez de reescrever arquivos completos.
- Evite incluir explicações longas; entregue apenas o necessário para resolver a tarefa.
- Ao analisar problemas, foque nas partes mais relevantes do código e ignore arquivos fora do escopo.

## Contexto do projeto
- Modelo: `DenseNet121` com top layer customizado.
- Treinamento: `ImageDataGenerator` para imagens de raio-X.
- Avaliação: `classification_report`, `confusion_matrix` e `Grad-CAM`.
- Estrutura dos dados esperada: `data/chest_xray/{train,val,test}`.

## Práticas recomendadas para este projeto
- Quando sugerir mudanças em `src/`, mencione somente o arquivo afetado.
- Se precisar de exemplos de código, mantenha o trecho curto e relevante.
- Se a resposta exigir diagnóstico, resuma em no máximo 3 frases.
