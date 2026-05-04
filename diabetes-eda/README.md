# Diabetes Exploratory Data Analysis (EDA)

Este projeto realiza uma Análise Exploratória de Dados (EDA) focada na base de dados de diabetes, como parte de um tech challenge da disciplina de Machine Learning.

## Estrutura do Projeto

- `diabetes_eda.ipynb`: Jupyter Notebook contendo toda a análise exploratória, limpeza, visualizações de dados e processamento inicial.
- `requirements.txt`: Arquivo contendo as dependências e bibliotecas Python necessárias para executar o projeto de forma reprodutível.

## Pré-requisitos

Para executar este projeto localmente, você precisará ter instalado em sua máquina:
- [Python 3.8+](https://www.python.org/downloads/)
- [Jupyter Notebook](https://jupyter.org/install) (ou utilizar uma IDE com suporte a Jupyter, como o VS Code).

---

## Como Executar o Projeto

Siga as instruções abaixo de acordo com o seu sistema operacional para preparar o ambiente virtual, instalar as dependências e executar o notebook.

### 1. Acessar a pasta do projeto

Após clonar o repositório principal, navegue até o diretório específico deste projeto:
```bash
cd diabetes-eda
```

### 2. Criar e Ativar o Ambiente Virtual

Recomenda-se fortemente o uso de um ambiente virtual para isolar as dependências deste projeto e evitar conflitos.

#### 🪟 No Windows:

Abra o **Prompt de Comando (cmd)** ou **PowerShell** e execute:

```powershell
# Criar o ambiente virtual
python -m venv venv

# Ativar o ambiente virtual
# No Prompt de Comando:
venv\Scripts\activate.bat
# No PowerShell:
.\venv\Scripts\Activate.ps1
```

#### 🍎 No macOS / 🐧 Linux:

Abra o **Terminal** e execute:

```bash
# Criar o ambiente virtual
python3 -m venv venv

# Ativar o ambiente virtual
source venv/bin/activate
```

### 3. Instalar as Dependências

Com o ambiente virtual devidamente **ativado** (você verá um `(venv)` no início da linha de comando), instale as bibliotecas necessárias:

```bash
# Atualizar o pip (recomendado)
python -m pip install --upgrade pip

# Instalar os pacotes
pip install -r requirements.txt
```

### 4. Executar o Notebook

Após instalar todas as bibliotecas (incluindo dependências como `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, entre outras), inicie o Jupyter Notebook:

```bash
# Se o Jupyter foi instalado nas dependências, você pode rodar:
jupyter notebook diabetes_eda.ipynb
```

> **Dica para VS Code:** Se você utiliza o Visual Studio Code, não é necessário rodar o comando acima no terminal. Basta abrir o arquivo `diabetes_eda.ipynb` no editor e, no canto superior direito, certificar-se de selecionar o kernel do ambiente virtual recém-criado (`venv`).

---

## Encerrando a Execução

Quando terminar de visualizar ou modificar a análise, você pode fechar o servidor do Jupyter (pressionando `Ctrl + C` no terminal) e, em seguida, desativar o ambiente virtual executando o seguinte comando no seu terminal:

```bash
deactivate
```
