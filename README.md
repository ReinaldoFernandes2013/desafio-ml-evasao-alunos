# Desafio Técnico - Agente de Previsão de Evasão Escolar

## Objetivo do Projeto

Este projeto tem como objetivo desenvolver um agente de inteligência artificial (IA) capaz de prever o risco de evasão escolar de alunos, utilizando dados públicos do Ministério da Educação (MEC) e um modelo de Machine Learning clássico. O desenvolvimento segue os princípios do Google Agent Development Kit (ADK), embora a integração final com o framework ADK seja conceitual e demonstrada através de uma função de predição isolada.

## Tema Escolhido

Educação: Previsão de Evasão Escolar.

## Fonte de Dados

Os dados utilizados para o treinamento do modelo são os Microdados de Eficiência Acadêmica do ano de 2018, obtidos no Portal de Dados Abertos do Ministério da Educação (MEC).
- **URL do dataset**: `https://dadosabertos.mec.gov.br/images/conteudo/pnp/2018/microdados_eficiencia_academica_2018.csv`

## Modelo de Machine Learning

Foi utilizado um modelo de Machine Learning clássico para o problema de classificação.
- **Tipo de Problema**: Classificação Binária (Evasão vs. Não Evasão).
- **Modelo**: Random Forest Classifier.

## Estrutura do Projeto

O projeto é composto pelos seguintes arquivos principais:
- `analise_evasao.py`: Script Python responsável pelo carregamento dos dados, pré-processamento (limpeza, tratamento de valores ausentes, One-Hot Encoding), treinamento do modelo de Machine Learning e salvamento do modelo treinado e do OneHotEncoder.
- `agente_evasao.py`: Script Python que encapsula a lógica de previsão. Ele carrega o modelo e o OneHotEncoder salvos, e contém uma função para realizar previsões em novos dados de alunos, aplicando as mesmas transformações do treinamento.
- `microdados_eficiencia_academica_2018.csv`: O arquivo original do dataset.
- `modelo_evasao.joblib`: O arquivo do modelo Random Forest treinado. (Gerado após a execução de `analise_evasao.py`)
- `onehot_encoder.joblib`: O arquivo do OneHotEncoder treinado. (Gerado após a execução de `analise_evasao.py`)
- `README.md`: Este arquivo, contendo a documentação do projeto.
- `requirements.txt`: Lista todas as bibliotecas Python e suas versões exatas necessárias para o projeto, permitindo uma instalação fácil e reproduzível.
- `web_scraper.py`: Um script Python adicional que demonstra a capacidade de coletar dinamicamente links para novos datasets de educação do Portal de Dados Abertos do MEC.
- `tests/`: Pasta contendo os testes unitários do projeto, desenvolvidos com `pytest`, para garantir a funcionalidade e a consistência da lógica de previsão.
  - `tests/test_agente_evasao.py`: Contém os testes específicos para a função `predict_evasion_status` em `agente_evasao.py`.

## Pré-requisitos

Antes de executar os scripts, certifique-se de ter o Python instalado (versão 3.8+ recomendada) e as seguintes bibliotecas Python instaladas. É altamente recomendável usar um ambiente virtual para gerenciar as dependências.

### Configuração do Ambiente Virtual

1.  Abra seu terminal (ou Git Bash no Windows) na pasta raiz do projeto (`desafio_adk`).
2.  Crie um ambiente virtual:
    ```bash
    python -m venv env_desafio
    ```
3.  Ative o ambiente virtual:
    -   No Windows (Git Bash/WSL):
        ```bash
        source env_desafio/Scripts/activate
        ```
    -   No macOS / Linux:
        ```bash
        source env_desafio/bin/activate
        ```
4.  Com o ambiente ativado, instale as dependências usando o `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
    **Nota**: Para a funcionalidade de Web Scraping, certifique-se de que `requests` e `beautifulsoup4` estejam instalados. Eles geralmente já são incluídos pelas dependências principais, mas caso contrário: `pip install requests beautifulsoup4`.

## Como Executar

Siga os passos abaixo para treinar o modelo e testar a função de previsão.

### 1. Treinar e Salvar o Modelo (Executar `analise_evasao.py`)

Este script fará todo o pré-processamento, treinará o modelo de Random Forest e salvará os arquivos `modelo_evasao.joblib` e `onehot_encoder.joblib` na pasta do projeto.

Certifique-se de que o arquivo `microdados_eficiencia_academica_2018.csv` esteja na mesma pasta que `analise_evasao.py`.

No terminal (com o ambiente virtual ativado), execute:

```bash
python analise_evasao.py
