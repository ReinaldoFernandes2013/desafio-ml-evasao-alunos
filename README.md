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

Aguarde a conclusão da execução. Você verá mensagens de progresso e a confirmação de que os modelos foram salvos.

2. Testar a Função de Previsão (Executar agente_evasao.py)
Este script carregará o modelo e o encoder salvos e usará a função predict_evasao_status com dados de exemplo para demonstrar a previsão de evasão.

No terminal (com o ambiente virtual ativado), execute:

Bash

python agente_evasao.py
A saída mostrará as previsões para os alunos de teste. Você pode ignorar qualquer UserWarning relacionada a "X does not have valid feature names", pois ela é esperada devido à forma como os dados são passados ao modelo para contornar um problema de validação.

Web Scraping para Coleta de Dados
Um componente adicional do projeto é o script web_scraper.py, que demonstra a capacidade de coletar informações diretamente de portais web para identificar novos datasets.

Como Executar o Web Scraper
No terminal (com o ambiente virtual ativado), execute:

Bash

python web_scraper.py
Este script acessará o Portal de Dados Abertos do MEC, buscará links relevantes para dados de educação e imprimirá as URLs encontradas. Isso ilustra como um agente poderia proativamente buscar e integrar novas fontes de dados.

Testes Unitários com Pytest
O projeto inclui testes unitários para garantir a funcionalidade e a consistência da lógica de previsão.

Como Executar os Testes
No terminal (com o ambiente virtual ativado e na raiz do projeto), execute o pytest:

Bash

pytest tests/
A saída indicará quantos testes foram coletados e se passaram ou falharam. Um resultado como "2 passed, 2 warnings" é esperado (os warnings são devido à forma como os dados são passados ao modelo para evitar o erro de validação de features, o que é um comportamento conhecido e aceitável neste contexto).

Resultados do Modelo (Execução de analise_evasao.py)
Aqui está uma ilustração que representa a ideia do projeto:

<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/39f63fc5-65d5-48e4-a411-bb3ca08759fd" />

Métricas de Avaliação
Após o treinamento, o modelo Random Forest alcançou as seguintes métricas no conjunto de teste:

Acurácia (Accuracy): Aproximadamente 0.6767 (67.67%).

Relatório de Classificação:

          precision    recall  f1-score   support

       0       0.69      0.66      0.67     38088
       1       0.66      0.70      0.68     36720

accuracy                           0.68     74808
macro avg       0.68      0.68      0.68     74808
weighted avg       0.68      0.68      0.68     74808
```

Matriz de Confusão:

[[25013 13075]
 [11109 25611]]
Interpretação
A acurácia de ~67.7% indica que o modelo classifica corretamente se um aluno irá evadir ou não em mais de dois terços dos casos. As métricas de precisão e recall para ambas as classes (0: Não Evasão, 1: Evasão) são balanceadas, sugerindo que o modelo tem um desempenho razoável na identificação tanto de alunos que evadem quanto dos que não evadem. Isso o torna um ponto de partida promissor para intervenções.

Conceito de Agente ADK
A função predict_evasion_status no agente_evasao.py serve como a "ferramenta" central que um agente construído com o Google Agent Development Kit (ADK) utilizaria. Dentro de um framework ADK, esta função seria exposta e o agente, baseado em suas instruções ou no contexto de uma conversa, decidiria quando chamar essa ferramenta para obter uma previsão de evasão.

Por exemplo, um agente ADK poderia:

Receber uma pergunta: "Qual o risco de evasão do aluno João Silva com carga horária X, tipo de curso Y, etc.?"

Extrair as informações relevantes da pergunta.

Chamar a ferramenta predict_evasion_status (passando os dados do aluno como argumentos).

Receber a resposta ("Alto risco de evasão.").

Formular uma resposta amigável para o usuário.

Esta abordagem modular permite que o modelo de ML seja facilmente integrado a sistemas de agentes mais complexos e inteligentes, aproveitando a flexibilidade do ADK para orquestrar diversas ferramentas e capacidades de IA.
