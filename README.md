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

<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/46b64755-f746-45e7-a04c-75832d97d03a" />

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
