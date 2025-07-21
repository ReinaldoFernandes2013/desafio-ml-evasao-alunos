# main.py
# Este arquivo simula a integração com o Google Agent Development Kit (ADK)
# mostrando como a função de previsão de evasão seria exposta como uma ferramenta.

# NOTA: A biblioteca 'google-generative-ai' com a funcionalidade 'tooling'
# precisaria ser instalada e configurada com credenciais para um agente ADK real.
# Este é um exemplo conceitual de como a ferramenta seria estruturada.

import inspect
from typing import Dict, Any, Union
import json

# Importa sua função de previsão do agente_evasao.py
from agente_evasao import predict_evasion_status

# --- Definição da Ferramenta para o Agente ADK ---

# No contexto do Google ADK, você registraria suas funções como ferramentas.
# O ADK usa anotações de tipo e docstrings para entender como a ferramenta funciona.
# Esta classe simula como o ADK "veria" sua ferramenta.

class EvasionPredictionTool:
    """
    Ferramenta para prever o risco de evasão escolar de um aluno.
    """

    def predict_evasion(
        self,
        Carga_Horaria: int,
        Carga_Horaria_Mínima: float,
        Fator_Esforço_Curso: str,
        Fonte_de_Financiamento: str,
        Modalidade_de_Ensino: str,
        Sexo: str,
        Tipo_de_Curso: str,
        Tipo_de_Oferta: str,
        UF: str,
        Eixo_Tecnológico: str,
        Subeixo_Tecnológico: str,
        Número_de_registros: int,
        Código_da_Unidade_de_Ensino_SISTEC: int,
    ) -> str:
        """
        Prevê se um aluno tem alto ou baixo risco de evasão baseado em suas informações.

        Args:
            Carga_Horaria (int): Carga Horaria total do curso (por exemplo, 1000).
            Carga_Horaria_Mínima (float): Carga Horaria Mínima exigida pelo curso (por exemplo, 800.0).
            Fator_Esforço_Curso (str): Fator de Esforço do Curso (por exemplo, '1', '2', '3').
            Fonte_de_Financiamento (str): Fonte de Financiamento do aluno (por exemplo, 'Público', 'Privado').
            Modalidade_de_Ensino (str): Modalidade de Ensino (por exemplo, 'Presencial', 'EAD').
            Sexo (str): Sexo do aluno (por exemplo, 'Feminino', 'Masculino').
            Tipo_de_Curso (str): Tipo de Curso (por exemplo, 'Técnico', 'FIC').
            Tipo_de_Oferta (str): Tipo de Oferta (por exemplo, 'Regular', 'PROEJA - Integrado').
            UF (str): Unidade Federativa do curso (por exemplo, 'SP', 'MG').
            Eixo_Tecnológico (str): Eixo Tecnológico do curso (por exemplo, 'Gestão e Negócios', 'Informação e Comunicação').
            Subeixo_Tecnológico (str): Subeixo Tecnológico do curso (por exemplo, 'Comércio', 'Redes de Computadores').
            Número_de_registros (int): Número de registros do aluno (por exemplo, 1, 5).
            Código_da_Unidade_de_Ensino_SISTEC (int): Código da Unidade de Ensino - SISTEC (por exemplo, 12345).

        Returns:
            str: "Alto risco de evasão." se o risco for alto, "Baixo risco de evasão." se o risco for baixo.
        """
        # Esta função é um wrapper para a sua lógica de previsão real
        # que está em agente_evasao.py.
        return predict_evasion_status(
            Carga_Horaria=Carga_Horaria,
            Carga_Horaria_Mínima=Carga_Horaria_Mínima,
            Fator_Esforço_Curso=Fator_Esforço_Curso,
            Fonte_de_Financiamento=Fonte_de_Financiamento,
            Modalidade_de_Ensino=Modalidade_de_Ensino,
            Sexo=Sexo,
            Tipo_de_Curso=Tipo_de_Curso,
            Tipo_de_Oferta=Tipo_de_Oferta,
            UF=UF,
            Eixo_Tecnológico=Eixo_Tecnológico,
            Subeixo_Tecnológico=Subeixo_Tecnológico,
            Número_de_registros=Número_de_registros,
            Código_da_Unidade_de_Ensino_SISTEC=Código_da_Unidade_de_Ensino_SISTEC
        )

# --- Simulação de como o ADK registraria e chamaria a ferramenta ---

# Em um ambiente ADK real, você faria algo como:
# from google.generativeai.client import get_default_agent_client
# from google.generativeai.tooling import Tool
#
# client = get_default_agent_client()
# tool_instance = EvasionPredictionTool()
# tool_descriptor = Tool(tool_instance)
#
# agent = client.create_agent(
#     display_name="Agente de Previsão de Evasão Escolar",
#     description="Um agente que pode prever o risco de evasão escolar.",
#     tools=[tool_descriptor]
# )
#
# # E então você interagiria com o agente.

# Para demonstração, vamos apenas mostrar o schema (definição) da ferramenta
# e como a função seria chamada.

if __name__ == "__main__":
    print("--- Demonstração Conceitual de Agente ADK ---")

    tool_instance = EvasionPredictionTool()

    # O ADK inspeciona a função para entender seus parâmetros
    # Podemos simular o schema que seria gerado.
    tool_schema = {
        "name": "predict_evasion",
        "description": tool_instance.predict_evasion.__doc__.strip().split("\n")[0],
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }

    signature = inspect.signature(tool_instance.predict_evasion)
    for param_name, param in signature.parameters.items():
        if param_name == 'self':
            continue
        param_type = str(param.annotation).replace("<class '", "").replace("'>", "")
        tool_schema["parameters"]["properties"][param_name] = {
            "type": "integer" if "int" in param_type else "number" if "float" in param_type else "string",
            "description": f"Parâmetro {param_name} ({param_type})." # Docstring completa seria extraída pelo ADK
        }
        if param.default is inspect.Parameter.empty:
            tool_schema["parameters"]["required"].append(param_name)

    print("\nSchema (definição) da ferramenta 'predict_evasion' (como o ADK veria):")
    print(json.dumps(tool_schema, indent=2, ensure_ascii=False))

    print("\n--- Simulação de chamada da ferramenta pelo Agente ---")

    # Exemplo de como o agente chamaria a ferramenta com argumentos extraídos de uma conversa.
    # Estes são os mesmos dados de teste do agente_evasao.py

    student_data_1 = {
        'Carga_Horaria': 1000,
        'Carga_Horaria_Mínima': 800.0,
        'Fator_Esforço_Curso': '1',
        'Fonte_de_Financiamento': 'Público',
        'Modalidade_de_Ensino': 'Presencial',
        'Sexo': 'Feminino',
        'Tipo_de_Curso': 'Técnico',
        'Tipo_de_Oferta': 'Regular',
        'UF': 'SP',
        'Eixo_Tecnológico': 'Gestão e Negócios',
        'Subeixo_Tecnológico': 'Comércio',
        'Número_de_registros': 5,
        'Código_da_Unidade_de_Ensino_SISTEC': 26437
    }

    student_data_2 = {
        'Carga_Horaria': 500,
        'Carga_Horaria_Mínima': 1000.0,
        'Fator_Esforço_Curso': '3',
        'Fonte_de_Financiamento': 'Privado',
        'Modalidade_de_Ensino': 'EAD',
        'Sexo': 'Masculino',
        'Tipo_de_Curso': 'FIC',
        'Tipo_de_Oferta': 'Regular',
        'UF': 'RJ',
        'Eixo_Tecnológico': 'Informação e Comunicação',
        'Subeixo_Tecnológico': 'Redes de Computadores',
        'Número_de_registros': 1,
        'Código_da_Unidade_de_Ensino_SISTEC': 67890
    }

    try:
        print("\nChamando predict_evasion para o Aluno 1:")
        result_1 = tool_instance.predict_evasion(**student_data_1)
        print(f"Resultado: {result_1}")

        print("\nChamando predict_evasion para o Aluno 2:")
        result_2 = tool_instance.predict_evasion(**student_data_2)
        print(f"Resultado: {result_2}")

    except Exception as e:
        print(f"\nErro ao simular chamada da ferramenta: {e}")
        print("Certifique-se de que 'agente_evasao.py' está funcionando e que os arquivos .joblib estão presentes.")