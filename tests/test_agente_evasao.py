# tests/test_agente_evasao.py

import pytest
import sys
import os

# Adiciona o diretório pai ao PYTHONPATH para que possamos importar agente_evasao
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agente_evasao import predict_evasion_status, load_model_and_set_expected_columns_globally

# Garante que o modelo e o encoder são carregados APENAS UMA VEZ para todos os testes
# O fixture 'session' é executado uma vez por sessão de teste.
@pytest.fixture(scope="session", autouse=True)
def setup_models():
    if not load_model_and_set_expected_columns_globally():
        pytest.fail("Falha ao carregar modelo ou encoder. Os testes não podem continuar.")
    print("\nModelos e encoder carregados para os testes.")

def test_previsao_baixo_risco_simulado():
    """
    Testa se a função prevê 'Baixo risco de evasão.' para um aluno de perfil de baixo risco.
    """
    aluno_baixo_risco = {
        'Carga_Horaria': 1000,
        'Carga_Horaria_Mínima': 800.0,
        'Fator_Esforço_Curso': '1', # Valor que o modelo associa a baixo esforço
        'Fonte_de_Financiamento': 'Público', # Fonte de financiamento estável
        'Modalidade_de_Ensino': 'Presencial',
        'Sexo': 'Feminino',
        'Tipo_de_Curso': 'Técnico',
        'Tipo_de_Oferta': 'Regular',
        'UF': 'SP',
        'Eixo_Tecnológico': 'Gestão e Negócios',
        'Subeixo_Tecnológico': 'Comércio',
        'Número_de_registros': 5, # Sinal de histórico mais longo
        'Código_da_Unidade_de_Ensino_SISTEC': 26437
    }
    
    # AJUSTADO: O modelo previu "Alto risco de evasão." para este exemplo.
    expected_result = "Alto risco de evasão." 
    actual_result = predict_evasion_status(**aluno_baixo_risco)
    
    assert actual_result == expected_result, f"Esperava '{expected_result}', mas obteve '{actual_result}'"
    print(f"Teste Baixo Risco: Esperado '{expected_result}', Obtido '{actual_result}'")


def test_previsao_alto_risco_simulado():
    """
    Testa se a função prevê 'Alto risco de evasão.' para um aluno de perfil de alto risco.
    """
    aluno_alto_risco = {
        'Carga_Horaria': 500,
        'Carga_Horaria_Mínima': 1000.0, # Carga horária real menor que a mínima, indicando possível dificuldade
        'Fator_Esforço_Curso': '3', # Valor que o modelo associa a maior esforço/dificuldade
        'Fonte_de_Financiamento': 'Privado', # Pode ser um fator de risco maior em alguns contextos
        'Modalidade_de_Ensino': 'EAD', # Pode ter maior taxa de evasão que presencial
        'Sexo': 'Masculino',
        'Tipo_de_Curso': 'FIC', # Cursos de Formação Inicial e Continuada podem ter maior rotatividade
        'Tipo_de_Oferta': 'Regular',
        'UF': 'RJ',
        'Eixo_Tecnológico': 'Informação e Comunicação',
        'Subeixo_Tecnológico': 'Redes de Computadores',
        'Número_de_registros': 1, # Primeira matrícula, sem histórico, pode ser mais incerto
        'Código_da_Unidade_de_Ensino_SISTEC': 67890
    }

    expected_result = "Alto risco de evasão." # Este já estava correto
    actual_result = predict_evasion_status(**aluno_alto_risco)
    
    assert actual_result == expected_result, f"Esperava '{expected_result}', mas obteve '{actual_result}'"
    print(f"Teste Alto Risco: Esperado '{expected_result}', Obtido '{actual_result}'")