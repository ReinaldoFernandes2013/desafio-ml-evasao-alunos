# agente_evasao.py

import pandas as pd
import joblib
import os
import numpy as np # Importa numpy

# --- Variáveis Globais e Caminhos ---
model = None
encoder = None # O OneHotEncoder agora será carregado
expected_feature_columns = None # Será preenchido dinamicamente pelo encoder do encoder.get_feature_names_out()

# Colunas categóricas que foram usadas para One-Hot Encoding
# ESTA LISTA PRECISA SER IDÊNTICA À 'cols_to_onehot_refined' NO SEU ANALISE_EVASAO.PY
categorical_cols_for_ohe = [
    'Fator Esforço Curso',
    'Fonte de Financiamento',
    'Modalidade de Ensino',
    'Sexo',
    'Tipo de Curso',
    'Tipo de Oferta',
    'UF',
    'Eixo Tecnológico',
    'Subeixo Tecnológico'
]

# Colunas que foram descartadas explicitamente no treinamento (IDs e outras não usadas como features)
all_cols_to_drop = [
    'Código da Matricula',
    'Co Inst',
    'Cod Unidade',
    'Código do Ciclo Matricula',
    'Código do Município com DV',
    'Data de Fim Previsto do Ciclo',
    'Data de Inicio do Ciclo',
    'Data de Ocorrencia da Matricula',
    'Mês De Ocorrência da Situação',
    'Instituição',
    'Nome de Curso',
    'Unidade de Ensino',
    'Município',
    'Região',
    'Situação de Matrícula',
    'Ano', # Removida como constante
    'Cor / Raça', # Removida como constante
    'Faixa Etária', # Removida como constante
    'Idade', # Removida como constante
    'Matrícula Atendida', # Removida como constante
    'Renda Familiar', # Removida como constante
    'Turno' # Removida como constante
]


# --- Funções de Pré-processamento (usando o OneHotEncoder salvo) ---

def preprocess_new_data_for_prediction(new_student_data: dict) -> pd.DataFrame:
    """
    Prepara um dicionário de dados de um novo aluno para a previsão.
    Utiliza o OneHotEncoder treinado para garantir a consistência das features.
    """
    ordered_input_columns = [
        'Carga Horaria', 'Carga Horaria Mínima', 'Fator Esforço Curso',
        'Fonte de Financiamento', 'Modalidade de Ensino', 'Sexo',
        'Tipo de Curso', 'Tipo de Oferta', 'UF', 'Eixo Tecnológico',
        'Subeixo Tecnológico', 'Número de registros', 'Código da Unidade de Ensino - SISTEC'
    ]

    df_new = pd.DataFrame([new_student_data], columns=ordered_input_columns)


    # 1. Limpar caracteres em colunas categóricas de entrada
    for col in [c for c in categorical_cols_for_ohe if c in df_new.columns]:
        s = df_new[col].astype(str)
        s = s.str.replace('Ã§', 'ç').str.replace('Ã£', 'ã').str.replace('Ã³', 'ó').str.replace('Ãª', 'ê').str.replace('Ã¡', 'á').str.replace('Ã­', 'í').str.replace('Ãº', 'ú')
        s = s.str.replace('Ã', 'a').str.replace('Â', '')
        s = s.str.replace(',', '.') # Converte vírgulas para pontos em números como '1,01' -> '1.01'
        df_new[col] = s

    # 2. Tratamento de Valores Ausentes para novas entradas
    median_carga_horaria_minima = 1000.0 # Valor da mediana do treino
    if 'Carga Horaria Mínima' in df_new.columns and df_new['Carga Horaria Mínima'].isnull().any():
        df_new['Carga Horaria Mínima'] = df_new['Carga Horaria Mínima'].fillna(median_carga_horaria_minima)
    
    mode_values = { # Modas das colunas categóricas do treino
        'Subeixo Tecnológico': 'Desenvolvimento Educacional e Social',
        'Eixo Tecnológico': 'Desenvolvimento Educacional e Social',
        'Fator Esforço Curso': '1',
    }
    for col, mode_val in mode_values.items():
        if col in df_new.columns and df_new[col].isnull().any():
            df_new[col] = df_new[col].fillna(mode_val)
    
    # 3. Remover colunas descartadas (IDs e outras que não devem ser features)
    cols_to_remove_from_input = [col for col in all_cols_to_drop if col in df_new.columns]
    
    if 'Código da Unidade de Ensino - SISTEC' in cols_to_remove_from_input:
        cols_to_remove_from_input.remove('Código da Unidade de Ensino - SISTEC')

    df_new = df_new.drop(columns=cols_to_remove_from_input, errors='ignore')

    # 4. Separar colunas numéricas e categóricas para o encoder
    X_numeric_input = df_new[[col for col in ['Carga Horaria', 'Carga Horaria Mínima', 'Número de registros', 'Código da Unidade de Ensino - SISTEC'] if col in df_new.columns]]
    X_categorical_input = df_new[[col for col in categorical_cols_for_ohe if col in df_new.columns]]


    # Verificar se o encoder foi carregado
    if encoder is None:
        raise RuntimeError("OneHotEncoder não carregado. Chame load_model_and_set_columns_globally() primeiro.")

    # Transformar dados categóricos usando o encoder carregado
    X_categorical_encoded_array = encoder.transform(X_categorical_input)
    
    # Criar DataFrame com as colunas dummy geradas
    X_categorical_encoded_df = pd.DataFrame(X_categorical_encoded_array, columns=encoder.get_feature_names_out(X_categorical_input.columns), index=df_new.index)

    # Construir o DataFrame final garantindo a ordem EXATA de expected_feature_columns
    df_final = pd.DataFrame(0.0, index=df_new.index, columns=expected_feature_columns, dtype=np.float64)

    for col in expected_feature_columns:
        if col in X_numeric_input.columns:
            df_final[col] = X_numeric_input[col].astype(np.float64)
        elif col in X_categorical_encoded_df.columns:
            df_final[col] = X_categorical_encoded_df[col].astype(np.float64)
            
    return df_final


# --- Inicialização do Modelo e Encoder ---

def load_model_and_set_expected_columns_globally():
    """
    Carrega o modelo treinado e o OneHotEncoder.
    Define as colunas de features esperadas com base no encoder.
    """
    global model, encoder, expected_feature_columns

    model_path = 'modelo_evasao.joblib'
    encoder_path = 'onehot_encoder.joblib'

    if not os.path.exists(model_path):
        print(f"Erro: Modelo não encontrado em {model_path}.")
        return False
    if not os.path.exists(encoder_path):
        print(f"Erro: OneHotEncoder não encontrado em {encoder_path}.")
        return False

    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path) # Carrega o OneHotEncoder

    numeric_features_in_X_final = ['Carga Horaria', 'Carga Horaria Mínima', 'Número de registros', 'Código da Unidade de Ensino - SISTEC']
    
    encoded_names = encoder.get_feature_names_out(categorical_cols_for_ohe)

    expected_feature_columns = numeric_features_in_X_final + list(encoded_names)

    print(f"Modelo de previsão de evasão carregado de {model_path}")
    print(f"OneHotEncoder carregado de {encoder_path}")
    print(f"Total de features esperadas pelo modelo: {len(expected_feature_columns)}")
    return True


# --- Função de Previsão de Evasão (Ferramenta do Agente) ---

def predict_evasion_status(
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
        Carga_Horaria (int): Carga Horaria total do curso.
        Carga_Horaria_Mínima (float): Carga Horaria Mínima exigida.
        Fator_Esforço_Curso (str): Fator de Esforço do Curso (ex: '1', '2', '3').
        Fonte_de_Financiamento (str): Fonte de Financiamento do aluno (ex: 'Público', 'Privado').
        Modalidade_de_Ensino (str): Modalidade de Ensino (ex: 'Presencial', 'EAD').
        Sexo (str): Sexo do aluno (e.g., 'Feminino', 'Masculino').
        Tipo_de_Curso (str): Tipo de Curso (e.g., 'Técnico', 'FIC').
        Tipo_de_Oferta (str): Tipo de Oferta (e.g., 'Regular', 'PRONATEC').
        UF (str): Unidade Federativa do curso (e.g., 'SP', 'MG').
        Eixo_Tecnológico (str): Eixo Tecnológico do curso (e.g., 'Informação e Comunicação').
        Subeixo_Tecnológico (str): Subeixo Tecnológico do curso.
        Número_de_registros (int): Número de registros do aluno.
        Código_da_Unidade_de_Ensino_SISTEC (int): Código da Unidade de Ensino - SISTEC.

    Returns:
        str: "Alto risco de evasão" ou "Baixo risco de evasão".
    """
    global model, encoder, expected_feature_columns

    if model is None or encoder is None:
        if not load_model_and_set_expected_columns_globally():
            return "Erro interno: Modelo ou Encoder não pôde ser carregado."

    new_student_data_dict = {
        'Carga Horaria': Carga_Horaria,
        'Carga Horaria Mínima': Carga_Horaria_Mínima,
        'Fator Esforço Curso': Fator_Esforço_Curso,
        'Fonte de Financiamento': Fonte_de_Financiamento,
        'Modalidade de Ensino': Modalidade_de_Ensino,
        'Sexo': Sexo,
        'Tipo de Curso': Tipo_de_Curso,
        'Tipo de Oferta': Tipo_de_Oferta,
        'UF': UF,
        'Eixo Tecnológico': Eixo_Tecnológico,
        'Subeixo Tecnológico': Subeixo_Tecnológico,
        'Número de registros': Número_de_registros,
        'Código da Unidade de Ensino - SISTEC': Código_da_Unidade_de_Ensino_SISTEC,
    }

    processed_data_df = preprocess_new_data_for_prediction(new_student_data_dict)

    # *** AQUI ESTÁ A ÚLTIMA TENTATIVA: CONVERTER PARA NUMPY ARRAY ANTES DE PREVER ***
    prediction = model.predict(processed_data_df.values)[0]

    if prediction == 1:
        return "Alto risco de evasão."
    else:
        return "Baixo risco de evasão."

# --- Carregar o modelo e encoder na inicialização do script (para uso em outros módulos ou diretamente) ---
if __name__ != '__main__':
    load_model_and_set_expected_columns_globally()

# --- Exemplo de uso direto da função (para teste independente do agente ADK) ---
if __name__ == "__main__":
    load_model_and_set_expected_columns_globally() # Garante que está carregado para o teste

    print("\n--- Testando a função de previsão (predict_evasion_status) ---")
    
    # Os valores das strings devem ser CATEGORIAS LIMPAS e no case correto (ex: 'Feminino', 'Gestão e Negócios').
    # O preprocess_new_data_for_prediction agora garante que elas sejam transformadas corretamente pelo encoder.
    
    test_student_data_baixo_risco = {
        'Carga_Horaria': 1000,
        'Carga_Horaria_Mínima': 800.0,
        'Fator_Esforço_Curso': '1', # Use o valor da categoria limpa
        'Fonte_de_Financiamento': 'Público', # Use o valor da categoria limpa
        'Modalidade_de_Ensino': 'Presencial',
        'Sexo': 'Feminino', # Use 'Feminino' ou 'Masculino' ou 'Não Informado' conforme as categorias originais.
        'Tipo_de_Curso': 'Técnico',
        'Tipo_de_Oferta': 'Regular',
        'UF': 'SP',
        'Eixo_Tecnológico': 'Gestão e Negócios', # Use o nome limpo da categoria
        'Subeixo_Tecnológico': 'Comércio', # Use o nome limpo da categoria
        'Número_de_registros': 5,
        'Código_da_Unidade_de_Ensino_SISTEC': 26437
    }
    
    # Exemplo de entrada 2: Aluno com perfil de alto risco simulado
    test_student_data_alto_risco = {
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
        'Código_da_Unidade_de_Ensino_SISTEC': 26422
    }

    pred_baixo = predict_evasion_status(**test_student_data_baixo_risco)
    print(f"Previsão para aluno 1 (baixo risco simulado): {pred_baixo}")

    pred_alto = predict_evasion_status(**test_student_data_alto_risco)
    print(f"Previsão para aluno 2 (alto risco simulado): {pred_alto}")