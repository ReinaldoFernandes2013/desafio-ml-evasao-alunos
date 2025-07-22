import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import OneHotEncoder # Importa o OneHotEncoder
import joblib # Importa joblib para salvar o modelo e o encoder

# --- Configuração Inicial e Carregamento de Dados ---

caminho_arquivo_csv = 'microdados_eficiencia_academica_2018.csv'
df = None

try:
    df = pd.read_csv(caminho_arquivo_csv, encoding='latin1', sep=';')
    print("Arquivo carregado com sucesso!")
    print("\nPrimeiras 5 linhas do DataFrame (original):")
    print(df.head())
    print("\nInformações sobre o DataFrame (original):")
    df.info()
    print("\nDescrição estatística básica (original - para colunas numéricas):")
    print(df.describe())

except FileNotFoundError:
    print(f"Erro: O arquivo '{caminho_arquivo_csv}' não foi encontrado.")
    print("Por favor, verifique o caminho e o nome do arquivo.")
    exit()
except Exception as e:
    print(f"Ocorreu um erro ao carregar o arquivo: {e}")
    print("Tente outras opções de 'encoding' como 'ISO-8859-1' ou 'utf-8', ou verifique o separador (sep=',' se for vírgula).")
    exit()


# --- Pré-processamento de Dados ---

print("\n--- Início do Pré-processamento ---")

# 1. Tentar corrigir nomes de colunas com problemas de codificação
df.columns = df.columns.str.replace('Ã§', 'ç').str.replace('Ã£', 'ã').str.replace('Ã³', 'ó').str.replace('Ãª', 'ê').str.replace('Ã¡', 'á').str.replace('Ã­', 'í').str.replace('Ãº', 'ú')
df.columns = df.columns.str.replace('Ã', 'a').str.replace('Â', '')

print("\nNomes das colunas após tentativa de correção:")
print(df.columns.tolist())

# 2. Análise e Preparação da Variável Alvo: 'Categoria da Situação'
df = df.rename(columns={'Categoria da Situação': 'Status_Aluno'})

df['Status_Aluno'] = df['Status_Aluno'].astype(str).str.replace('Ã§', 'ç').str.replace('Ã£', 'ã').str.replace('Ã³', 'ó').str.replace('Ãª', 'ê').str.replace('Ã¡', 'á').str.replace('Ã­', 'í').str.replace('Ãº', 'ú')
df['Status_Aluno'] = df['Status_Aluno'].str.replace('Ã', 'a').str.replace('Â', '')

print("\nValores únicos da coluna 'Status_Aluno' **APÓS** correção e antes do mapeamento:")
print(df['Status_Aluno'].value_counts())

df.loc[df['Status_Aluno'] == 'Evasão', 'Evasao'] = 1
df.loc[df['Status_Aluno'] != 'Evasão', 'Evasao'] = 0
df['Evasao'] = df['Evasao'].astype(int)

print("\nContagem da nova coluna 'Evasao':")
print(df['Evasao'].value_counts())
print("0: Não Evasão | 1: Evasão")

# 3. Remover colunas com muitos valores NaN ou irrelevantes (inicialmente)
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns, 'percent_missing': percent_missing})
missing_value_df.sort_values('percent_missing', inplace=True, ascending=False)

columns_to_drop_high_nan = missing_value_df[missing_value_df['percent_missing'] > 90]['column_name'].tolist()
df = df.drop(columns=columns_to_drop_high_nan, errors='ignore')

print(f"\nColunas removidas por ter mais de 90% de NaNs: {columns_to_drop_high_nan}")
print(f"Número de colunas após remoção de NaNs altos: {df.shape[1]}")

# --- Tratamento de Valores Ausentes Restantes ---

print("\n--- Tratamento de Valores Ausentes Restantes ---")

# Para colunas categóricas (object) com NaNs, preencher com a moda.
# E limpar os VALORES das categorias para que o OneHotEncoder treine com valores limpos.
for column in ['Subeixo Tecnológico', 'Eixo Tecnológico', 'Mês De Ocorrência da Situação', 'Fator Esforço Curso']:
    if column in df.columns and df[column].isnull().any():
        mode_value = df[column].mode()[0]
        df[column] = df[column].fillna(mode_value)
    # Limpa caracteres nos VALORES das colunas categóricas.
    if column in df.columns and df[column].dtype == 'object':
        df[column] = df[column].astype(str).str.replace('Ã§', 'ç').str.replace('Ã£', 'ã').str.replace('Ã³', 'ó').str.replace('Ãª', 'ê').str.replace('Ã¡', 'á').str.replace('Ã­', 'í').str.replace('Ãº', 'ú')
        df[column] = df[column].str.replace('Ã', 'a').str.replace('Â', '')


# Para colunas numéricas (float64, int64) com NaNs, preencher com a mediana
if 'Carga Horaria Mínima' in df.columns and df['Carga Horaria Mínima'].isnull().any():
    median_value = df['Carga Horaria Mínima'].median()
    df['Carga Horaria Mínima'] = df['Carga Horaria Mínima'].fillna(median_value)

print("\nVerificação de valores ausentes após preenchimento final:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# --- CODIFICAÇÃO DE VARIÁVEIS CATEGÓRICAS (USANDO OneHotEncoder) ---

print("\n--- Codificação de Variáveis Categóricas (USANDO OneHotEncoder) ---")

# Remover colunas com apenas 1 valor único (constantes)
cols_to_check_for_constant = [col for col in df.columns if col != 'Evasao' and col != 'Status_Aluno']
cols_to_drop_constant = [col for col in cols_to_check_for_constant if df[col].nunique() == 1]
df = df.drop(columns=cols_to_drop_constant, errors='ignore')
print(f"\nColunas constantes removidas: {cols_to_drop_constant}")
print(f"Número de colunas após remover constantes: {df.shape[1]}")

# Definir as colunas categóricas que IRÃO para o OneHotEncoder
# Elas são as mesmas que foram definidas como `cols_to_onehot_refined` anteriormente.
categorical_cols_to_encode = [
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

# Definir outras colunas a serem descartadas (IDs, datas, alta cardinalidade não usadas)
other_cols_to_discard = [
    'Código da Matricula',
    'Co Inst',
    'Cod Unidade', # ID
    'Código do Ciclo Matricula',
    'Código do Município com DV',
    'Data de Fim Previsto do Ciclo',
    'Data de Inicio do Ciclo',
    'Data de Ocorrencia da Matricula',
    'Mês De Ocorrência da Situação', # Alta cardinalidade
    'Instituição', # Alta cardinalidade
    'Nome de Curso', # Alta cardinalidade
    'Unidade de Ensino', # Alta cardinalidade
    'Município', # Cardinalidade média, mas descartada para simplicidade
    'Região', # Cardinalidade média
    'Situação de Matrícula' # Status original
]

# Remover as colunas que decidimos descartar (que não serão OHE nem numéricas)
# Garante que as colunas removidas não estejam nas que serão OHE.
df_filtered_for_ohe = df.drop(columns=[col for col in other_cols_to_discard if col in df.columns], errors='ignore')


# Separar features numéricas e categóricas para o encoder.
# X_numeric deve conter todas as colunas numéricas que não são a variável alvo e não foram descartadas.
X_numeric = df_filtered_for_ohe.select_dtypes(include=['int64', 'float64']).drop(columns=['Evasao'], errors='ignore')
X_categorical = df_filtered_for_ohe[categorical_cols_to_encode] # Usar a lista definida para o encoder.

# Inicializar e treinar o OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(X_categorical) # Treina o encoder com os dados categóricos

# Transformar os dados categóricos usando o encoder treinado
X_categorical_encoded = encoder.transform(X_categorical)

# Obter os nomes das colunas após o OneHotEncoding - ESTE É O PASSO CHAVE PARA NOMES LIMPOS
encoded_feature_names = encoder.get_feature_names_out(X_categorical.columns)
X_categorical_encoded_df = pd.DataFrame(X_categorical_encoded, columns=encoded_feature_names, index=X_numeric.index)

# Combinar as features numéricas e as categóricas codificadas
X_processed = pd.concat([X_numeric, X_categorical_encoded_df], axis=1)

# X final são as features processadas
X = X_processed
y = df['Evasao']

# Removendo a coluna 'Status_Aluno' que não é mais necessária em X (se ainda estiver lá)
if 'Status_Aluno' in X.columns:
    X = X.drop(columns=['Status_Aluno'])

print(f"\nNúmero de colunas após OneHotEncoder e remoção final: {X.shape[1]}")
print("\nPrimeiras 5 linhas do DataFrame X (Features preparadas):")
print(X.head())
print("\nInformações sobre o DataFrame X (Features preparadas):")
X.info()

# --- Separação de Features (X) e Variável Alvo (y) ---

print("\n--- Separação de Features (X) e Variável Alvo (y) ---")
print(f"Shape de X (features): {X.shape}")
print(f"Shape de y (variável alvo): {y.shape}")

# ADICIONADO: Lista final das colunas de features para uso no agente
# Esta é a lista exata que o modelo espera, com nomes limpos e consistentes!
expected_feature_columns_for_agent = X.columns.tolist()
print("\n--- Colunas de Features Finais (X.columns) para o Agente (USE ESTA LISTA NO AGENTE_EVASAO.PY) ---")
print(expected_feature_columns_for_agent)


# --- Divisão em Conjuntos de Treino e Teste ---

print("\n--- Divisão em Conjuntos de Treino e Teste ---")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Shape de X_train: {X_train.shape}")
print(f"Shape de X_test: {X_test.shape}")
print(f"Shape de y_train: {y_train.shape}")
print(f"Shape de y_test: {y_test.shape}")

# --- Treinamento do Modelo de Machine Learning ---

print("\n--- Treinamento do Modelo de Machine Learning ---")

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

print("Iniciando treinamento do modelo RandomForestClassifier...")
model.fit(X_train, y_train)
print("Treinamento concluído!")

# --- Avaliação do Modelo ---

print("\n--- Avaliação do Modelo ---")

print("Realizando previsões no conjunto de teste...")
y_pred = model.predict(X_test)

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAcurácia do Modelo: {accuracy:.4f}")

# --- Salvar o Modelo Treinado e o OneHotEncoder ---
print("\n--- Salvando o Modelo Treinado e o OneHotEncoder ---")
joblib.dump(model, 'modelo_evasao.joblib')
joblib.dump(encoder, 'onehot_encoder.joblib') # Salva o OneHotEncoder
print("Modelo salvo como 'modelo_evasao.joblib'")
print("OneHotEncoder salvo como 'onehot_encoder.joblib'")