import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# NOMES DAS COLUNAS (consulte a documentação do dataset)
colunas = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income"
]
# CARREGAMENTO DOS DADOS
df = pd.read_csv("/home/carlos_git-coder/PycharmProjects/Atividade_Avaliativa_Helo/DataSet/adult.data",
    header=None, names=colunas, skipinitialspace=True)

# INSPEÇÃO DAS COLUNAS CATEGÓRICAS
colunas_categoricas = df.select_dtypes(include=["object"]).columns

for coluna in colunas_categoricas:
    print(f"\n{coluna}")
    print(df[coluna].unique())

# PADRONIZAÇÃO DOS TEXTOS (ANTES x DEPOIS)
categorias_antes = df[colunas_categoricas].nunique()

df_padronizado = df.copy()

for coluna in colunas_categoricas:
    df_padronizado[coluna] = (
        df_padronizado[coluna]
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )

categorias_depois = df_padronizado[colunas_categoricas].nunique()

comparacao = pd.DataFrame({
    "Antes": categorias_antes,
    "Depois": categorias_depois
})

print(comparacao)

# CONVERSÃO DOS MARCADORES DE AUSÊNCIA
df_padronizado[colunas_categoricas] = (
    df_padronizado[colunas_categoricas]
    .replace(["?", ""], np.nan)
)

print(df_padronizado.isna().sum())

# ESCALONAMENTO — MIN-MAX
colunas_numericas = df_padronizado.select_dtypes(
    include=["number"]
).columns

dados_numericos = df_padronizado[colunas_numericas].dropna()

minmax = MinMaxScaler()

df_minmax = pd.DataFrame(
    minmax.fit_transform(dados_numericos),
    columns=colunas_numericas,
    index=dados_numericos.index
)

print(df_minmax.describe())

# ESCALONAMENTO — Z-SCORE
zscore = StandardScaler()

df_zscore = pd.DataFrame(
    zscore.fit_transform(dados_numericos),
    columns=colunas_numericas,
    index=dados_numericos.index
)

print(df_zscore.describe())

# ESCALONAMENTO — ROBUSTO
robusto = RobustScaler()

df_robusto = pd.DataFrame(
    robusto.fit_transform(dados_numericos),
    columns=colunas_numericas,
    index=dados_numericos.index
)

print(df_robusto.describe())

# COMPARAÇÃO ENTRE AS TÉCNICAS
for nome, dados in {
    "Original": dados_numericos,
    "Min-Max": df_minmax,
    "Z-score": df_zscore,
    "Robusto": df_robusto
}.items():
    print(f"\n{nome}")
    print(dados.agg(["min", "max", "mean", "std"]).round(3))

# RESPOSTAS ÀS PERGUNTAS DE REFLEXÃO

"""
1) Porque alguns datasets utilizam alguns simbolos para representar dados ausentes como (?) ou palavras como (Não existe)
2) Min-Max por que eles depende muito dos valores maximos e minimos ou seja ele e bastante afetado pelos valores maximos
3) Não acredito que possa haver uma tecnica que sirva para tudo por isso acho que não se aplica a todas as situações
como se fosse uma regra e a fnlwgt representa o peso amostral da população de cada registro
"""