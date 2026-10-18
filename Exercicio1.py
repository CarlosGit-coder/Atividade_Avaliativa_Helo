import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# CARREGAMENTO DOS DADOS
path = "/home/carlos_git-coder/PycharmProjects/Atividade_Avaliativa_Helo/DataSet/melb_data.csv"
df = pd.read_csv(path, sep = ",")


# DIAGNÓSTICO INICIAL

print(f"===== CABEÇALHO ===== \n {df.head(5)}")

print(f"\n==== DIMENSÃO DATASET ==== \n Linhas e Colunas: {df.shape}")

print(f"\n ==== TIPOS ==== \n {df.dtypes}")

print(f"\n ==== QUANTIDADE DE NULOS ==== \n {df.isna().sum()}")

# RESUMO DE VALORES AUSENTES (QUANTIDADE E %)
quantidade_valores_ausentes = df.isna().sum()

porcentagem_valores_ausentes = (df.isna().mean() * 100)

organizacao_valores_ausentes = pd.DataFrame({
    "Quantidade de ausentes" : quantidade_valores_ausentes,
    "Porcentagem de ausentes (%)": porcentagem_valores_ausentes,
})

organizacao_valores_ausentes = organizacao_valores_ausentes[
    organizacao_valores_ausentes["Quantidade de ausentes"] > 0
]

organizacao_valores_ausentes = organizacao_valores_ausentes.sort_values(
    by="Porcentagem de ausentes (%)",
    ascending=False
)

print(f"\n ==== RESUMO DOS VALORES INEXISTENTES ==== \n {organizacao_valores_ausentes.round(2)}")

# VISUALIZAÇÃO DO PADRÃO DE AUSÊNCIA
msno.matrix(df)
plt.title("Padrão de ausência")
plt.show()

msno.bar(df)
plt.title("Padrão de ausência por coluna")
plt.show()

msno.heatmap(df)
plt.title("Correlação Padrão de ausência")
plt.show()

# ESTRATÉGIA 1 — REMOÇÃO
df_remocao_valores_nulos = df.copy()

df_remocao_valores_nulos = df_remocao_valores_nulos.dropna(subset=["BuildingArea"])

print("\n ==== ESTRATÉGIA 1 -- REMOÇÃO ====")
print(f"Linhas no arquivo original: {len(df)}")
print(f"Linhas no arquivo que teve a remoção: {len(df_remocao_valores_nulos)}")
print(f"Linhas que foram removidas: {len(df) - len(df_remocao_valores_nulos)}")
print(f"Valores ausêntes que sobraram: {df_remocao_valores_nulos.isna().sum().sum()}")

# ESTRATÉGIA 2 — IMPUTAÇÃO
df_imputacao_em_valores_nulos = df.copy()

colunas_numericas = ["BuildingArea", "YearBuilt", "Car"]
for coluna in colunas_numericas:
    moda = df_imputacao_em_valores_nulos[coluna].mode()[0]
    df_imputacao_em_valores_nulos[coluna] = df_imputacao_em_valores_nulos[coluna].fillna(moda)

print("\n ==== ESTRATÉGIA 2 -- IMPUTAÇÃO ====")
print(f"Linhas no arquivo original: {len(df)}")
print(f"Linhas após imputação: {len(df_imputacao_em_valores_nulos)}")
print(f"\nValores ausêntes por coluna: \n{df_imputacao_em_valores_nulos[colunas_numericas].isna().sum()}")

# COMPARAÇÃO ENTRE AS ESTRATÉGIAS
colunas_comparadas = ["BuildingArea", "YearBuilt", "Car"]

resultados = []

comparacao_dataframes = [
    ("Original", df),
    ("Remoção", df_remocao_valores_nulos),
    ("Imputacao", df_imputacao_em_valores_nulos)
]

for nome, dados in comparacao_dataframes:
    for coluna in colunas_comparadas:
        resultados.append({
            "Estratégia": nome,
            "Coluna": coluna,
            "Linhas": len(dados),
            "Média": dados[coluna].mean(),
            "Mediana": dados[coluna].median(),
            "Ausentes": dados[coluna].isna().sum(),
        })

resultado_comparacao = pd.DataFrame(resultados)

print(f"\n ==== COMPARAÇÃO DAS ESTRATÉGIAS ==== \n {resultado_comparacao.round(2).to_string(index=False)}")
# RESPOSTAS ÀS PERGUNTAS DE REFLEXÃO
"""
1) Depende muito pois as vezes a remoção pode implicar em alteração em analises futuras por isso cabe sempre a 
analise para decidir qual o mais correto
2) Não dariam a media ela não e tão confiavel para essas anlises pois e muito influenciada por outliers já a mediana e 
menos sujeita a essas mudanças mas o mais correto seria a moda
3) Por exemplo quando a uma coluna com muitos valores vazios pode ser sinal de que a empresa está negligenciando esse 
dado seja ele de cadastro ou não  
"""

#DESAFIO
colunas_knn = [
    "Rooms",
    "Price",
    "Distance",
    "Bathroom",
    "Car",
    "Landsize",
    "BuildingArea"
]

df_knn = df.copy()

dados_numericos = df_knn[colunas_knn]

escalonador = StandardScaler()
dados_escalonados = escalonador.fit_transform(dados_numericos)

imputador = KNNImputer(n_neighbors=5)
dados_imputados = imputador.fit_transform(dados_escalonados)

dados_originais = escalonador.inverse_transform(dados_imputados)

df_knn[colunas_knn] = dados_originais

print("\n===== DESAFIO — KNN IMPUTER =====")
print("Linhas preservadas:", len(df_knn))
print("\nValores ausentes nas colunas utilizadas:")
print(df_knn[colunas_knn].isna().sum())

print("\nBuildingArea — comparação:")
print("Média original:", df["BuildingArea"].mean())
print("Média com mediana:", df_imputacao_em_valores_nulos["BuildingArea"].mean())
print("Média com KNN:", df_knn["BuildingArea"].mean())
