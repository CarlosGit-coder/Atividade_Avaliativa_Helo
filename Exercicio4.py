# IMPORTAÇÕES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# CARREGAMENTO DOS DADOS

df = pd.read_csv('/home/carlos_git-coder/PycharmProjects/Atividade_Avaliativa_Helo/DataSet/diamonds.csv', sep = ",")

# ESTATÍSTICAS DESCRITIVAS E VALORES IMPOSSÍVEIS
print(df.describe())
print(df.isna().sum())

print(df[df['x'] <= 0])
print(df[df['y'] <= 0])
print(df[df['z'] <= 0])

# DETECÇÃO — Z-SCORE
colunas_numericas = df.select_dtypes(include=[np.number]).columns

z_scores = (
    (df[colunas_numericas] - df[colunas_numericas].mean())
    / df[colunas_numericas].std()
)

outliers_zscore = (z_scores.abs() > 3)

print(outliers_zscore.sum())

# DETECÇÃO — IQR
Q1 = df[colunas_numericas].quantile(0.25)
Q3 = df[colunas_numericas].quantile(0.75)
IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

outliers_iqr = (
    (df[colunas_numericas] < limite_inferior)
    | (df[colunas_numericas] > limite_superior)
)

print(outliers_iqr.sum())

# VISUALIZAÇÕES
for coluna in colunas_numericas:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[coluna])
    plt.title(f'Distribuição de {coluna}')
    plt.show()

# ERRO x VALOR EXTREMO LEGÍTIMO
print(df[(df['x'] <= 0) | (df['y'] <= 0) | (df['z'] <= 0)])

# TRATAMENTO — REMOÇÃO / CAPPING / TRANSFORMAÇÃO
df_sem_erros = df[
    (df['x'] > 0) &
    (df['y'] > 0) &
    (df['z'] > 0)
].copy()

df_remocao = df_sem_erros.copy()

df_capping = df_sem_erros.copy()

for coluna in colunas_numericas:
    q1 = df_capping[coluna].quantile(0.25)
    q3 = df_capping[coluna].quantile(0.75)
    iqr = q3 - q1

    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr

    df_capping[coluna] = df_capping[coluna].clip(
        lower=limite_inferior,
        upper=limite_superior
    )

df_transformacao = df_sem_erros.copy()

for coluna in colunas_numericas:
    if (df_transformacao[coluna] >= 0).all():
        df_transformacao[coluna] = np.log1p(
            df_transformacao[coluna]
        )

# COMPARAÇÃO ENTRE AS ESTRATÉGIAS
print("Quantidade original de registros:", len(df))
print("Após remover dimensões impossíveis:", len(df_sem_erros))
print("Após estratégia de remoção:", len(df_remocao))
print("Após capping:", len(df_capping))
print("Após transformação:", len(df_transformacao))

print("\nEstatísticas originais:")
print(df[colunas_numericas].describe())

print("\nEstatísticas após remoção:")
print(df_remocao[colunas_numericas].describe())

print("\nEstatísticas após capping:")
print(df_capping[colunas_numericas].describe())

print("\nEstatísticas após transformação:")
print(df_transformacao[colunas_numericas].describe())

# RESPOSTAS ÀS PERGUNTAS DE REFLEXÃO

"""
1) Ele considera a distância de um valor em relção as medidas de distribuição, já a price geralmente mostra uma
distribuição assimetrica
2) Na realidade não tem como saber pois embora seja um valor alto e um valor real o ideal seria separar ele da 
analise para não influenciar a analise e ser analisado separadamente 
3) Como todo contexto de loja de joias tem peças de entrada e alta rotatividade e peças de alto padrão e menor
rotatividade mas mesmo assim elas significam uma parte importante da receita e o objetivo da analise pode ser 
justamente o de aumentar a rotatividade dessas peças e quando retiramos ela da analise por terem um valor alto
estamos mascarando um historico importante que pode fazer toda a diferença
"""