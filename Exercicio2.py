# IMPORTAÇÕES
import pandas as pd
import numpy as np

# CARREGAMENTO DOS DADOS
path = "/home/carlos_git-coder/PycharmProjects/Atividade_Avaliativa_Helo/DataSet/KPMG_VI_New_raw_data_update_final.xlsx"

df = pd.read_excel(
    path,
    sheet_name="Transactions",
    header=1
)

# CONFERÊNCIA DO CABEÇALHO E DOS TIPOS
print("Dimensões:", df.shape)

print("\nPrimeiras linhas:")
print(df.head())

print("\nNomes das colunas:")
print(df.columns.tolist())

print("\nTipos das colunas:")
print(df.dtypes)

print(df.head(20).to_string())

# DUPLICATAS EXATAS
duplicatas_exatas = df.duplicated()

quantidade_duplicatas_exatas = duplicatas_exatas.sum()

print(f"\n==== DUPLICATAS EXATAS ====\n Quantidade de linhas duplicadas: {quantidade_duplicatas_exatas}")

if quantidade_duplicatas_exatas > 0:
    print(f"\n==== LINHAS DUPLICADAS ====\n{df[duplicatas_exatas].to_string()}")
else:
    print("Nenhuma duplicada foi achada")

# DUPLICATAS POR SUBCONJUNTO DE COLUNAS
colunas_a_serem_analisadas = [
    "customer_id",
    "product_id",
    "transaction_date",
    "list_price"
]

duplicatas_subset = df.duplicated(subset=colunas_a_serem_analisadas, keep=False)

quantidade_subset = duplicatas_subset.sum()
print("\n--- DUPLICATAS POR SUBCONJUNTO ---")
print("Colunas analisadas:", colunas_a_serem_analisadas)
print("Quantidade de linhas em grupos repetidos:", quantidade_subset)

if quantidade_subset > 0:
    print("\nRegistros potencialmente repetidos:")
    print(
        df.loc[duplicatas_subset, colunas_a_serem_analisadas]
        .sort_values(colunas_a_serem_analisadas)
        .to_string(index=False)
    )
else:
    print("Nenhum grupo repetido encontrado nessas colunas.")


# UNICIDADE DOS IDENTIFICADORES

coluna_id = "transaction_id"

total_linhas = len(df)
ids_unicos = df[coluna_id].nunique()
ids_duplicados = df[coluna_id].duplicated().sum()
ids_ausentes = df[coluna_id].isna().sum()

print("\n--- UNICIDADE DO IDENTIFICADOR ---")
print("Total de linhas:", total_linhas)
print("Quantidade de IDs únicos:", ids_unicos)
print("Ocorrências de IDs duplicados:", ids_duplicados)
print("IDs ausentes:", ids_ausentes)

if ids_duplicados > 0:
    repetidos = df.loc[
        df[coluna_id].duplicated(keep=False)
    ].sort_values(coluna_id)

    print("\nIDs repetidos:")
    print(repetidos.to_string(index=False))
else:
    print("\nNenhum ID repetido foi encontrado.")

# ESTRATÉGIA DE TRATAMENTO E IMPACTO
df_original = df.copy()

df_tratado = df.drop_duplicates().copy()

linhas_antes = len(df_original)
linhas_depois = len(df_tratado)
linhas_removidas = linhas_antes - linhas_depois

soma_antes = df_original["list_price"].sum()
soma_depois = df_tratado["list_price"].sum()
diferenca = soma_antes - soma_depois

print("\n--- ESTRATÉGIA DE TRATAMENTO E IMPACTO ---")
print("Estratégia: remover somente duplicatas exatas")
print("Linhas antes do tratamento:", linhas_antes)
print("Linhas depois do tratamento:", linhas_depois)
print("Linhas removidas:", linhas_removidas)

print(f"\nSoma de list_price antes: R$ {soma_antes:,.2f}")
print(f"Soma de list_price depois: R$ {soma_depois:,.2f}")
print(f"Diferença na soma: R$ {diferenca:,.2f}")

# RESPOSTAS ÀS PERGUNTAS DE REFLEXÃO
"""
1) Na realidade isso não garante 100% pois as vezes pode haver duplicatas com algumas pequenas diferenças por exemplo
se houver repetidamente o nome Carlos mas escrito de diferentes formas como carlos ou cArlos isso são dados repetidos
mas com uma pequena diferença que não muda o fato de ser repetido
2) Não necessariamente por exemplo uma pessoa numa loja de roupa compra um short que está na moda e no caixa ao lado 
outra pessoa compra a mesma roupa no exato momento dependendo do jeito que esas loja guarda os dados da venda
pode indicar que essas linhas são repetidas mesmo não sendo erro
3) E uma decisão de negocio pois depende muito do contexto tipo oq eu dei na questão anterior isso passa pela etapa 
de entendimento de negócio do CRISP-DM
"""