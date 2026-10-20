# IMPORTAÇÕES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob

"""
        PERGUNTA DE NEGÓCIO

1) Como os preços e descontos anunciados variam entre produtos de
2) esportes e atividades ao ar livre e roupas femininas?
3) A análise busca comparar as categorias sem confundir dados ausentes,
4) duplicados ou valores extremos com características reais dos produtos.

"""


# CARREGAMENTO DOS ARQUIVOS
arquivos = glob.glob('*.csv')

arquivo_esportes = '/home/carlos_git-coder/PycharmProjects/Atividade_Avaliativa_Helo/DataSet/us-shein-sports_and_outdoors-3853.csv'
arquivo_roupas = '/home/carlos_git-coder/PycharmProjects/Atividade_Avaliativa_Helo/DataSet/us-shein-womens_clothing-4620.csv'

if arquivo_esportes not in arquivos or arquivo_roupas not in arquivos:
    raise FileNotFoundError(
        'Coloque os dois arquivos CSV na mesma pasta do notebook.'
    )

esportes_original = pd.read_csv(arquivo_esportes)
roupas_original = pd.read_csv(arquivo_roupas)


# Padroniza as colunas dos dois arquivos para permitir a comparação.
def preparar_colunas(df, categoria):
    df = df.copy()

    if categoria == 'Esportes e atividades ao ar livre':
        coluna_titulo = 'goods-title-link--jump'
        coluna_imagem = 'blackfridaybelts-bg src'
        coluna_url = 'goods-title-link--jump href'
    else:
        coluna_titulo = 'goods-title-link'
        coluna_imagem = 'product-locatelabels-img src'
        coluna_url = None

    df_padronizado = pd.DataFrame({
        'categoria': categoria,
        'titulo': df[coluna_titulo],
        'preco': df['price'],
        'desconto': df['discount'],
        'quantidade_cores': df['color-count'],
        'classificacao': df['rank-title'],
        'detalhes_classificacao': df['rank-sub'],
        'proposicao_venda': df['selling_proposition'],
        'imagem_url': df[coluna_imagem]
    })

    if coluna_url is not None:
        df_padronizado['produto_url'] = df[coluna_url]
    else:
        df_padronizado['produto_url'] = np.nan

    return df_padronizado


esportes = preparar_colunas(
    esportes_original,
    'Esportes e atividades ao ar livre'
)

roupas = preparar_colunas(
    roupas_original,
    'Roupas femininas'
)

df_original = pd.concat(
    [esportes, roupas],
    ignore_index=True
)

df = df_original.copy()


# DIAGNÓSTICO INICIAL (REGISTRAR PROBLEMAS ANTES DE CORRIGIR)
def diagnosticar(df):
    print('\n========== DIAGNÓSTICO INICIAL ==========')

    print('\nDimensões:')
    print(f'Linhas: {df.shape[0]}')
    print(f'Colunas: {df.shape[1]}')

    print('\nTipos das colunas:')
    print(df.dtypes)

    print('\nValores ausentes por coluna:')
    print(df.isna().sum())

    percentual_ausentes = df.isna().sum().sum() / df.size * 100
    print(f'\nPercentual total de ausentes: {percentual_ausentes:.2f}%')

    print('\nLinhas completamente duplicadas:')
    print(df.duplicated().sum())

    print('\nValores de preço antes da conversão:')
    print(df['preco'].head(10))

    print('\nValores de desconto antes da conversão:')
    print(df['desconto'].head(10))

    print('\nQuantidade de produtos por categoria:')
    print(df['categoria'].value_counts())


diagnosticar(df)


# Guarda indicadores iniciais para comparação posterior.
def indicadores_qualidade(df):
    percentual_ausentes = df.isna().sum().sum() / df.size * 100

    return {
        'Linhas': len(df),
        'Colunas': len(df.columns),
        'Ausentes (%)': round(percentual_ausentes, 2),
        'Duplicatas exatas': int(df.duplicated().sum()),
        'Tipos das colunas': df.dtypes.astype(str).to_dict()
    }


indicadores_antes = indicadores_qualidade(df)


# FUNÇÕES DO PIPELINE
def corrigir_tipos(df):
    df = df.copy()

    # Remove símbolos monetários e converte preço para número.
    df['preco'] = (
        df['preco']
        .astype('string')
        .str.replace(r'[$,]', '', regex=True)
        .str.strip()
    )

    df['preco'] = pd.to_numeric(
        df['preco'],
        errors='coerce'
    )

    # Converte descontos como "-40%" para 40.
    df['desconto'] = (
        df['desconto']
        .astype('string')
        .str.replace('%', '', regex=False)
        .str.strip()
    )

    df['desconto'] = pd.to_numeric(
        df['desconto'],
        errors='coerce'
    ).abs()

    # Corrige a quantidade de cores para um tipo numérico.
    df['quantidade_cores'] = pd.to_numeric(
        df['quantidade_cores'],
        errors='coerce'
    )

    return df


def tratar_ausentes(df):
    df = df.copy()

    # Sem título ou preço não é possível analisar o produto.
    df = df.dropna(subset=['titulo', 'preco'])

    # Preços zero ou negativos são tratados como inválidos.
    df = df[df['preco'] > 0].copy()

    # Ausência de desconto anunciado será considerada como
    # nenhum desconto informado, equivalente a 0% para esta análise.
    df['desconto'] = df['desconto'].fillna(0)

    # Usa a mediana para preencher a quantidade de cores ausente.
    mediana_cores = df['quantidade_cores'].median()

    if pd.notna(mediana_cores):
        df['quantidade_cores'] = (
            df['quantidade_cores'].fillna(mediana_cores)
        )

    # Campos textuais opcionais recebem uma indicação explícita.
    colunas_texto = [
        'classificacao',
        'detalhes_classificacao',
        'proposicao_venda'
    ]

    for coluna in colunas_texto:
        df[coluna] = df[coluna].fillna('Não informado')

    # URLs de imagem e produto podem continuar ausentes:
    # isso não impede a comparação de preço e desconto.

    return df


def tratar_duplicatas(df):
    df = df.copy()

    # Remove registros completamente iguais.
    df = df.drop_duplicates()

    # Também remove produtos com mesmo título, categoria e preço.
    # Mantém produtos de categorias diferentes separados.
    df = df.drop_duplicates(
        subset=['categoria', 'titulo', 'preco'],
        keep='first'
    )

    return df.reset_index(drop=True)


def padronizar_textos(df):
    df = df.copy()

    colunas_texto = [
        'categoria',
        'titulo',
        'classificacao',
        'detalhes_classificacao',
        'proposicao_venda'
    ]

    for coluna in colunas_texto:
        df[coluna] = (
            df[coluna]
            .astype('string')
            .str.strip()
            .str.replace(r'\s+', ' ', regex=True)
        )

    # Padroniza maiúsculas e minúsculas nos campos de classificação.
    for coluna in ['classificacao', 'detalhes_classificacao']:
        df[coluna] = df[coluna].str.lower()

    # Títulos são mantidos com sua capitalização original,
    # pois isso melhora a legibilidade dos nomes dos produtos.

    return df


def tratar_outliers(df):
    df = df.copy()

    # O IQR identifica preços extremos dentro de cada categoria.
    # Os valores são sinalizados, não removidos automaticamente,
    # pois podem representar produtos legítimos e caros.

    df['preco_outlier'] = False

    for categoria in df['categoria'].dropna().unique():
        mascara = df['categoria'] == categoria
        precos = df.loc[mascara, 'preco'].dropna()

        if len(precos) < 4:
            continue

        q1 = precos.quantile(0.25)
        q3 = precos.quantile(0.75)
        iqr = q3 - q1

        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr

        outliers = (
            (df['preco'] < limite_inferior)
            | (df['preco'] > limite_superior)
        )

        df.loc[mascara & outliers, 'preco_outlier'] = True

    return df


def criar_features(df):
    df = df.copy()

    # Cria uma classificação de preço para facilitar a análise.
    df['faixa_preco'] = pd.cut(
        df['preco'],
        bins=[0, 10, 30, 60, np.inf],
        labels=[
            'Até $10',
            '$10 a $30',
            '$30 a $60',
            'Acima de $60'
        ],
        right=True
    )

    # Identifica se o produto possui desconto anunciado.
    df['tem_desconto'] = df['desconto'] > 0

    return df


# EXECUÇÃO DO PIPELINE
df = corrigir_tipos(df)
df = padronizar_textos(df)
df = tratar_ausentes(df)
df = tratar_duplicatas(df)
df = tratar_outliers(df)
df = criar_features(df)

print('\n========== PIPELINE CONCLUÍDO ==========')
print(f'Linhas antes: {len(df_original)}')
print(f'Linhas depois: {len(df)}')

print('\nTipos depois da limpeza:')
print(df.dtypes)

print('\nAusentes depois da limpeza:')
print(df.isna().sum())

print('\nOutliers de preço identificados:')
print(df['preco_outlier'].sum())


# TABELA ANTES x DEPOIS
indicadores_depois = indicadores_qualidade(df)

comparacao = pd.DataFrame({
    'Antes': {
        chave: valor
        for chave, valor in indicadores_antes.items()
        if chave != 'Tipos das colunas'
    },
    'Depois': {
        chave: valor
        for chave, valor in indicadores_depois.items()
        if chave != 'Tipos das colunas'
    }
})

print('\n========== COMPARAÇÃO ANTES X DEPOIS ==========')
print(comparacao)

print('\nTipos antes:')
print(pd.Series(indicadores_antes['Tipos das colunas']))

print('\nTipos depois:')
print(pd.Series(indicadores_depois['Tipos das colunas']))


# Comparação de ausentes por coluna.
ausentes_antes = df_original.isna().sum()
ausentes_depois = df.isna().sum()

comparacao_ausentes = pd.DataFrame({
    'Ausentes antes': ausentes_antes,
    'Ausentes depois': ausentes_depois
})

print('\n========== AUSENTES POR COLUNA ==========')
print(comparacao_ausentes)


# VISUALIZAÇÕES
sns.set_theme(style='whitegrid')

# Gráfico 1: preços por categoria depois da limpeza.
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df,
    x='categoria',
    y='preco'
)
plt.title('Distribuição de preços por categoria')
plt.xlabel('Categoria')
plt.ylabel('Preço (US$)')
plt.xticks(rotation=10)
plt.tight_layout()
plt.show()


# Gráfico 2: percentual de valores ausentes antes e depois.
percentual_antes = df_original.isna().sum().sum() / df_original.size * 100
percentual_depois = df.isna().sum().sum() / df.size * 100

plt.figure(figsize=(8, 5))
sns.barplot(
    x=['Antes da limpeza', 'Depois da limpeza'],
    y=[percentual_antes, percentual_depois]
)
plt.title('Percentual total de valores ausentes')
plt.ylabel('Ausentes (%)')
plt.xlabel('Etapa')
plt.tight_layout()
plt.show()


# Gráfico 3: desconto médio anunciado por categoria.
desconto_medio = (
    df.groupby('categoria', observed=True)['desconto']
    .mean()
    .reset_index()
)

plt.figure(figsize=(9, 5))
sns.barplot(
    data=desconto_medio,
    x='categoria',
    y='desconto'
)
plt.title('Desconto médio anunciado por categoria')
plt.xlabel('Categoria')
plt.ylabel('Desconto médio (%)')
plt.xticks(rotation=10)
plt.tight_layout()
plt.show()


# Gráfico 4: preços extremos identificados pelo IQR.
plt.figure(figsize=(9, 5))
sns.countplot(
    data=df,
    x='categoria',
    hue='preco_outlier'
)
plt.title('Produtos com preços extremos identificados')
plt.xlabel('Categoria')
plt.ylabel('Quantidade de produtos')
plt.xticks(rotation=10)
plt.tight_layout()
plt.show()


# ERRO x VALOR EXTREMO LEGÍTIMO
print('\n========== ANÁLISE DE OUTLIERS ==========')

print(
    df.groupby('categoria', observed=True)['preco_outlier']
    .sum()
    .rename('Quantidade de preços extremos')
)

print(
    '\nUm preço extremo não é necessariamente um erro. '
    'Pode representar um produto mais caro ou especializado. '
    'Por isso, os valores foram sinalizados para análise, '
    'em vez de removidos automaticamente.'
)


# TRATAMENTO — REMOÇÃO / CAPPING / TRANSFORMAÇÃO
# A estratégia principal foi preservar os preços extremos e sinalizá-los.
# Para demonstrar o efeito de estratégias alternativas, criamos cópias.

df_sem_outliers = df.loc[~df['preco_outlier']].copy()

df_capping = df.copy()

for categoria in df_capping['categoria'].dropna().unique():
    mascara = df_capping['categoria'] == categoria
    precos = df_capping.loc[mascara, 'preco'].dropna()

    if len(precos) < 4:
        continue

    q1 = precos.quantile(0.25)
    q3 = precos.quantile(0.75)
    iqr = q3 - q1

    limite_inferior = max(0, q1 - 1.5 * iqr)
    limite_superior = q3 + 1.5 * iqr

    df_capping.loc[mascara, 'preco'] = (
        df_capping.loc[mascara, 'preco']
        .clip(lower=limite_inferior, upper=limite_superior)
    )

df_log = df.copy()
df_log['preco_log'] = np.log1p(df_log['preco'])

print('\nQuantidade de linhas na base original combinada:', len(df_original))
print('Quantidade após limpeza:', len(df))
print('Quantidade após remoção de preços extremos:', len(df_sem_outliers))

print('\nEstatísticas do preço original após limpeza:')
print(df['preco'].describe())

print('\nEstatísticas do preço após capping:')
print(df_capping['preco'].describe())

print('\nEstatísticas do preço após transformação logarítmica:')
print(df_log['preco_log'].describe())


# COMPARAÇÃO ENTRE AS ESTRATÉGIAS
estrategias = pd.DataFrame({
    'Estratégia': [
        'Dados limpos, sem remoção de outliers',
        'Remoção de preços extremos',
        'Capping pelo IQR'
    ],
    'Quantidade de linhas': [
        len(df),
        len(df_sem_outliers),
        len(df_capping)
    ],
    'Preço médio': [
        df['preco'].mean(),
        df_sem_outliers['preco'].mean(),
        df_capping['preco'].mean()
    ],
    'Mediana do preço': [
        df['preco'].median(),
        df_sem_outliers['preco'].median(),
        df_capping['preco'].median()
    ],
    'Preço máximo': [
        df['preco'].max(),
        df_sem_outliers['preco'].max(),
        df_capping['preco'].max()
    ]
})

print('\n========== COMPARAÇÃO ENTRE ESTRATÉGIAS ==========')
print(estrategias.round(2).to_string(index=False))


# RELATÓRIO E RESPOSTAS ÀS PERGUNTAS DE REFLEXÃO
print("""
========== RELATÓRIO ==========

Pergunta de negócio:
Como os preços e descontos anunciados variam entre esportes e atividades
ao ar livre e roupas femininas?

Decisões de limpeza:
1. Os preços e descontos foram convertidos de texto para números.
2. Registros sem título ou preço válido foram removidos, pois não permitem
   uma comparação confiável.
3. Descontos ausentes foram considerados como 0% de desconto anunciado.
   Essa é uma hipótese de análise, não uma garantia de que o produto nunca
   teve desconto.
4. A quantidade de cores ausente foi preenchida com a mediana.
5. Duplicatas exatas e produtos repetidos por categoria, título e preço
   foram removidos.
6. Os textos foram padronizados para reduzir diferenças de espaços e
   capitalização.
7. Os preços extremos foram identificados com o IQR por categoria e
   preservados na base principal, pois podem ser legítimos.
8. Foram criadas as features faixa_preco e tem_desconto.

Reflexão 1 — Qual problema foi mais difícil de perceber?
Números armazenados como texto e informações ausentes em campos opcionais
podem passar despercebidos em uma análise rápida. Um preço pode parecer
correto visualmente, mas não funcionar como número em cálculos estatísticos.

Reflexão 2 — A ordem das etapas importa?
Sim. É importante converter os preços para números antes de detectar
outliers. Se o preço continuar como texto, as operações estatísticas não
serão executadas corretamente. Também é importante tratar os valores
ausentes antes de calcular limites estatísticos.

Reflexão 3 — Se a pergunta de negócio fosse outra, o que mudaria?
Se o objetivo fosse encontrar produtos premium, remover preços extremos
poderia esconder justamente os produtos mais relevantes. Para analisar
descontos, seria necessário investigar com mais cuidado os descontos
ausentes e validar se a ausência realmente significa 0%.
""")