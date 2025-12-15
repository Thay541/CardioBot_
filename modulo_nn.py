# Este módulo é responsável pelas seguintes funções:
# 1) Leitura do dataset Framingham (framingham.csv)
# 2) Análise exploratória básica:
#    - Formato do dataset
#    - Tipos de dados
#    - Estatísticas descritivas (média, desvio padrão etc.)
#    - Contagem de valores ausentes (NA)
#    - Distribuição da variável alvo (TenYearCHD)
# 3) Tratamento de dados:
#    - Imputação de linhas com valores faltantes (evitamos remover por conta do desbalanceamento do dataset)
#    - Análise de outliers extremos (ruídos)
# 4) Treino de um modelo de rede neural MLP usando um Pipeline (StandardScaler + MLP)
#    - Normalização das features numéricas
#    - Cálculo de acurácia, matriz de confusão e classification_report
# 5) Treino de outros modelos (regressão logística, Random Forest, XGBoost etc) para fins de comparação didática com a MLP
# 6) Salvamento do modelo MLP e dos nomes das features para uso no restante do sistema
#
# A interface externa é compatível com o sistema:
# - treinar_modelo()  -> treina e salva o modelo
# - carregar_modelo() -> retorna (modelo, feature_nomes)
# - predicao_diagnostico(features_dict) -> retorna o rótulo previsto (0 ou 1)
#
# O alvo aqui é TenYearCHD (0 = sem risco, 1 = risco de problemas coronarianos em 10 anos).


# Importando bibliotecas para trabalhar com tabelas (DataFrame) e vetores/matrizes.
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier # importando rede neural
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score # métricas para avaliar o modelo
from sklearn.pipeline import Pipeline # encadeia etapas (ex.: normalizar dados → treinar rede) de forma organizada
from sklearn.preprocessing import StandardScaler # normaliza os dados (média 0, desvio padrão 1)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression # para comparar com a rede neural devido ao desbalanceamento do dataset
# Nesse dataset, a regressão logística é clássica, interpretável e muitas vezes tem recall melhor
from sklearn.impute import SimpleImputer # ao invés de dar drop nos NaNs, vamos imputar para não perder ~14% do dataset, o que
# prejudica ainda mais a classe desbalanceada

# Comparar MLP com outros modelos
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import joblib # salvar e carregar o modelo treinado em disco

# Caminho do dataset Framingham
CAMINHO_DATASET = "framingham.csv"

# Caminho do arquivo de modelo salvo
CAMINHO_MODELO = "modelo_diagnostico.joblib"

# Nome da coluna alvo no dataset Framingham
COLUNA_ALVO_DATASET = "TenYearCHD"

# Esse threshold foi decidido incialmente após diversos testes com predict para aumentarmos
# a sensibilidade da classe minoritária (ocorrência de evento cardíaco)
# Optou-se por um limiar de decisão inferior a 0.5 com o objetivo de priorizar a sensibilidade do modelo para a classe positiva, 
# uma vez que, em contextos clínicos, falsos negativos representam maior risco do que falsos positivos.
THRESHOLD_RISCO = 0.30  # setamos esse porque foi o que teve melhor recall para a classe 1 sem reduzir muito a acurácia

def avaliar_modelo(nome, modelo, X_teste, y_teste, threshold=THRESHOLD_RISCO):
    # Essa função avalia um modelo no conjunto de teste e imprime métricas por classe
    # Usamos threshold quando o modelo tem predict_proba; caso contrário usamos predict()

    if hasattr(modelo, "predict_proba"):
        y_prob = modelo.predict_proba(X_teste)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = modelo.predict(X_teste)

    print(f"\n=== {nome} ===")
    print("Acurácia:", accuracy_score(y_teste, y_pred))
    print("Acurácia balanceada:", balanced_accuracy_score(y_teste, y_pred))
    print("Matriz de confusão [0, 1]:")
    print(confusion_matrix(y_teste, y_pred, labels=[0, 1]))
    print("Relatório de classificação:")
    print(classification_report(y_teste, y_pred, digits=3))


def treinar_modelo():
    # Função que lê o dataset Framingham, realiza análise exploratória,
    # tratamento de valores ausentes, treina um MLP dentro de um Pipeline (StandardScaler + MLPClassifier),
    # avalia o modelo e salva (modelo, feature_nomes) em CAMINHO_MODELO

    # ----------------------------------------------------------------
    # Explorando o dataset para ver as colunas, tipos de dados etc
    # ----------------------------------------------------------------
    print("=== Carregando dataset Framingham ===")
    df = pd.read_csv(CAMINHO_DATASET)
    print("Formato inicial do dataset (linhas, colunas):", df.shape)
    print("\nColunas disponíveis:")
    print(df.columns.tolist())

    print("\nTipos de dados (df.dtypes):")
    print(df.dtypes)

    print("\nPrimeiras 5 linhas (df.head()):")
    print(df.head())

    print("\nResumo estatístico das colunas numéricas:")
    # describe() já traz média, desvio padrão, quartis, min, max
    print(df.describe())

    # ---------------------------------------------------------
    # Verificando valores ausentes (NA) e sua proporção
    # ---------------------------------------------------------
    print("\n=== Análise de valores ausentes (NA) ===")
    missing_abs = df.isna().sum()
    missing_rel = (missing_abs / len(df)) * 100

    print("\nValores ausentes por coluna (absoluto):")
    print(missing_abs)

    print("\nValores ausentes por coluna (%):")
    print(missing_rel.round(2))

    # Contagem de linhas com pelo menos um NA
    linhas_com_na = df.isna().any(axis=1).sum()
    perc_linhas_com_na = 100.0 * linhas_com_na / len(df)
    print(
        f"\nNúmero de linhas com pelo menos um valor ausente: {linhas_com_na} "
        f"({perc_linhas_com_na:.2f}% do dataset)"
    )

    # ---------------------------------------------------------
    # Separando features (X) e alvo (y)
    # ---------------------------------------------------------
    print("\n=== Preparando features (X) e alvo (y) ===")

    if COLUNA_ALVO_DATASET not in df.columns:
        raise ValueError(f"Coluna alvo '{COLUNA_ALVO_DATASET}' não encontrada no dataset limpo.")

    y = df[COLUNA_ALVO_DATASET].astype(int)
    X = df.drop(columns=[COLUNA_ALVO_DATASET])

    print("Formato de X:", X.shape)
    print("Formato de y:", y.shape)

    # Distribuição da classe (para ver se é desbalanceada)
    print("\nDistribuição da coluna alvo TenYearCHD (contagem absoluta):")
    print(y.value_counts())

    print("\nDistribuição da coluna alvo TenYearCHD (%):")
    print((y.value_counts(normalize=True) * 100).round(2))

    # ---------------------------------------------------------
    # Separando treino e teste (estratificado)
    # ---------------------------------------------------------
    print("\n=== Separando treino e teste (80% / 20%, estratificado) ===")
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=0.2, # 80% dos dados vão para treino, 20% para teste
        random_state=42,
        stratify=y # garante que a proporção de 0 e 1 é parecida em treino e teste (isso é importante em problemas desbalanceados)
    )
    print("Tamanho X_treino:", X_treino.shape)
    print("Tamanho X_teste :", X_teste.shape)

    # ---------------------------------------------------------
    # Criando Pipeline: StandardScaler + MLPClassifier
    # ---------------------------------------------------------
    print("\n=== Modelo MLPClassifier com Pipeline (StandardScaler) ===")

    modelo = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="tanh",
                solver="lbfgs",
                alpha=1e-4,
                max_iter=2000,
                random_state=42
            ))
        ]
    )

    # Calcula pesos para cada amostra, dando mais peso à classe minoritária (TenYearCHD = 1)
    pesos_amostras = compute_sample_weight(class_weight="balanced", y=y_treino)

    # Treina o modelo passando os pesos para a MLP dentro do Pipeline
    modelo.fit(X_treino, y_treino, mlp__sample_weight=pesos_amostras)

    # A ideia dp pipeline é garantir que tudo que for feito no treino (normalização) será repetido
    # exatamente igual no momento da predição

    # Avaliação no conjunto de teste (com threshold)
    y_prob = modelo.predict_proba(X_teste)[:, 1]  # probabilidade da classe 1
    
    # ---------------------------------------------------------
    # Testando diferentes thresholds para classe positiva (TenYearCHD = 1)
    # ---------------------------------------------------------
    print("\n=== Verificando threshold (classe positiva) ===")

    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)

        print(f"\n--- Threshold = {t:.2f} ---")
        print("Acurácia balanceada:", balanced_accuracy_score(y_teste, y_pred_t))
        print(classification_report(y_teste, y_pred_t, digits=3))

    y_pred = (y_prob >= THRESHOLD_RISCO).astype(int)


    acuracia_teste = accuracy_score(y_teste, y_pred)
    print("Acurácia balanceada:", balanced_accuracy_score(y_teste, y_pred))

    print("\n=== Modelo MPL ===")
    print(f"Acurácia no conjunto de teste: {acuracia_teste:.3f}") # porcentagem de acertos no conjunto de teste

    print("\nMatriz de confusão (rótulos na ordem [0, 1]):") # mostra onde o modelo erra mais (falsos negativos/falsos positivos)
    cm = confusion_matrix(y_teste, y_pred, labels=[0, 1])
    print(cm)

    print("\nRelatório de classificação (precision, recall, f1-score):")
    print(classification_report(y_teste, y_pred, digits=3))
    # precision --> entre os casos que o modelo disse “risco”, quantos realmente tinham risco?
    # recall: entre os casos com risco, quantos o modelo conseguiu pegar?
    # f1-score: média harmônica entre precision e recall
    # Objetivo --> para saber que o modelo funciona melhor que um chute. 
    # Onde ele erra? O que é mais crítico (perder positivos ou alarmar falsos positivos)?

    # ---------------------------------------------------------
    # Modelo baseline: regressão logística (comparação)
    # ---------------------------------------------------------

    modelo_logreg = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42
            ))
        ]
    )
    modelo_logreg.fit(X_treino, y_treino)
    avaliar_modelo("Modelo Regressão Logística", modelo_logreg, X_teste, y_teste)

    # ---------------------------------------------------------
    # Modelo Random Forest (comparação)
    # ---------------------------------------------------------
    modelo_rf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42
            ))
        ]
    )
    modelo_rf.fit(X_treino, y_treino)
    avaliar_modelo("Modelo Random Forest", modelo_rf, X_teste, y_teste)

    # ---------------------------------------------------------
    # Modelo XGBoost (comparação)
    # ---------------------------------------------------------

    neg = int((y_treino == 0).sum())
    pos = int((y_treino == 1).sum())
    scale_pos_weight = neg / max(pos, 1)
    print("\nscale_pos_weight (XGBoost):", scale_pos_weight)

    modelo_xgb = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("xgb", XGBClassifier(
                n_estimators=500,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                scale_pos_weight=scale_pos_weight,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42
            ))
        ]
    )
    modelo_xgb.fit(X_treino, y_treino)
    avaliar_modelo("Modelo XGBoost", modelo_xgb, X_teste, y_teste)

    # ---------------------------------------------------------
    # Modelo LightGBM (comparação)
    # ---------------------------------------------------------
    modelo_lgbm = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("lgbm", LGBMClassifier(
                n_estimators=800,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                class_weight="balanced",
                random_state=42
            ))
        ]
    )
    modelo_lgbm.fit(X_treino, y_treino)
    avaliar_modelo("Modelo LightGBM", modelo_lgbm, X_teste, y_teste)


    # ---------------------------------------------------------
    # Salvando o modelo treinado e as features
    # ---------------------------------------------------------
    print("\n=== Salvando o modelo treinado ===")
    feature_nomes = X.columns.tolist()
    joblib.dump((modelo, feature_nomes), CAMINHO_MODELO)
    print(f"Modelo + nomes das features salvo em: {CAMINHO_MODELO}")
    print("Número de features:", len(feature_nomes))
    print("Algumas features:", feature_nomes[:10])
    # Guarda o pipeline inteiro já treinado + a ordem das features
    # Isso é o que depois o módulo de orquestração (CardioBot) vai carregar para fazer diagnósticos em produção


def carregar_modelo():
    # Carrega o modelo treinado e a lista de features a partir do arquivo .joblib.
    # Mantém a mesma interface usada no orquestrador: retorna (modelo, feature_nomes).
    
    modelo, feature_nomes = joblib.load(CAMINHO_MODELO)
    return modelo, feature_nomes


def predicao_diagnostico(features_dicionario):
    # Realiza a predição (TenYearCHD) a partir de um dicionário de features.
    """
    features_dicionario : dict
        Exemplo:
        {
            "age": 55,
            "male": 1,
            "currentSmoker": 1,
            "sysBP": 145,
            "totChol": 230,
            "diabetes": 0,
            ...
        }
    """

    modelo, feature_nomes = carregar_modelo()

    # Monta o vetor de entrada na mesma ordem usada no treino
    # Se alguma feature não estiver presente no dicionário, assume 0
    linha = [features_dicionario.get(nome, 0) for nome in feature_nomes]
    df_entrada = pd.DataFrame([linha], columns=feature_nomes)

    probs = modelo.predict_proba(df_entrada)[0]
    classes = list(modelo.classes_)

    # pega a probabilidade da classe 1
    idx_1 = classes.index(1)
    prob_risco = float(probs[idx_1])

    predicao = int(prob_risco >= THRESHOLD_RISCO)
    return predicao


def predicao_prob_risco(features_dicionario):
    # Retorna a probabilidade prevista de evento cardíaco (TenYearCHD = 1), usando predict_proba do modelo
    # Saída: float entre 0.0 e 1.0 representando P(Classe 1 | features).

    modelo, feature_nomes = carregar_modelo()

    # Monta o vetor de entrada na mesma ordem usada no treino
    linha = [features_dicionario.get(nome, 0) for nome in feature_nomes]
    df_entrada = pd.DataFrame([linha], columns=feature_nomes)

    # predict_proba retorna um array [ [p_classe0, p_classe1] ] na ordem modelo.classes_
    probs = modelo.predict_proba(df_entrada)[0]
    classes = list(modelo.classes_)

    if 1 in classes:
        idx = classes.index(1)
        prob_risco = float(probs[idx])
    else:
        # fallback só por segurança
        prob_risco = 0.0

    return prob_risco


if __name__ == "__main__":
    treinar_modelo()
