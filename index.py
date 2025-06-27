from collections import deque
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from river import anomaly
from river import metrics
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import pdb
from river import compose
from river import preprocessing
from sklearn.metrics import (
    auc,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from river import ensemble
from river import tree
import pickle

######### Constantes
NORMAL_CLASS = 6
ZERO_DAY_CLASSES = [0, 8]
ZERO_DAY = 9
KNOW_CLASSES = [ 1, 2, 3, 4, 5, 7]
# Carregar o arquivo CSV
df = pd.read_csv("ERENO-2.0-100K.csv", low_memory=False)

best = [
    "Time",
    "isbA",
    "isbB",
    "isbC",
    "vsbA",
    "vsbB",
    "vsbC",
    "isbARmsValue",
    "isbBRmsValue",
    "isbCRmsValue",
    "vsbARmsValue",
    "vsbBRmsValue",
    "vsbCRmsValue",
    "isbATrapAreaSum",
    "isbBTrapAreaSum",
    "isbCTrapAreaSum",
    "vsbATrapAreaSum",
    "vsbBTrapAreaSum",
    "vsbCTrapAreaSum",
    "t",
    "GooseTimestamp",
    "SqNum",
    "StNum",
    "cbStatus",
    "gooseTimeAllowedtoLive",
    "confRev",
    "APDUSize",
    "stDiff",
    "sqDiff",
    "cbStatusDiff",
    "timestampDiff",
    "tDiff",
    "timeFromLastChange",
    "delay",
]

best.append("class")

df = df[best]


df = (
    df.groupby("class")
    .apply(lambda x: x.sample(min(len(x), 20000), random_state=42))
    .sort_values(by="Time")  # ou qualquer coluna estável
    .reset_index(drop=True)
)

# tenta converter cada coluna para numérico; se não der, deixa como está
for col in df.columns:
    if col == "class":
        continue
    df[col] = pd.to_numeric(df[col], errors="ignore")

print("DTypes finais:\n", df.dtypes.value_counts())

# ------------------ 2. Separação de X e y ------------------
df = df.reset_index(drop=True)
y = df["class"].astype(str)  # string para o label encoder
X = df.drop(columns=["class"])


# ------------------ 3. Identificação de numérico vs categórico ------------------

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

print(
    f"{len(numeric_cols)} colunas numéricas e {len(categorical_cols)} colunas categóricas"
)

# ------------------ 4. Encoding ------------------

# 4.1 label-encode y
le = LabelEncoder()
y_true = le.fit_transform(y)

for i, class_name in enumerate(le.classes_):
    print(f"{i} => {class_name}")


y_true_bin = []
# y_true_bin = [0 if y == 6 else 1 for y in y_true]


# 4.2 ordinal-encode as categóricas (cada valor vira um inteiro)
ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_cat = X[categorical_cols].fillna("##MISSING##")
X_cat_enc = pd.DataFrame(
    ord_enc.fit_transform(X_cat), columns=categorical_cols, index=X.index
)

# 4.3 junta tudo num DataFrame só
X_num = X[numeric_cols]
X_proc = pd.concat([X_num, X_cat_enc], axis=1)
print("Dimensão de X após encoding:", X_proc.shape)


feature_names = X_proc.columns.tolist()

# Cria uma máscara booleana onde a classe não é 4 nem 5
mask = ~np.isin(y_true, ZERO_DAY_CLASSES)
mask2 = np.isin(y_true, ZERO_DAY_CLASSES)

X_zero_day = X_proc[mask2]
y_zero_day = y_true[mask2]

# Filtra os dados sem zero day
X_proc_filtered = X_proc[mask]
y_true_filtered = y_true[mask]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_proc_filtered,
    y_true_filtered,
    test_size=0.1,
    random_state=42,
    stratify=y_true_filtered,
)

_, contagens = np.unique(y_test, return_counts=True)


X_zero_day_sample = X_zero_day.iloc[:max(contagens)] 
y_zero_day_sample = y_zero_day[:max(contagens)]

pdb.set_trace()

metric = metrics.ROCAUC()

# -----------------------------------------HST-----------------------------------------

# Inicializar o modelo de detecção de anomalias
model = compose.Pipeline(
    preprocessing.MinMaxScaler(),
    anomaly.HalfSpaceTrees(
        n_trees=10,  
        height=11, 
        window_size=100,  
        seed=42,
    ),
)

# Classificador
classifierModel = compose.Pipeline(
    preprocessing.StandardScaler(),
    ensemble.AdaBoostClassifier(model=tree.HoeffdingTreeClassifier(split_criterion="gini", max_depth=15), n_models=11, seed=42),
)


X_train = X_train.reset_index(drop=True).values
X_test = X_test.reset_index(drop=True).values

# y_train = y_train.reset_index(drop=True)
# y_test = y_test.reset_index(drop=True)

y_true = y_test
y_pred = []
all_scores = []


# Parâmetros da janela
WINDOW_SIZE = 1000
score_window = deque(maxlen=WINDOW_SIZE)
label_window = deque(maxlen=WINDOW_SIZE)

# Histórico geral
all_scores = []
all_labels = []
all_preds = []

best_threshold = 0.5  # inicial arbitrário

thresholds_over_time = []
threshold_steps = []

for i, x_row in enumerate(X_train):

    xi = {feature_names[j]: x_row[j] for j in range(len(x_row))}
    y_full = y_train[i]

    y_bin = 0 if y_full == NORMAL_CLASS else 1  # binariza (0 = normal, 1 = ataque)

    # Predição de score de anomalia
    score = model.score_one(xi)
    all_scores.append(score)
    all_labels.append(y_bin)

    # Detecta com threshold atual
    y_pred = 1 if score > best_threshold else 0
    all_preds.append(y_pred)

    # Treina detector se for normal
    if y_bin == 0:
        model.learn_one(xi)

    # Treina classificador se detectado como anomalia e é ataque rotulado
    if y_pred == 1 and y_bin == 1:
        classifierModel.learn_one(xi, y_full)

    # Atualiza janela
    score_window.append(score)
    label_window.append(y_bin)

    # Recalibra threshold a cada janela cheia
    if len(score_window) == WINDOW_SIZE and i % (WINDOW_SIZE * 10) == 0:
        print(f"[{i}] Score range: {min(score_window):.5f} to {max(score_window):.5f}")
        thresholds = np.linspace(min(score_window), max(score_window), num=50)
        f1s = [
            f1_score(label_window, [1 if s > t else 0 for s in score_window])
            for t in thresholds
        ]
        best_threshold = thresholds[np.argmax(f1s)]
        print(f"[{i}] Novo threshold recalibrado: {best_threshold:.4f}")
        thresholds_over_time.append(best_threshold)
        threshold_steps.append(i)


print(f"Threshold final {best_threshold}")
plt.figure(figsize=(10, 5))
plt.plot(threshold_steps, thresholds_over_time, marker="o", linestyle="-")
plt.xlabel("Instância (índice)")
plt.ylabel("Threshold recalibrado")
plt.title("Evolução do Threshold ao Longo do Tempo")
plt.grid(True)
plt.tight_layout()
plt.show()


attack_true = []
attack_pred = []

attack_zd_true = []
attack_zd_pred = []

all_labels_test= []
all_labels_pred_test = []


X_test_full = pd.concat(
    [pd.DataFrame(X_test, columns=X_proc.columns), X_zero_day_sample], axis=0
).reset_index(drop=True)

y_test_full = np.concatenate([y_test, y_zero_day_sample])

probs = []

for i, x_row in enumerate(X_test_full.values):
    xi = {feature_names[j]: x_row[j] for j in range(len(x_row))}
    y_full = y_test_full[i]

    y_bin = 0 if y_full == NORMAL_CLASS else 1
    score = model.score_one(xi)
    y_pred = 1 if score > best_threshold else 0
    
    all_labels_pred_test.append(y_pred)
    all_labels_test.append(y_bin)

    if y_pred == 1 and y_bin == 1:
        y_pred_class = classifierModel.predict_one(xi)

        # Armazena resultado real e previsto
        attack_true.append(y_full)
        attack_pred.append(y_pred_class)

        # Mapeia o rótulo real
        y_true_m = ZERO_DAY if y_full in ZERO_DAY_CLASSES else y_full

        probas = classifierModel.predict_proba_one(xi)
        max_prob = max(probas.values()) if probas else 0

      
        probs.append(max_prob)

        if max_prob < 0.70:
            y_pred_m = ZERO_DAY
        else:
            y_pred_m = y_pred_class

        attack_zd_true.append(y_true_m)
        attack_zd_pred.append(y_pred_m)



# Avaliação final
print("\n=== Avaliação final ===")
print("F1:", f1_score(all_labels_test, all_labels_pred_test))
print("Precision:", precision_score(all_labels_test, all_labels_pred_test))
print("Recall:", recall_score(all_labels_test, all_labels_pred_test))
print("Accuracy:", accuracy_score(all_labels_test, all_labels_pred_test))

# 8.6 relatório multi‐classe
labels_zd = KNOW_CLASSES + [ZERO_DAY]
target_names = [f"class_{c}" for c in KNOW_CLASSES] + ["zero_day"]

print("\n=== Relatório Multi-classe (Zero-Day) ===")
print(
    classification_report(
        attack_zd_true,
        attack_zd_pred,
        labels=labels_zd,
        target_names=target_names,
        zero_division=0,
    )
)

