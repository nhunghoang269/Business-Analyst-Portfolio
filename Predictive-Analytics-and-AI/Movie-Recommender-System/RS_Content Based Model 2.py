import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack, csr_matrix

# ---------------------------
# Inputs
# ---------------------------
def ask_int(prompt, default, min_val=None, max_val=None):
    """Ask for an integer input with optional range."""
    while True:
        try:
            txt = input(f"{prompt} [{default}]: ").strip()
            if not txt:
                return default
            val = int(txt)
            if min_val is not None and val < min_val:
                print(f"Please enter a value >= {min_val}.")
                continue
            if max_val is not None and val > max_val:
                print(f"Please enter a value <= {max_val}.")
                continue
            return val
        except ValueError:
            print("Please enter a valid integer.")

def ask_str(prompt, default):
    txt = input(f"{prompt} [{default}]: ").strip()
    return txt if txt else default

# Ask for inputs
MIN_TRAIN = ask_int("Enter minimum training ratings per user (3–15)", 3, 3, 15)
genre_input = ask_str("Enter genres to keep (comma-separated) or 'all' for all", "all")
reg_input = ask_str("Choose regularisation (none, l2, l1)", "l2")

try:
    C_value = float(input("Enter C value for regularisation strength (default=1.0): ") or 1.0)
except ValueError:
    print("Invalid input, using default C=1.0")
    C_value = 1.0

# ---------------------------
# Load data
# ---------------------------
ratings = pd.read_csv("C:\...ratings.csv")
movies = pd.read_csv("C:\...movies.csv")

genre_ohe_full = movies["genres"].str.get_dummies(sep="|")
all_genres = genre_ohe_full.columns.tolist()

if genre_input.lower() == "all":
    genre_ohe = genre_ohe_full.copy()
    kept_genres = all_genres
else:
    requested = [g.strip() for g in genre_input.split(",") if g.strip()]
    kept_genres = [g for g in requested if g in all_genres]
    if not kept_genres:
        print("No valid genres entered, using all genres.")
        genre_ohe = genre_ohe_full.copy()
        kept_genres = all_genres
    else:
        genre_ohe = genre_ohe_full[kept_genres]

genre_ohe = genre_ohe.astype(float)
genre_ohe.index = movies["movieId"]

movieid_to_title = dict(zip(movies["movieId"], movies["title"]))
all_movie_ids = set(genre_ohe.index.tolist())

print("\nKept genres:", kept_genres)

def split_by_user(df, test_frac=0.2, seed=42):
    train_parts, test_parts = [], []
    for uid, g in df.groupby("userId"):
        g = g.sample(frac=1.0, random_state=seed)
        n = len(g)
        if n <= 1:
            train_parts.append(g)
            continue
        n_test = max(1, int(round(n * test_frac)))
        test_parts.append(g.iloc[:n_test])
        train_parts.append(g.iloc[n_test:])
    train_df = pd.concat(train_parts, ignore_index=True)
    test_df  = pd.concat(test_parts,  ignore_index=True)
    return train_df, test_df

train, test = split_by_user(ratings, test_frac=0.2, seed=42)

# Drop users with too few training samples
train_counts = train.groupby("userId").size()
valid_users = train_counts[train_counts >= MIN_TRAIN].index
train = train[train["userId"].isin(valid_users)]
test  = test[test["userId"].isin(valid_users)]

print(f"\nUsers with at least {MIN_TRAIN} ratings in training: {len(valid_users)}")

train = train[train["movieId"].isin(all_movie_ids)].copy()
test  = test[test["movieId"].isin(all_movie_ids)].copy()

train["like"] = (train["rating"] >= 4.0).astype(int)

user_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
X_user_train = user_enc.fit_transform(train[["userId"]])

X_genre_train = csr_matrix(genre_ohe.loc[train["movieId"]].values)

X_train = hstack([X_user_train, X_genre_train], format="csr")
y_train = train["like"].values

# ---------------------------
# Build Model
# ---------------------------
def make_logreg(reg_choice, C_value):
    rc = reg_choice.lower()
    if rc == "none":
        try:
            return LogisticRegression(max_iter=500, penalty=None, solver="lbfgs")
        except Exception:
            print("penalty='none' not supported; using large C for near-no regularisation.")
            return LogisticRegression(max_iter=500, penalty="l2", C=1e6, solver="lbfgs")
    elif rc == "l1":
        return LogisticRegression(max_iter=700, penalty="l1", solver="saga", C=C_value)
    else:
        return LogisticRegression(max_iter=500, penalty="l2", solver="lbfgs", C=C_value)

clf = make_logreg(reg_input, C_value)
print("\nTraining...")
clf.fit(X_train, y_train)

# ---------------------------
# Evaluation
# ---------------------------
def apk(actual, predicted, k=10):
    if k == 0:
        return 0.0
    predicted = predicted[:k]
    score, hits = 0.0, 0.0
    actual_set = set(actual)
    for i, p in enumerate(predicted):
        if p in actual_set:
            hits += 1.0
            score += hits / (i + 1.0)
    return score / min(len(actual), k) if actual else 0.0

def mapk(actual_lists, predicted_lists, k=10):
    if len(actual_lists) == 0:
        return 0.0
    return sum(apk(a, p, k) for a, p in zip(actual_lists, predicted_lists)) / len(actual_lists)

K5, K10 = 5, 10
LIKE_THRESHOLD = 4.0

train_by_user = {uid: g for uid, g in train.groupby("userId")}
test_by_user  = {uid: g for uid, g in test.groupby("userId")}

actual_k5, preds_k5 = [], []
actual_k10, preds_k10 = [], []

users_evaluated = 0

for uid, g_test in test_by_user.items():
    seen_train = set(train_by_user.get(uid, pd.DataFrame()).get("movieId", pd.Series([], dtype=int)).tolist())
    candidates = list(all_movie_ids - seen_train)
    if not candidates:
        continue

    X_user_row = user_enc.transform(pd.DataFrame({"userId": [uid]}))
    X_user_block = csr_matrix(np.repeat(X_user_row.toarray(), len(candidates), axis=0))
    X_genre_block = csr_matrix(genre_ohe.loc[candidates].values)
    X_cand = hstack([X_user_block, X_genre_block], format="csr")

    probs = clf.predict_proba(X_cand)[:, 1]
    order = np.argsort(-probs)
    ranked = [candidates[i] for i in order]

    actual_likes = g_test[g_test["rating"] >= LIKE_THRESHOLD]["movieId"].tolist()
    actual_k5.append(actual_likes); preds_k5.append(ranked[:K5])
    actual_k10.append(actual_likes); preds_k10.append(ranked[:K10])
    users_evaluated += 1

map5  = mapk(actual_k5, preds_k5, 5) if users_evaluated > 0 else 0.0
map10 = mapk(actual_k10, preds_k10, 10) if users_evaluated > 0 else 0.0

# ---------------------------
# Report
# ---------------------------
print("\n================ RESULTS ================")
print(f"Min train per user : {MIN_TRAIN}")
print(f"Genres kept        : {kept_genres}")
print(f"Regularisation     : {reg_input}")
print(f"C value            : {C_value}")
print("----------------------------------------")
print(f"Users evaluated    : {users_evaluated}")
print(f"MAP@5              : {map5:.4f}")
print(f"MAP@10             : {map10:.4f}")
