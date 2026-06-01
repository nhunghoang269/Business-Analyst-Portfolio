import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Inputs
# ---------------------------
def ask_int(prompt, default, min_val=None, max_val=None):
    while True:
        txt = input(f"{prompt} [{default}]: ").strip()
        if not txt:
            return default
        try:
            val = int(txt)
            if min_val is not None and val < min_val:
                print(f"Enter a value >= {min_val}."); continue
            if max_val is not None and val > max_val:
                print(f"Enter a value <= {max_val}."); continue
            return val
        except ValueError:
            print("Please enter an integer.")

def ask_float(prompt, default):
    while True:
        txt = input(f"{prompt} [{default}]: ").strip()
        if not txt:
            return default
        try:
            return float(txt)
        except ValueError:
            print("Please enter a number.")

def ask_topk(prompt, default_str="50"):
    txt = input(f"{prompt} [{default_str}]: ").strip().lower()
    if not txt:
        txt = default_str
    if txt == "all":
        return None
    try:
        k = int(txt)
        if k < 1:
            print("Top-K must be >= 1. Using 50.")
            return 50
        return k
    except ValueError:
        print("Invalid input. Using 50.")
        return 50

def ask_yesno(prompt, default_yes=True):
    d = "y" if default_yes else "n"
    txt = input(f"{prompt} [y/n, default={d}]: ").strip().lower()
    if not txt:
        return default_yes
    return txt in ("y", "yes")

MIN_TRAIN       = ask_int("Minimum training ratings per user (3–15), default is", 3, 3, 15)
TOPK_NEIGHBORS  = ask_topk("Top-K similar items to use (enter a number or 'all'), default is", "50")
LIKE_THRESHOLD  = ask_float("Like threshold (rating >= this is 'relevant'), default is", 4.0)
USE_MEAN_CENTER = ask_yesno("Apply user mean-centering before similarity?", True)

# ---------------------------
# Load data
# ---------------------------
ratings = pd.read_csv("\...ratings.csv")
movies = pd.read_csv("\...movies.csv")
movieid_to_title = dict(zip(movies["movieId"], movies["title"]))

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
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(test_parts,  ignore_index=True)
    )

train, test = split_by_user(ratings, test_frac=0.2, seed=42)


train_counts = train.groupby("userId").size()
valid_users = set(train_counts[train_counts >= MIN_TRAIN].index)
train = train[train["userId"].isin(valid_users)].copy()
test  = test[test["userId"].isin(valid_users)].copy()

# ---------------------------
# Build Model
# ---------------------------
ui = train.pivot_table(index="userId", columns="movieId", values="rating", fill_value=0.0)
user_ids  = ui.index.to_list()
movie_ids = ui.columns.to_list()
movie_idx = {m:i for i,m in enumerate(movie_ids)}
M = ui.values.astype(float)  # shape [num_users, num_items]

# Optional mean-centering by user
if USE_MEAN_CENTER:
    counts = (M != 0).sum(axis=1, keepdims=True)
    means  = np.divide(M.sum(axis=1, keepdims=True), np.maximum(counts, 1), where=(counts>0))
    M_centered = np.where(M != 0, M - means, 0.0)
else:
    M_centered = M

S = cosine_similarity(M_centered.T)  # items × items
np.fill_diagonal(S, 0.0)

if TOPK_NEIGHBORS is not None:
    K = min(TOPK_NEIGHBORS, S.shape[1]-1)
    to_zero = np.argpartition(-S, K, axis=1)[:, K+1:]
    S[np.arange(S.shape[0])[:, None], to_zero] = 0.0

# ---------------------------
# Evaluation
# ---------------------------
def apk(actual, predicted, k=10):
    if k == 0:
        return 0.0
    predicted = predicted[:k]
    score = 0.0
    hits = 0.0
    A = set(actual)
    for i, p in enumerate(predicted):
        if p in A:
            hits += 1.0
            score += hits / (i + 1.0)
    return score / min(len(actual), k) if actual else 0.0

def mapk(actual_lists, predicted_lists, k=10):
    if len(actual_lists) == 0:
        return 0.0
    return sum(apk(a, p, k) for a, p in zip(actual_lists, predicted_lists)) / len(actual_lists)

# ---------------------------
# Scoring
# ---------------------------
K5, K10 = 5, 10
actual_k5, preds_k5 = [], []
actual_k10, preds_k10 = [], []
users_evaluated = 0

train_by_user = {uid: g for uid, g in train.groupby("userId")}
test_by_user  = {uid: g for uid, g in test.groupby("userId")}

movie_set = set(movie_ids)

for uid in sorted(train_by_user.keys()):
    g_tr = train_by_user.get(uid)
    g_te = test_by_user.get(uid)
    if g_te is None or g_te.empty:
        continue

    rated_pairs = [(movie_idx[m], r) for m, r in zip(g_tr["movieId"], g_tr["rating"]) if m in movie_set]
    if not rated_pairs:
        continue

    scores = np.zeros(len(movie_ids), dtype=float)
    seen = set()
    for j, r in rated_pairs:
        scores += r * S[j, :]
        seen.add(j)

    if seen:
        scores[list(seen)] = -np.inf

    # Rank candidates
    order = np.argsort(-scores)
    ranked_ids = [movie_ids[i] for i in order if np.isfinite(scores[i])]

    # Ground-truth likes from TEST
    actual_likes = g_te[g_te["rating"] >= LIKE_THRESHOLD]["movieId"].tolist()

    actual_k5.append(actual_likes);  preds_k5.append(ranked_ids[:K5])
    actual_k10.append(actual_likes); preds_k10.append(ranked_ids[:K10])
    users_evaluated += 1

# ---------------------------
# Report
# ---------------------------
map5  = mapk(actual_k5,  preds_k5,  5) if users_evaluated > 0 else 0.0
map10 = mapk(actual_k10, preds_k10, 10) if users_evaluated > 0 else 0.0

print("\n===== RESULTS =====")
print(f"Users evaluated        : {users_evaluated}")
print(f"Min train per user     : {MIN_TRAIN}")
print(f"Top-K neighbors        : {'all' if TOPK_NEIGHBORS is None else TOPK_NEIGHBORS}")
print(f"Like threshold         : {LIKE_THRESHOLD}")
print(f"Mean-centering applied : {USE_MEAN_CENTER}")
print(f"MAP@5                  : {map5:.4f}")
print(f"MAP@10                 : {map10:.4f}")
