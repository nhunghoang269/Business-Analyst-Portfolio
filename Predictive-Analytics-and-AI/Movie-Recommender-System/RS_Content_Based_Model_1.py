import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# ---------------------------
# Inputs
# ---------------------------
def ask_int(prompt, default):
    try:
        txt = input(f"{prompt} [{default}]: ").strip()
        return int(txt) if txt else default
    except:
        print("Invalid input; using default.")
        return default

def ask_str(prompt, default):
    txt = input(f"{prompt} [{default}]: ").strip()
    return txt if txt else default

MIN_TRAIN = ask_int("Minimum number of training ratings per user, default is", 3)

genre_input = ask_str(
    "Genres to keep (comma-separated), or 'all' for all genres",
    "all"
)

reg_input = ask_str(
    "Regularisation (none, l2, l1). 'none' = no regularisation",
    "none"
)

C_value = float(input("Enter C value for regularisation strength (default=1.0): ") or 1.0)

# ---------------------------
# Load data
# ---------------------------
ratings = pd.read_csv("\...ratings.csv")
movies = pd.read_csv("C:\...movies.csv")

genre_ohe_full = movies["genres"].str.get_dummies(sep="|")
all_genres_list = genre_ohe_full.columns.tolist()

if genre_input.lower() == "all":
    genre_ohe = genre_ohe_full.copy()
    kept_genres = all_genres_list
else:
    requested = [g.strip() for g in genre_input.split(",") if g.strip()]
    # keep only valid genres
    kept_genres = [g for g in requested if g in all_genres_list]
    if not kept_genres:
        print("No valid genres found in input; falling back to ALL genres.")
        kept_genres = all_genres_list
        genre_ohe = genre_ohe_full.copy()
    else:
        genre_ohe = genre_ohe_full[kept_genres]

X_movies = genre_ohe.astype(float)
X_movies.index = movies["movieId"]

movieid_to_title = dict(zip(movies["movieId"], movies["title"]))
all_movie_ids = set(X_movies.index.tolist())

print("\nKept genres:", kept_genres)

def split_by_user(df, test_frac=0.2, seed=42):
    train_parts, test_parts = [], []
    for uid, g in df.groupby("userId"):
        g = g.sample(frac=1.0, random_state=seed)  # shuffle
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

# ---------------------------
# Evaluation
# ---------------------------
def apk(actual, predicted, k=10):
    if k == 0:
        return 0.0
    predicted = predicted[:k]
    score = 0.0
    hits = 0.0
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

# ---------------------------
# Build Model
# ---------------------------

def make_logreg(reg_choice, C_value):
    rc = reg_choice.lower()
    if rc == "none":
        try:
            return LogisticRegression(max_iter=500, penalty=None, solver="lbfgs")
        except Exception:
            print("penalty='none' not supported; using large C to approximate no regularisation.")
            return LogisticRegression(max_iter=500, penalty="l2", C=1e6, solver="lbfgs")
    elif rc == "l1":
        # L1 regularisation (sparse weights)
        return LogisticRegression(max_iter=700, penalty="l1", solver="saga", C=C_value)
    else:
        # Default to L2 regularisation
        return LogisticRegression(max_iter=500, penalty="l2", solver="lbfgs", C=C_value)

# create a template for reference
clf_template = make_logreg(reg_input, C_value)

K5, K10 = 5, 10
LIKE_THRESHOLD = 4.0

train_by_user = {uid: g for uid, g in train.groupby("userId")}
test_by_user  = {uid: g for uid, g in test.groupby("userId")}

actual_k5, preds_k5 = [], []
actual_k10, preds_k10 = [], []

users_evaluated = 0
skipped_too_few = 0
skipped_one_class = 0

for uid, g_train in train_by_user.items():
    g_test = test_by_user.get(uid)
    if g_test is None or g_test.empty:
        continue

    # enforce student-selected minimum train size
    if len(g_train) < MIN_TRAIN:
        skipped_too_few += 1
        continue

    y = (g_train["rating"] >= LIKE_THRESHOLD).astype(int).values
    if y.sum() == 0 or y.sum() == len(y):
        skipped_one_class += 1
        continue

    mids_train = [m for m in g_train["movieId"].tolist() if m in all_movie_ids]
    if len(mids_train) == 0:
        skipped_too_few += 1
        continue

    X_u = X_movies.loc[mids_train]
    y_u = (g_train[g_train["movieId"].isin(mids_train)]["rating"] >= LIKE_THRESHOLD).astype(int).values

    clf = make_logreg(reg_input, C_value)
    clf.fit(X_u, y_u)

    # candidates: unseen in TRAIN
    candidates = list(all_movie_ids - set(mids_train))
    if not candidates:
        continue

    probs = clf.predict_proba(X_movies.loc[candidates])[:, 1]
    order = np.argsort(-probs)
    ranked = [candidates[i] for i in order]

    actual_likes = g_test[g_test["rating"] >= LIKE_THRESHOLD]["movieId"].tolist()

    actual_k5.append(actual_likes);  preds_k5.append(ranked[:K5])
    actual_k10.append(actual_likes); preds_k10.append(ranked[:K10])

    users_evaluated += 1

map5  = mapk(actual_k5,  preds_k5,  k=K5) if users_evaluated > 0 else 0.0
map10 = mapk(actual_k10, preds_k10, k=K10) if users_evaluated > 0 else 0.0

# ---------------------------
# Report
# ---------------------------
print("\n================= RESULTS =================")
print(f"Min train per user     : {MIN_TRAIN}")
print(f"Kept genres            : {kept_genres}")
print(f"Regularisation         : {reg_input}")
print("-------------------------------------------")
print(f"Users evaluated        : {users_evaluated}")
print(f"Skipped: too few train : {skipped_too_few}")
print(f"Skipped: one-class y   : {skipped_one_class}")
print(f"MAP@5                  : {map5:.4f}")
print(f"MAP@10                 : {map10:.4f}")
