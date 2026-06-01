import pandas as pd
from sklearn.model_selection import train_test_split

ratings = pd.read_csv("\...ratings.csv")
movies = pd.read_csv("\...movies.csv")

# For reproducibility
ratings = ratings.sample(frac=1, random_state=42).reset_index(drop=True)

# 80% train, 20% test
train, test = train_test_split(ratings, test_size=0.2, random_state=42)

# Compute how many ratings and average rating per movie
movie_stats = train.groupby('movieId').agg(
    rating_count=('rating', 'count'),
    average_rating=('rating', 'mean')
).reset_index()

# Merge titles
movie_stats = movie_stats.merge(movies, on='movieId')

# Filter to only movies with enough ratings
movie_stats = movie_stats[movie_stats['rating_count'] >= 10]

# Sort by popularity (rating count)
popular_movies = movie_stats.sort_values('rating_count', ascending=False)

def recommend_popular(n=10):
    return list(popular_movies['movieId'].head(n))

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k for one user.
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual_list, predicted_list, k=10):
    """
    Mean average precision at k over all users.
    """
    return sum(apk(a, p, k) for a, p in zip(actual_list, predicted_list)) / len(actual_list)

# Get "liked" movies per user in the test set
test_likes = test[test['rating'] >= 4.0].groupby('userId')['movieId'].apply(list)

# Recommended movies (same list for everyone)
recommended = recommend_popular(10)

# Build prediction list for each user
predicted_list = [recommended] * len(test_likes)

# Evaluate MAP@5 and MAP@10
map5 = mapk(test_likes, predicted_list, k=5)
map10 = mapk(test_likes, predicted_list, k=10)

print(f"MAP@5 : {map5:.4f}")
print(f"MAP@10: {map10:.4f}")
