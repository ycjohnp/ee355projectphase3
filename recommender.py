from itertools import combinations
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path(__file__).parent / "data"
USERS_FILE = DATA_DIR / "users.csv"
FRIENDSHIPS_FILE = DATA_DIR / "friendships.csv"
FEATURE_COLUMNS = [
    "age_difference",
    "same_email_domain",
    "same_area_code",
    "same_college",
    "same_major",
    "same_state",
    "same_zip_code",
    "shared_interest_count",
]


def load_data():
    """Load the TrojanBook-style users and known friend connections."""
    users = pd.read_csv(USERS_FILE, dtype={"zip_code": str})
    friendships = pd.read_csv(FRIENDSHIPS_FILE)
    return users, friendships


def calculate_age(birthdate):
    """Approximate age from a birthdate string."""
    birthdate = pd.to_datetime(birthdate)
    today = pd.Timestamp.today()
    return today.year - birthdate.year - (
        (today.month, today.day) < (birthdate.month, birthdate.day)
    )


def get_email_domain(email):
    return email.split("@", 1)[1].lower().strip()


def get_area_code(phone):
    digits = "".join(character for character in str(phone) if character.isdigit())
    return digits[:3]


def parse_interests(interests):
    return {
        interest.strip().lower()
        for interest in str(interests).split(";")
        if interest.strip()
    }


def add_processed_columns(users):
    """Create simple processed fields used by pairwise recommendation features."""
    users = users.copy()
    users["age"] = users["birthdate"].apply(calculate_age)
    users["email_domain"] = users["email"].apply(get_email_domain)
    users["area_code"] = users["phone"].apply(get_area_code)
    users["interest_set"] = users["interests"].apply(parse_interests)
    return users


def friendship_key(user_id_1, user_id_2):
    """Store friendships as unordered pairs, matching Phase 2 double-sided friends."""
    return tuple(sorted((int(user_id_1), int(user_id_2))))


def build_friendship_set(friendships):
    return {
        friendship_key(row.user_id_1, row.user_id_2)
        for row in friendships.itertuples(index=False)
    }


def make_pair_features(user_1, user_2):
    shared_interests = user_1.interest_set.intersection(user_2.interest_set)

    return {
        "age_difference": abs(user_1.age - user_2.age),
        "same_email_domain": int(user_1.email_domain == user_2.email_domain),
        "same_area_code": int(user_1.area_code == user_2.area_code),
        "same_college": int(user_1.college == user_2.college),
        "same_major": int(user_1.major == user_2.major),
        "same_state": int(user_1.state == user_2.state),
        "same_zip_code": int(user_1.zip_code == user_2.zip_code),
        "shared_interest_count": len(shared_interests),
    }


def build_pair_dataset(users, friendship_set):
    """Generate all unordered pairs and label friends as 1, non-friends as 0."""
    rows = []
    user_records = list(users.itertuples(index=False))

    for user_1, user_2 in combinations(user_records, 2):
        row = {
            "user_id_1": int(user_1.id),
            "user_id_2": int(user_2.id),
            "label": int(friendship_key(user_1.id, user_2.id) in friendship_set),
        }
        row.update(make_pair_features(user_1, user_2))
        rows.append(row)

    return pd.DataFrame(rows)


def train_model(pair_dataset):
    X = pair_dataset[FEATURE_COLUMNS]
    y = pair_dataset["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
        ),
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }
    return model, metrics


def explain_recommendation(user, candidate):
    reasons = []
    shared_interests = sorted(user.interest_set.intersection(candidate.interest_set))

    if user.college == candidate.college:
        reasons.append("same college")
    if user.major == candidate.major:
        reasons.append("same major")
    if user.state == candidate.state:
        reasons.append("same state")
    if user.zip_code == candidate.zip_code:
        reasons.append("same zip code")
    if user.email_domain == candidate.email_domain:
        reasons.append("same email domain")
    if user.area_code == candidate.area_code:
        reasons.append("same phone area code")
    if shared_interests:
        reasons.append("shared interests: " + ", ".join(shared_interests))

    if not reasons:
        reasons.append("closest overall profile match from the model")

    return ", ".join(reasons)


def recommend_friends(user_id, top_n=3, users=None, friendship_set=None, model=None):
    """Recommend non-friends with the highest learned friendship probability."""
    if users is None or friendship_set is None or model is None:
        raise ValueError("users, friendship_set, and model are required")

    user_id = int(user_id)
    selected_user = users.loc[users["id"] == user_id].iloc[0]
    candidates = []

    for candidate in users.itertuples(index=False):
        candidate_id = int(candidate.id)
        if candidate_id == user_id:
            continue
        if friendship_key(user_id, candidate_id) in friendship_set:
            continue

        features = pd.DataFrame([make_pair_features(selected_user, candidate)])
        probability = model.predict_proba(features[FEATURE_COLUMNS])[0][1]
        candidates.append(
            {
                "candidate": candidate,
                "score": probability,
                "reasons": explain_recommendation(selected_user, candidate),
            }
        )

    candidates.sort(key=lambda recommendation: recommendation["score"], reverse=True)
    return candidates[:top_n]


def print_evaluation(metrics):
    print("\nModel Evaluation:")
    print(f"Accuracy: {metrics['accuracy']:.2f}")
    print(f"Precision: {metrics['precision']:.2f}")
    print(f"Recall: {metrics['recall']:.2f}")
    print(f"F1 Score: {metrics['f1']:.2f}")


def print_recommendations(user_id, users, friendship_set, model):
    user = users.loc[users["id"] == user_id].iloc[0]
    print(f"\nRecommendations for {user.first_name} {user.last_name}:")

    recommendations = recommend_friends(
        user_id,
        top_n=3,
        users=users,
        friendship_set=friendship_set,
        model=model,
    )
    for index, recommendation in enumerate(recommendations, start=1):
        candidate = recommendation["candidate"]
        print(f"{index}. {candidate.first_name} {candidate.last_name} ({candidate.code_name})")
        print(f"   Score: {recommendation['score']:.2f}")
        print(f"   Reasons: {recommendation['reasons']}")


def main():
    users, friendships = load_data()
    users = add_processed_columns(users)
    friendship_set = build_friendship_set(friendships)
    pair_dataset = build_pair_dataset(users, friendship_set)

    print(f"Loaded {len(users)} users.")
    print(f"Loaded {len(friendships)} known friendships.")
    print(f"Generated {len(pair_dataset)} user pairs.")

    model, metrics = train_model(pair_dataset)
    print_evaluation(metrics)

    print_recommendations(1, users, friendship_set, model)
    print_recommendations(15, users, friendship_set, model)
    print_recommendations(16, users, friendship_set, model)


if __name__ == "__main__":
    main()
