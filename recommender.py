from itertools import combinations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "age_difference",
    "same_email_domain",
    "same_area_code",
    "same_college",
    "same_major",
    "same_state",
    "same_zip_code",
    "shared_interest_count",
]

USERS = None
FRIEND_SET = None
MODEL = None
SCALER = None


def load_data():
    users = pd.read_csv("data/users.csv", dtype={"zip_code": str})
    friendships = pd.read_csv("data/friendships.csv")
    return users, friendships


def get_age(birthdate):
    birthdate = pd.to_datetime(birthdate)
    today = pd.Timestamp.today()
    age = today.year - birthdate.year
    if (today.month, today.day) < (birthdate.month, birthdate.day):
        age -= 1
    return age


def email_domain(email):
    return email.split("@")[1].lower()


def area_code(phone):
    digits = ""
    for ch in str(phone):
        if ch.isdigit():
            digits += ch
    return digits[:3]


def interest_set(interests):
    answer = set()
    for item in str(interests).split(";"):
        if item.strip() != "":
            answer.add(item.strip().lower())
    return answer


def process_users(users):
    # Extra processed columns for the ML features.
    users["age"] = users["birthdate"].apply(get_age)
    users["email_domain"] = users["email"].apply(email_domain)
    users["area_code"] = users["phone"].apply(area_code)
    users["interest_set"] = users["interests"].apply(interest_set)
    return users


def friend_key(id1, id2):
    # This makes friendships double sided like Phase 2 myfriends.
    return tuple(sorted((int(id1), int(id2))))


def make_friend_set(friendships):
    friends = set()
    for row in friendships.itertuples(index=False):
        friends.add(friend_key(row.user_id_1, row.user_id_2))
    return friends


def pair_features(person1, person2):
    shared = person1.interest_set.intersection(person2.interest_set)

    return {
        "age_difference": abs(person1.age - person2.age),
        "same_email_domain": int(person1.email_domain == person2.email_domain),
        "same_area_code": int(person1.area_code == person2.area_code),
        "same_college": int(person1.college == person2.college),
        "same_major": int(person1.major == person2.major),
        "same_state": int(person1.state == person2.state),
        "same_zip_code": int(person1.zip_code == person2.zip_code),
        "shared_interest_count": len(shared),
    }


def make_pair_data(users, friend_set):
    rows = []
    people = list(users.itertuples(index=False))

    for person1, person2 in combinations(people, 2):
        row = {
            "user_id_1": int(person1.id),
            "user_id_2": int(person2.id),
            "label": int(friend_key(person1.id, person2.id) in friend_set),
        }
        row.update(pair_features(person1, person2))
        rows.append(row)

    return pd.DataFrame(rows)


def train_model(pair_data):
    X = pair_data[FEATURES]
    y = pair_data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
    )
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    return model, scaler, accuracy, precision, recall, f1


def reasons_for(user, candidate):
    reasons = []
    shared = sorted(user.interest_set.intersection(candidate.interest_set))

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
    if len(shared) > 0:
        reasons.append("shared interests: " + ", ".join(shared))

    if len(reasons) == 0:
        reasons.append("closest match from the model")

    return ", ".join(reasons)


def recommend_friends(user_id, top_n=3):
    user_id = int(user_id)
    user = USERS.loc[USERS["id"] == user_id].iloc[0]
    answers = []

    for candidate in USERS.itertuples(index=False):
        candidate_id = int(candidate.id)

        if candidate_id == user_id:
            continue
        if friend_key(user_id, candidate_id) in FRIEND_SET:
            continue

        one_row = pd.DataFrame([pair_features(user, candidate)])
        one_row = one_row[FEATURES]
        score = MODEL.predict_proba(SCALER.transform(one_row))[0][1]

        answers.append(
            {
                "person": candidate,
                "score": score,
                "reasons": reasons_for(user, candidate),
            }
        )

    answers.sort(key=lambda item: item["score"], reverse=True)
    return answers[:top_n]


def print_recommendations(user_id):
    user = USERS.loc[USERS["id"] == user_id].iloc[0]
    print(f"\nRecommendations for {user.first_name} {user.last_name}:")

    recs = recommend_friends(user_id)
    count = 1
    for rec in recs:
        person = rec["person"]
        print(f"{count}. {person.first_name} {person.last_name} ({person.code_name})")
        print(f"   Score: {rec['score']:.2f}")
        print(f"   Reasons: {rec['reasons']}")
        count += 1


def main():
    global USERS, FRIEND_SET, MODEL, SCALER

    users, friendships = load_data()
    USERS = process_users(users)
    FRIEND_SET = make_friend_set(friendships)
    pair_data = make_pair_data(USERS, FRIEND_SET)

    print(f"Loaded {len(USERS)} users.")
    print(f"Loaded {len(friendships)} known friendships.")
    print(f"Generated {len(pair_data)} user pairs.")

    MODEL, SCALER, accuracy, precision, recall, f1 = train_model(pair_data)

    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    print_recommendations(1)
    print_recommendations(15)
    print_recommendations(16)


if __name__ == "__main__":
    main()
