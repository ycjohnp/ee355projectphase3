# TrojanBook Phase 3 Extra Credit: Friend Recommendation System

## 1. Project Description

Phase 3 adds a basic friend recommendation program for TrojanBook. It is in Python and uses a small csv database. The idea is that if two people have similar profile information, they might be good people to recommend as friends.

The Python code is separate from the C++ files, but it uses the same TrojanBook idea from phase 1 and phase 2.

## 2. How This Extends Phase 1 and Phase 2

In phase 1, the project had `Person` objects and a `Network` class. The people had names, birthdates, emails, and phone numbers.

In phase 2, the project added friends. Each person had `myfriends`, which was a `vector<Person*>`. When two people became friends, both sides got updated. Phase 2 also used `codeName`, made from first name plus last name, with spaces removed and lowercase.

This part uses the same idea but in csv files. `users.csv` has `code_name` using the same Phase 2 rule, and `friendships.csv` stores which users are already friends. Friend pairs are labeled as `1`, and non-friend pairs are labeled as `0`.

## 3. Database Description

The database is made up for this phase.

The old information from the other phases is still included:

- birthdate
- email
- phone

Extra fields added:

- college
- major
- state
- zip code
- interests

Files:

- `data/users.csv`
- `data/friendships.csv`

## 4. Features Used

For each possible pair of users, the code calculates:

- age difference
- same email domain
- same phone area code
- same college
- same major
- same state
- same zip code
- number of shared interests

The code also does some data processing, like getting the email domain after `@`, getting the first 3 phone digits, calculating age, and splitting interests by semicolons.

## 5. Model Used

Logistic Regression from scikit-learn is used. The model learns from the pairs of users and tries to predict if two users are friends or not.

The program makes all possible pairs of users. If the pair is in `friendships.csv`, the label is `1`. Otherwise it is `0`.

The data is split into 90% training and 10% testing with `random_state=42`. Since there are more non-friend pairs than friend pairs, balanced class weights are used. The features are also scaled before training.

The program prints:

- accuracy
- precision
- recall
- F1 score
- top 3 friend recommendations for a few users

## 6. How to Run

Run:

```bash
pip install -r requirements.txt
python recommender.py
```

## 7. Example Output

Example of what it prints:

```text
Loaded 18 users.
Loaded 25 known friendships.
Generated 153 user pairs.

Model Evaluation:
Accuracy: 0.88
Precision: 0.50
Recall: 0.50
F1 Score: 0.50

Recommendations for Julia Louis-Dreyfus:
1. Phoebe Buffay (phoebebuffay)
   Score: 0.60
   Reasons: shared interests: music
2. Miles Morales (milesmorales)
   Score: 0.59
   Reasons: shared interests: music, robotics
3. Rachel Green (rachelgreen)
   Score: 0.54
   Reasons: shared interests: music
```

The program also prints 3 recommendations for Peter Parker and Miles Morales.

## 8. Limitations

The data set is small, so the scores are just for this project.

This only uses profile similarity. It does not use mutual friends or a bigger social graph.

## 9. Future Improvements

Some possible improvements would be:

- add more users
- include mutual friends as a feature
- try another simple model
- let the user type in a `code_name` and get recommendations
