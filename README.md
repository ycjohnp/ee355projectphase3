# TrojanBook Phase 3 Extra Credit: Friend Recommendation System

## 1. Project Description

This Phase 3 project adds a simple friend recommendation system to TrojanBook. It uses a small TrojanBook-style database of users, known friendships, and extra profile information. The Python program learns from existing friendships and then recommends new people who may be worth connecting to.

The project is standalone inside `ee355projectphase3/`, so it does not need to compile or directly connect to the C++ files from Phase 1 and Phase 2.

## 2. How This Extends Phase 1 and Phase 2

Phase 1 implemented the core TrojanBook `Person` and `Network` database. A `Person` stored first name, last name, birthdate, email, and phone number. The `Network` class stored multiple people and could load, save, add, remove, and print users.

Phase 2 added friendship connections between `Person` objects. Each `Person` has a `vector<Person*>` called `myfriends`, and the connect option makes friendship double-sided by adding each person to the other person's friend list. Phase 2 also uses `codeName`, which combines first name and last name, removes spaces, and converts the result to lowercase.

Phase 3 mirrors those ideas in Python. `users.csv` includes a `code_name` column that follows the same idea as Phase 2 `codeName`. `friendships.csv` stores the existing friend network. Existing friendships are positive training examples, and non-friend pairs are negative training examples. The model learns which profile similarities make two users more likely to be friends, then recommends new people who are not already friends with the selected user.

## 3. Database Description

The database was created for this project, which is allowed by the Phase 3 instructions. It includes the original fields from the earlier phases:

- `birthdate`
- `email`
- `phone`

It also enlarges the database with extra profile fields:

- `college`
- `major`
- `state`
- `zip_code`
- `interests`

The data files are:

- `data/users.csv`: 18 TrojanBook users with profile information
- `data/friendships.csv`: 25 known double-sided friendships

## 4. Features Used

For every possible unordered pair of users, the program creates these content-based recommendation features:

- `age_difference`
- `same_email_domain`
- `same_area_code`
- `same_college`
- `same_major`
- `same_state`
- `same_zip_code`
- `shared_interest_count`

The program also performs data processing by calculating approximate age from birthdate, extracting the email domain after `@`, extracting the first three phone digits as the area code, and splitting interests on semicolons.

## 5. Model Used

The recommendation model is `LogisticRegression` from scikit-learn. The code uses standard scaling and balanced class weights because there are more non-friend pairs than friend pairs. It is a simple content-based filtering model because it learns from profile similarity features instead of using movie ratings or a neural network.

The program generates all possible user pairs. A pair is labeled `1` if the users are already friends and `0` if they are not friends. The data is split into 90% training data and 10% testing data using `train_test_split(test_size=0.10, random_state=42)`.

The model is evaluated with:

- accuracy
- precision
- recall
- F1 score

## 6. How to Run

From inside `ee355projectphase3/`, run:

```bash
pip install -r requirements.txt
python recommender.py
```

## 7. Example Output

The exact scores may vary slightly depending on installed package versions, but the output will look like this:

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
```

The program prints recommendation sections for:

- Julia Louis-Dreyfus
- Peter Parker
- Miles Morales

Each recommendation includes the person's name, `code_name`, model probability score, and readable reasons such as same college, same major, same state, same email domain, same phone area code, same zip code, and shared interests.

## 8. Limitations

This is a beginner-friendly extra credit implementation, so the database is small and synthetic. Because there are only 18 users, the test set is also small. The model is useful for demonstrating the recommendation idea, but it should not be interpreted as a production-level social network recommender.

The model uses simple profile similarity features. It does not use message history, friendship timing, mutual friend counts, graph algorithms, or advanced machine learning.

## 9. Future Improvements

Future versions could add more users, include mutual friend counts from the friendship graph, support friend recommendations by `code_name`, store output in a file for reports, or compare Logistic Regression against other simple models such as Decision Trees or Random Forests.
