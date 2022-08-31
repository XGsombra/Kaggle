import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler

# constants
TRAIN_DATA_PATH = "titanic\data\\train.csv"
TEST_DATA_PATH = "titanic\data\\test.csv"
OUTPUT_PATH = "titanic\data\\submission.csv"
Y_LABEL = "Survived"
CV_NUM = 10

# Load data
train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)

# Preprocessing
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = pd.get_dummies(train_df[features])
y = train_df[Y_LABEL]
X_test = pd.get_dummies(test_df[features])
# X.loc[(X.Age.isnull()), 'Age'] = X.Age.dropna().mean()

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# deal with missing values


# Classification
model = LGBMClassifier(random_state=0)
params = {
    "boosting_type": ["gbdt", "dart", "goss"],
    "learning_rate": [0.1, 0.05, 0.01],
    "n_estimators": [10, 50, 100, 300]
}
clf = GridSearchCV(model, params, cv=10)
clf.fit(X, y)

# Outputing
# cv_results = cross_validate(model, X, y, cv=CV_NUM, scoring=('accuracy'))
# print(sum(cv_results["test_score"]) / CV_NUM)

y_hat = clf.predict(X_test)
output = pd.DataFrame(
    {'PassengerId': test_df["PassengerId"], "Survived": y_hat})
output.to_csv(OUTPUT_PATH, index=False)
