import pandas as pd
from lightgbm import LGBMClassifier

# constants
DEBUG = 0
TRAIN_DATA_PATH = "titanic\data\\train.csv"
TEST_DATA_PATH = "titanic\data\\test.csv"
OUTPUT_PATH = "titanic\data\\submission.csv"
Y_LABEL = "Survived"

# load data
train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = pd.get_dummies(train_df[features])
X_test = pd.get_dummies(test_df[features])
y = train_df[Y_LABEL]
if DEBUG:
    print(X.shape)
    print(y)
    women = train_df.loc[train_df.Sex == 'female']["Survived"]
    print(women)

# data cleaning

# classification
model = LGBMClassifier()
model.fit(X, y)
y_hat = model.predict(X_test)
output = pd.DataFrame(
    {'PassengerId': test_df["PassengerId"], "Survived": y_hat})
output.to_csv(OUTPUT_PATH, index=False)
