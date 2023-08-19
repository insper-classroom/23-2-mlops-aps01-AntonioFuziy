import pickle
from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("../data/bank_proccess.csv")

cat_cols = ["job", "marital", "education", "housing"]

X = df.drop("deposit", axis=1)
y = df["deposit"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1912)

one_hot_enc = make_column_transformer(
  (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols), 
  remainder="passthrough"
)

X_train = one_hot_enc.fit_transform(X_train)
X_train = pd.DataFrame(X_train, columns=one_hot_enc.get_feature_names_out())

X_test = pd.DataFrame(one_hot_enc.transform(X_test), columns=one_hot_enc.get_feature_names_out())

model = LGBMClassifier()
model.fit(X_train, y_train)

X_train.to_csv("../data/x_train.csv", index=False)
X_test.to_csv("../data/x_test.csv", index=False)
y_train.to_csv("../data/y_train.csv", index=False)
y_test.to_csv("../data/y_test.csv", index=False)

model_file_path = "../models/model.pkl"

with open(model_file_path, "wb") as f:
  pickle.dump(model, f)

ohe_file_path = "../models/ohe.pkl"

with open(ohe_file_path, "wb") as f:
  pickle.dump(one_hot_enc, f)