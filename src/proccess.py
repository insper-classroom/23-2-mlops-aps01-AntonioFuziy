import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

dep_mapping = {"yes": 1, "no": 0}

df = pd.read_csv("../data/bank.csv")

df["deposit"] = df["deposit"].astype("category").map(dep_mapping)

df = df.drop(labels = ["default", "contact", "day", "month", "pdays", "previous", "loan", "poutcome", "poutcome"], axis=1)

pd.DataFrame(df.isnull().sum()).T

df.to_csv("../data/bank_proccess.csv", index=False)