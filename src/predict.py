import pandas as pd

model = pd.read_pickle("../models/model.pkl")
df_predict = pd.read_csv("../data/bank_predict.csv")

y_pred = model.predict(df_predict)

print(len(df_predict))
print(len(y_pred))

# df_predict["y_pred"] = y_pred
# df_predict = df_predict['y_pred'].map({ 1: 'yes', 2: 'no'})

# df_predict.to_csv("../data/bank_predict.csv", index=False)

# df_predict = pd.read_csv("../data/bank_predict.csv")

# print(df_predict["y_pred"])