import pandas as pd

model = pd.read_pickle("../models/model.pkl")
one_hot_enc = pd.read_pickle("../models/ohe.pkl")
df_predict = pd.read_csv("../data/bank_predict.csv")

X_test = pd.DataFrame(one_hot_enc.transform(df_predict), columns=one_hot_enc.get_feature_names_out())

y_pred = model.predict(X_test)

df_predict["y_pred"] = y_pred
df_predict = df_predict['y_pred'].map({ 1: 'yes', 0: 'no'})
df_predict.to_csv("../data/bank_result.csv", index=False)