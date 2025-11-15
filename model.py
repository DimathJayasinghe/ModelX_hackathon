import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('/content/sample_data/Dementia Prediction Dataset.csv')

non_medical_variables = [
    "MARISTAT", "NACCLIVS", "INDEPEND", "RESIDENC", "HANDED",
    "NACCAGE", "NACCAGEB", "BIRTHMO", "BIRTHYR", "SEX",
    "HISPANIC", "HISPOR", "HISPORX", "RACE", "RACEX",
    "RACESEC", "RACESECX", "RACETER", "RACETERX",
    "PRIMLANG", "PRIMLANX", "EDUC", "NACCREAS",
    "NACCREFR", "NACCNIHR",

    "INBIRMO", "INBIRYR", "INSEX", "NEWINF", "INHISP",
    "INHISPOR", "INHISPOX", "INRACE", "INRACEX",
    "INRASEC", "INRASECX", "INRATER", "INRATERX",
    "INEDUC", "INRELTO", "INRELTOX", "INKNOWN",
    "INLIVWTH", "INVISITS", "INCALLS", "INRELY",
    "NACCNINR",

    "NACCFAM", "NACCMOM", "NACCDAD", "NACCFADM",
    "NACCAM", "NACCAMS"
]
df_new=df[non_medical_variables]

df_new = df_new.apply(pd.to_numeric, errors='coerce')
cols_with_missing = df_new.columns[df_new.isnull().sum() != 0].tolist()
df_new=df_new.drop(cols_with_missing,axis=1)
from sklearn.model_selection import train_test_split

x = df_new
y = df['DEMENTED']

X_train, X_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Create model
model = LogisticRegression()

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

confusion_matrix(y_test, y_pred)


from xgboost import XGBClassifier
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


confusion_matrix(y_test, y_pred)

