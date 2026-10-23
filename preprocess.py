import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder

from Task1.models import NN

df=pd.read_csv(r"Task1\data\penguins.csv")
df.info()
df.duplicated().sum()   

X = df.drop(columns=["Species"])
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

df.describe()

X_train = pd.get_dummies(X_train, columns=["OriginLocation"],dtype=int)
X_test = pd.get_dummies(X_test, columns=["OriginLocation"],dtype=int)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)


num_cols = ["CulmenLength", "CulmenDepth",
            "FlipperLength", "BodyMass"]
X_train.isnull().sum()
X_test.isnull().sum()
med=X_train[num_cols].median()

X_train[num_cols] = X_train[num_cols].fillna(med)
X_test[num_cols] = X_test[num_cols].fillna(med)

X_train.isnull().sum()
X_test.isnull().sum()

scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

X_train.head()
X_test.head()



#############################################################################
##training and testing the model
#############################################################################

p = NN()
p.perc(X_train.values, y_train)

a = NN()
a.adaline(X_train.values, y_train)

p_pred = p.predict(X_test.values)
a_pred = a.predict(X_test.values)


print("Perceptron Accuracy:", (p_pred == y_test).mean())
print("Adaline Accuracy:", (a_pred == y_test).mean())

