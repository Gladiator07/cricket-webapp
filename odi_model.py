import pandas as pd
df = pd.read_csv('Data/odi.csv')
X = df.iloc[:, [7,8,9,12,13,14]].values # input features
y = df.iloc[:, 15].values # target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.linear_model import LogisticRegression
log = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
log.fit(X_train, y_train)


def predict(data):
    prediction = log.predict_proba(sc.transform(data))
    prediction = prediction[0][1]*100
    return prediction
