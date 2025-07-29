import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split , GridSearchCV , cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_and_clean_data():
    data = pd.read_csv(r"C:\Users\Acer\Pictures\Screenshots\coding\Rainfall prediction\Rainfall.csv")
    data.columns = data.columns.str.strip()
    data = data.drop(columns=["day"])
    column_renames = {col: col.strip() for col in data.columns}
    data.rename(columns=column_renames, inplace=True)
    data["winddirection"] = data["winddirection"].fillna(data["winddirection"].mode()[0])
    data["windspeed"] = data["windspeed"].fillna(data["windspeed"].median())
    data["rainfall"] =data["rainfall"].map({"yes":1,"no":0})
    data = data.drop(columns=['maxtemp','temparature','mintemp'])
    return data

def balance_data(data):
    df_majority = data[data["rainfall"]==1]
    df_minority = data[data["rainfall"]==0]
    df_majority_downsampled = resample(df_majority , replace =False, n_samples= len(df_majority), random_state=42)
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
    df_downsampled = df_downsampled.sample(frac=1, random_state=42).reset_index(drop=True)
    return df_downsampled

def train_rain_model(data):
    x = data.drop(columns=["rainfall"])
    y = data["rainfall"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(random_state=42)
    param_grid_rf = {
        "n_estimators": [100],
        "max_features": ["sqrt"],
        "max_depth": [20],
        "min_samples_split": [2],
        "min_samples_leaf": [1]
    }
    grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, n_jobs=-1)
    grid_search_rf.fit(x_train, y_train)
    best_model = grid_search_rf.best_estimator_

    train_acc = accuracy_score(y_train, best_model.predict(x_train))
    test_acc = accuracy_score(y_test, best_model.predict(x_test))

    return best_model, x.columns.tolist(), train_acc, test_acc

def predict_rain(model, input_data):
    arr = np.array(input_data).reshape(1, -1)
    prediction = model.predict(arr)[0]
    return "🌧️ Rainfall expected" if prediction == 1 else "☀️ No Rainfall"