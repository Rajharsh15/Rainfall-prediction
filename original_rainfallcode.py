import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV , cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report , confusion_matrix ,accuracy_score
import pickle

data = pd.read_csv(r"C:\Users\Acer\Pictures\Screenshots\coding\python\project\Rainfall.csv")
print("data info")
data.info()
print(data.columns)
data.columns = data.columns.str.strip()# these lines are used to align the and remove space

print(data.columns)
data = data.drop(columns=["day"])
print(data)
print(data.isnull().sum())

print(data["winddirection"].unique())
print(data["windspeed"].unique())

#handling the missing values
data["winddirection"]= data["winddirection"].fillna(data["winddirection"].mode()[0])
data["windspeed"] = data["windspeed"].fillna(data["windspeed"].median())
print(data.isnull().sum())
print(data["rainfall"].unique())
data["rainfall"]=data["rainfall"].map({"yes":1,"no":0})
print(data["rainfall"].unique())

#setting plt style for all point
sns.set(style="whitegrid")
print(data.describe())
plt.figure(figsize=(15,10))
for i , column in enumerate(['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint','humidity', 'cloud', 'sunshine', 'windspeed'],1):
    plt.subplot(3,3,i)
    sns.histplot(data[column],kde=True)
    plt.title(f"distribution of {column}")

plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="rainfall" , data= data)
plt.title("distribution of rainfall")
plt.show()

#correlation matrix
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot = True, cmap = "coolwarm",fmt=".2f")
plt.title("correlation heatmap")
plt.show()

#drop highly correlated column
data = data.drop(columns=['maxtemp','temparature','mintemp'])

print(data["rainfall"].value_counts())

#separate the majority and minority class
df_majority = data[data["rainfall"]==1]
df_minority = data[data["rainfall"]==0]
print(df_majority.shape)
print(df_minority.shape)

df_majority_downsampled = resample(df_majority , replace=False, n_samples = len(df_minority), random_state = 42)
print(df_majority_downsampled.shape)

df_downsampled = pd.concat([df_majority_downsampled, df_minority])
print(df_downsampled.shape)

#shuffle the final dataframe
df_downsampled = df_downsampled.sample(frac = 1 , random_state=42).reset_index(drop=True)
print(df_downsampled.head())
print(df_downsampled["rainfall"].value_counts())

#split feature and target x and y
x= df_downsampled.drop(columns=["rainfall"])
y = df_downsampled["rainfall"]
print(x)
print(y)

#splitting
x_train , x_test , y_train , y_test = train_test_split( x, y, test_size=0.2,random_state=42)

#model training
rf_model = RandomForestClassifier(random_state=42)
param_grid_rf = {
    "n_estimators": [50, 100,],
    "max_features": ["sqrt", "log2"],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}
# Hypertuning using GridSearchCV
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=1)
grid_search_rf.fit(x_train, y_train)

best_rf_model = grid_search_rf.best_estimator_
print("best parameters for Random Forest:", grid_search_rf.best_params_)

cv_scores = cross_val_score(best_rf_model, x_train, y_train, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", np.mean(cv_scores))

# test set performance
y_pred = best_rf_model.predict(x_test)

print("Test set Accuracy:", accuracy_score(y_test, y_pred))
print("Test set Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

input_data = (1015.9, 19.9, 95, 81, 0.0, 40.0, 13.7)

input_df = pd.DataFrame([input_data], columns=['pressure', 'dewpoint', 'humidity', 'cloud', 'sunshine','winddirection', 'windspeed'])
prediction = best_rf_model.predict(input_df)
print(prediction)

prediction = best_rf_model.predict(input_df)
print("Prediction result:", "Rainfall" if prediction[0] == 1 else "No Rainfall")
