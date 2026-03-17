# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Collect and preprocess weather dataset (temperature, humidity, rainfall, etc.).
2. Split the dataset into training and testing sets.
3. Train multiple decision trees using random subsets of data and features.
4. Combine predictions from all trees using majority voting (classification) or averaging (regression).
5. Evaluate the model accuracy and use it to predict future weather conditions.

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: Jayasri L
RegisterNumber:  212224040136
*/
```

```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("/content/Mall_Customers.csv")

print(data.head())
print(data.info())
print(data.isnull().sum())

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++")
    kmeans.fit(data.iloc[:, 3:5])
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

km = KMeans(n_clusters=5, init='k-means++')
y_pred = km.fit_predict(data.iloc[:, 3:5])

data["cluster"] = y_pred

df0 = data[data["cluster"] == 0]
df1 = data[data["cluster"] == 1]
df2 = data[data["cluster"] == 2]
df3 = data[data["cluster"] == 3]
df4 = data[data["cluster"] == 4]

plt.scatter(df0["Annual Income (k$)"], df0["Spending Score (1-100)"], c="red", label="Cluster 0")
plt.scatter(df1["Annual Income (k$)"], df1["Spending Score (1-100)"], c="black", label="Cluster 1")
plt.scatter(df2["Annual Income (k$)"], df2["Spending Score (1-100)"], c="blue", label="Cluster 2")
plt.scatter(df3["Annual Income (k$)"], df3["Spending Score (1-100)"], c="green", label="Cluster 3")
plt.scatter(df4["Annual Income (k$)"], df4["Spending Score (1-100)"], c="magenta", label="Cluster 4")

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments")
plt.legend()

plt.show()

```

## Output:

<img width="1056" height="429" alt="image" src="https://github.com/user-attachments/assets/d5c4c45a-3a6f-4b16-a3da-3e953882cf7e" />

<img width="1074" height="471" alt="image" src="https://github.com/user-attachments/assets/21791f82-65ad-4e9b-9b3d-1d7a48b2100a" />

<img width="1084" height="669" alt="image" src="https://github.com/user-attachments/assets/5571c985-5501-4135-9607-eca6801ce622" />

<img width="1198" height="750" alt="image" src="https://github.com/user-attachments/assets/396b3a80-2c16-4e39-ab12-9ab522bb34e5" />

<img width="1406" height="724" alt="image" src="https://github.com/user-attachments/assets/4f42f656-144e-49eb-bbe3-96f6ade72478" />

<img width="1290" height="750" alt="Screenshot 2026-03-17 193108" src="https://github.com/user-attachments/assets/11882dc8-8506-49cc-b4f5-f767e719f593" />



## Result:
Thus,Implementation of Random Forest Algorithm for Weather Prediction is successfully verified.

