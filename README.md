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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv("/content/weather-station-eee-block_2024_07_13 (1).csv")
df.columns = df.columns.str.strip()

df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time')

df = df.interpolate()

df['hour'] = df['time'].dt.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

targets = ['tem', 'pm2_5', 'tsr']
for t in targets:
    df[f'{t}_lag1'] = df[t].shift(1)

df = df.dropna()

features = ['hum', 'pressure', 'wind_speed', 'illumination', 'co2',
            'hour_sin', 'hour_cos',
            'tem_lag1', 'pm2_5_lag1', 'tsr_lag1']

split = int(len(df) * 0.8)
train = df.iloc[:split]
test = df.iloc[split:]

X_train = train[features]
X_test = test[features]

models = {}
results = {}

target_meta = {
    'tem': ('Temperature', '°C', 'red'),
    'pm2_5': ('PM2.5', 'µg/m³', 'green'),
    'tsr': ('Solar Radiation', 'W/m²', 'orange')
}

for target in targets:
    y_train = train[target]
    y_test = test[target]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    models[target] = model
    results[target] = {
        'actual': y_test.values,
        'preds': preds,
        'r2': r2_score(y_test, preds),
        'mae': mean_absolute_error(y_test, preds)
    }

fig, axes = plt.subplots(3, 2, figsize=(16, 15))

for i, target in enumerate(targets):
    label, unit, color = target_meta[target]
    res = results[target]

    axes[i, 0].plot(res['actual'], label='Actual', color='black')
    axes[i, 0].plot(res['preds'], label='Predicted', linestyle='--', color=color)

    axes[i, 0].set_title(f"{label} (R2: {res['r2']:.2f}, MAE: {res['mae']:.2f})")
    axes[i, 0].set_ylabel(unit)
    axes[i, 0].legend()
    axes[i, 0].grid(True)

    importance = pd.Series(models[target].feature_importances_, index=features)
    importance = importance.sort_values()

    importance.plot(kind='barh', ax=axes[i, 1], color=color)
    axes[i, 1].set_title(f"Key Drivers: {label}")

plt.tight_layout()
plt.show()
```

## Output:
<img width="1760" height="540" alt="image" src="https://github.com/user-attachments/assets/77f15d65-b9a5-4e51-bad5-9255a9a92c1c" />

<img width="1744" height="495" alt="image" src="https://github.com/user-attachments/assets/db5c959b-5c87-4c57-9c78-c8836920a0ee" />

<img width="1749" height="501" alt="image" src="https://github.com/user-attachments/assets/7c590f33-4248-4e88-a7ee-99ef8eac7b2f" />


## Result:
Thus,Implementation of Random Forest Algorithm for Weather Prediction is successfully verified.

