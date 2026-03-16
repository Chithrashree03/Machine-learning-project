import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Load the dataset (Make sure your CSV is in the same folder)
df = pd.read_csv(r'C:\Users\chith\OneDrive\Desktop\EV\station_data_dataverse.csv')

# 2. Preprocessing
# Fill missing 'distance' values with median
df['distance'] = df['distance'].fillna(df['distance'].median())

# Encode categorical features
le_weekday = LabelEncoder()
df['weekday_encoded'] = le_weekday.fit_transform(df['weekday'])

le_platform = LabelEncoder()
df['platform_encoded'] = le_platform.fit_transform(df['platform'])

# Select Features (X) and Target (y)
features = ['startTime', 'endTime', 'chargeTimeHrs', 'weekday_encoded', 
            'platform_encoded', 'distance', 'stationId', 'facilityType']
target = 'kwhTotal'

X = df[features]
y = df[target]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Building
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Save Model and Encoders
joblib.dump(model, 'ev_model.pkl')
joblib.dump(le_weekday, 'le_weekday.pkl')
joblib.dump(le_platform, 'le_platform.pkl')

print("Model trained and saved successfully!")