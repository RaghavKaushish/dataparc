import pandas as pd
import numpy as np
import joblib
import os
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# 1. Ensure folder exists
if not os.path.exists('model_data'):
    os.makedirs('model_data')

# 2. Load and Clean Data
# Reading header 2 for descriptions and header 8 for tags
df_desc = pd.read_excel('SampleData_Training.xlsx', header=2)
df_tags = pd.read_excel('SampleData_Training.xlsx', header=8)

# Create a mapping from Tag ID to readable Name
tag_to_name = {str(df_tags.columns[i]): str(df_desc.columns[i]).strip() for i in range(1, len(df_tags.columns))}

# Process data
df = df_tags.drop(index=0).apply(pd.to_numeric, errors='coerce')
df = df.fillna(df.mean()).replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all')

# 3. Define Targets and Features
target_names = [
    'Main Steam Flow Tx.',
    'Flue gas furnace pressure',
    'Main Steam Temperature after valve',
    'Main Steam before valve pressure',
    'Flue gas center-left O2'
]

# Convert names back to tags for training
name_to_tag = {v: k for k, v in tag_to_name.items()}
target_tags = [name_to_tag[name] for name in target_names if name in name_to_tag]

features = [c for c in df.columns if c not in ['Unnamed: 0'] + target_tags]

X = df[features]
y = df[target_tags]

# 4. Scale and Train
scaler = StandardScaler().fit(X)
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=50)).fit(scaler.transform(X), y)

# 5. Save with compression to keep files small for GitHub
joblib.dump(model, 'model_data/model.pkl', compress=3)
joblib.dump(scaler, 'model_data/scaler.pkl', compress=3)
joblib.dump(features, 'model_data/features.pkl')
joblib.dump(target_names, 'model_data/target_names.pkl')
joblib.dump(tag_to_name, 'model_data/tag_to_name.pkl')

print("Training complete! Files saved in 'model_data/' folder.")