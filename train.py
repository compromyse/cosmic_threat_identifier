import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import AdaBoostClassifier

from joblib import dump

# Load data
df = pd.read_csv('orbits.csv')

# Drop empty rows
df = df.dropna()

# Drop useless attributes
df = df.drop('Object Name', axis=1)

# Define X and y
X = df.drop('Hazardous', axis=1)
y = df['Hazardous']
X = pd.get_dummies(X, drop_first=True)
y = pd.get_dummies(y, drop_first=True)

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define model as AdaBoostClassifier
model = AdaBoostClassifier()

# Train model
model.fit(X, y.values.ravel())

# Save the model
dump(model, 'out/model.bin')

# Save the scaler
dump(scaler, 'out/scaler.bin')
