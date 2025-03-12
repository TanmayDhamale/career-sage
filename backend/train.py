import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle

# Load data
df = pd.read_csv("data/career_data.csv")

# Preprocessing
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
df["Education"] = le.fit_transform(df["Education"])
df["Job_Title"] = le.fit_transform(df["Job_Title"])  # Target

# Simple feature engineering (skills/interests as counts for now)
df["Skills_Count"] = df["Skills"].apply(lambda x: len(x.split(", ")))
df["Interests_Count"] = df["Interests"].apply(lambda x: len(x.split(", ")))

# Features and target
X = df[["Age", "Gender", "Education", "Skills_Count", "Interests_Count"]]
y = df["Job_Title"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build ANN
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(32, activation="relu"),
    Dense(len(df["Job_Title"].unique()), activation="softmax")  # Output layer
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save model and encoder
model.save("models/ann_model.h5")
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Model trained and saved.")