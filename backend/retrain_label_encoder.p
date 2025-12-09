from sklearn.preprocessing import LabelEncoder
import pickle

# Define all possible values for Gender and Education (must match the frontend)
genders = ["Male", "Female", "Other"]
education_levels = ["High School", "Bachelor's", "Master's", "PhD"]

# Combine all categories into one list (since we use a single LabelEncoder for both)
all_categories = genders + education_levels

# Train the LabelEncoder
le = LabelEncoder()
le.fit(all_categories)

# Save the updated LabelEncoder
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("LabelEncoder retrained and saved successfully!")
print("Classes:", le.classes_)