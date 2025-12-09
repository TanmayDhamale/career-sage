import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pickle

# Load data
df = pd.read_csv("data/career_data.csv")

# Normalize Education and Gender columns
df["Education"] = df["Education"].str.replace("’", "'")
df["Gender"] = df["Gender"].str.replace("’", "'")

# Print data distribution for debugging
print("Dataset Info:")
print(df.info())
print("\nUnique Job Titles:", df["Job_Title"].unique())
print("Job Title Distribution:\n", df["Job_Title"].value_counts())
print("\nUnique Genders:", df["Gender"].unique())
print("Unique Education Levels:", df["Education"].unique())

# Print sample rows for each Job_Title to inspect data quality
print("\nSample rows for each Job_Title:")
for job_title in df["Job_Title"].unique():
    print(f"\nJob Title: {job_title}")
    print(df[df["Job_Title"] == job_title].head(3))

# Define all possible values for Gender and Education (must match the frontend)
possible_genders = ["Male", "Female", "Other"]
possible_education_levels = ["High School", "Bachelor's", "Master's", "PhD"]

# Validate that all expected Gender and Education values are in the dataset
for gender in possible_genders:
    if gender not in df["Gender"].values:
        print(f"Adding dummy row for Gender: {gender}")
        dummy_row = pd.DataFrame({
            "Age": [25],
            "Gender": [gender],
            "Education": ["Bachelor's"],
            "Skills": ["Python, Communication"],
            "Interests": ["Technology"],
            "Job_Title": ["Software Engineer"],
            "Industry": ["Tech"],
            "Salary_Range": ["$80k–$120k"],
            "Growth_Rate": [0.07]
        })
        df = pd.concat([df, dummy_row], ignore_index=True)

for education in possible_education_levels:
    if education not in df["Education"].values:
        print(f"Adding dummy row for Education: {education}")
        dummy_row = pd.DataFrame({
            "Age": [25],
            "Gender": ["Male"],
            "Education": [education],
            "Skills": ["Python, Communication"],
            "Interests": ["Technology"],
            "Job_Title": ["Software Engineer"],
            "Industry": ["Tech"],
            "Salary_Range": ["$80k–$120k"],
            "Growth_Rate": [0.07]
        })
        df = pd.concat([df, dummy_row], ignore_index=True)

# Feature engineering: Extract unique skills and interests
all_skills = set()
all_interests = set()
for skills in df["Skills"]:
    for skill in skills.split(", "):
        all_skills.add(skill)
for interests in df["Interests"]:
    for interest in interests.split(", "):
        all_interests.add(interest)

# Select top skills and interests to encode
top_skills = ["Python", "Java", "Design", "Communication", "Data Analysis", "Management", "Leadership", "Creativity"]
top_interests = ["Technology", "Art", "Healthcare", "Finance", "Education", "Science", "Design"]

# One-hot encode skills
for skill in top_skills:
    df[f"Skill_{skill}"] = df["Skills"].apply(lambda x: 1 if skill in x.split(", ") else 0)

# One-hot encode interests
for interest in top_interests:
    df[f"Interest_{interest}"] = df["Interests"].apply(lambda x: 1 if interest in x.split(", ") else 0)

# Preprocessing
le_gender = LabelEncoder()
le_gender.fit(possible_genders)
df["Gender"] = le_gender.transform(df["Gender"])

le_education = LabelEncoder()
le_education.fit(possible_education_levels)
df["Education"] = le_education.transform(df["Education"])

le_job = LabelEncoder()
df["Job_Title"] = le_job.fit_transform(df["Job_Title"])  # Target

# Feature engineering: Skills and Interests counts
df["Skills_Count"] = df["Skills"].apply(lambda x: len(x.split(", ")))
df["Interests_Count"] = df["Interests"].apply(lambda x: len(x.split(", ")))

# Features and target
feature_columns = ["Age", "Gender", "Education", "Skills_Count", "Interests_Count"] + \
                  [f"Skill_{skill}" for skill in top_skills] + \
                  [f"Interest_{interest}" for interest in top_interests]
X = df[feature_columns]
y = df["Job_Title"]

# Normalize numerical features
scaler = MinMaxScaler()
numerical_features = ["Age", "Skills_Count", "Interests_Count"]
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compute class weights to handle imbalance
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print("\nClass Weights:", class_weight_dict)

# Train Random Forest model
print("\nTraining Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
rf_model.fit(X_train, y_train)

# Evaluate Random Forest
rf_train_acc = rf_model.score(X_train, y_train)
rf_val_acc = rf_model.score(X_test, y_test)
print(f"Random Forest Training Accuracy: {rf_train_acc}")
print(f"Random Forest Validation Accuracy: {rf_val_acc}")

# Test Random Forest on sample inputs
sample_inputs = [
    # Sample 1: Should predict Software Engineer or Data Scientist
    {
        "Age": 25,
        "Gender": "Male",
        "Education": "Bachelor's",
        "Skills": "Python, Communication",
        "Interests": "Technology"
    },
    # Sample 2: Should predict Data Scientist
    {
        "Age": 30,
        "Gender": "Female",
        "Education": "Master's",
        "Skills": "Java, Leadership",
        "Interests": "Art"
    },
    # Sample 3: Should predict Graphic Designer
    {
        "Age": 22,
        "Gender": "Other",
        "Education": "High School",
        "Skills": "Design, Creativity",
        "Interests": "Design"
    }
]

print("\nRandom Forest Predictions:")
for sample in sample_inputs:
    sample_gender = le_gender.transform([sample["Gender"]])[0]
    sample_education = le_education.transform([sample["Education"]])[0]
    sample_skills_count = len(sample["Skills"].split(", "))
    sample_interests_count = len(sample["Interests"].split(", "))
    sample_skills_list = sample["Skills"].split(", ")
    sample_interests_list = sample["Interests"].split(", ")

    sample_data_dict = {
        "Age": sample["Age"],
        "Gender": sample_gender,
        "Education": sample_education,
        "Skills_Count": sample_skills_count,
        "Interests_Count": sample_interests_count
    }

    for skill in top_skills:
        sample_data_dict[f"Skill_{skill}"] = 1 if skill in sample_skills_list else 0
    for interest in top_interests:
        sample_data_dict[f"Interest_{interest}"] = 1 if interest in sample_interests_list else 0

    sample_input = pd.DataFrame([sample_data_dict], columns=feature_columns)
    sample_input[numerical_features] = scaler.transform(sample_input[numerical_features])
    
    pred = rf_model.predict_proba(sample_input)
    job_idx = pred.argmax()
    job_title = le_job.inverse_transform([job_idx])[0]
    print(f"\nSample Input: {sample}")
    print(f"Prediction Probabilities: {pred.tolist()}")
    print(f"Predicted Job Title: {job_title}")

# Save the Random Forest model
with open("models/rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

# Save the encoders and scaler
with open("models/label_encoder_gender.pkl", "wb") as f:
    pickle.dump(le_gender, f)

with open("models/label_encoder_education.pkl", "wb") as f:
    pickle.dump(le_education, f)

with open("models/label_encoder_job.pkl", "wb") as f:
    pickle.dump(le_job, f)

with open("models/feature_columns.pkl", "wb") as f:
    pickle.dump(feature_columns, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nModel trained and saved.")
print("Gender Encoder classes:", le_gender.classes_)
print("Education Encoder classes:", le_education.classes_)
print("Job Title Encoder classes:", le_job.classes_)
print("Feature columns:", feature_columns)