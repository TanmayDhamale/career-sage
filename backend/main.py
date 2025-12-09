from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
import pandas as pd
import pickle
from database import get_db
from models import User, Profile

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and encoders
with open("models/rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/label_encoder_gender.pkl", "rb") as f:
    le_gender = pickle.load(f)

with open("models/label_encoder_education.pkl", "rb") as f:
    le_education = pickle.load(f)

with open("models/label_encoder_job.pkl", "rb") as f:
    le_job = pickle.load(f)

# Load feature columns
with open("models/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# Load scaler
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load career_data.csv to map Job_Title to Industry, Salary_Range, and Growth_Rate
career_data = pd.read_csv("data/career_data.csv")
career_data["Education"] = career_data["Education"].str.replace("’", "'")
career_data["Gender"] = career_data["Gender"].str.replace("’", "'")
job_info = career_data.groupby("Job_Title")[["Industry", "Salary_Range", "Growth_Rate"]].first().to_dict()

# Define top skills and interests (must match train.py)
top_skills = ["Python", "Java", "Design", "Communication", "Data Analysis", "Management", "Leadership", "Creativity"]
top_interests = ["Technology", "Art", "Healthcare", "Finance", "Education", "Science", "Design"]

# Pydantic models for request validation
class UserCreate(BaseModel):
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class UserProfile(BaseModel):
    age: int
    gender: str
    education: str
    skills: str
    interests: str

# Register a new user
@app.post("/register")
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    new_user = User(email=user.email, password=user.password)  # In production, hash the password
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User registered successfully"}

# Login a user
@app.post("/login")
async def login_user(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user or db_user.password != user.password:  # In production, compare hashed passwords
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return {"message": "Login successful", "user_id": db_user.id}

# Save user profile
@app.post("/profile")
async def save_profile(profile: UserProfile, user_id: int, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    new_profile = Profile(
        user_id=user_id,
        age=profile.age,
        gender=profile.gender,
        education=profile.education,
        skills=profile.skills,
        interests=profile.interests
    )
    db.add(new_profile)
    db.commit()
    db.refresh(new_profile)
    return {"message": "Profile saved successfully"}

# Predict career
@app.post("/predict")
async def predict_career(profile: UserProfile, db: Session = Depends(get_db)):
    try:
        print("Received profile:", profile.dict())

        # Normalize apostrophes in the input
        profile.gender = profile.gender.replace("’", "'")
        profile.education = profile.education.replace("’", "'")

        # Transform gender and education
        gender = le_gender.transform([profile.gender])[0]
        print("Transformed gender:", gender)

        education = le_education.transform([profile.education])[0]
        print("Transformed education:", education)

        # Calculate skills and interests counts
        skills_count = len(profile.skills.split(", "))
        print("Skills count:", skills_count)

        interests_count = len(profile.interests.split(", "))
        print("Interests count:", interests_count)

        # One-hot encode skills and interests
        skills_list = profile.skills.split(", ")
        interests_list = profile.interests.split(", ")

        input_data_dict = {
            "Age": profile.age,
            "Gender": gender,
            "Education": education,
            "Skills_Count": skills_count,
            "Interests_Count": interests_count
        }

        # Add one-hot encoded skills
        for skill in top_skills:
            input_data_dict[f"Skill_{skill}"] = 1 if skill in skills_list else 0

        # Add one-hot encoded interests
        for interest in top_interests:
            input_data_dict[f"Interest_{interest}"] = 1 if interest in interests_list else 0

        # Create input DataFrame
        input_data = pd.DataFrame([input_data_dict], columns=feature_columns)
        print("Input data (before scaling):", input_data.to_dict())

        # Normalize numerical features
        numerical_features = ["Age", "Skills_Count", "Interests_Count"]
        input_data[numerical_features] = scaler.transform(input_data[numerical_features])
        print("Input data (after scaling):", input_data.to_dict())

        # Verify input shape
        print("Input shape:", input_data.shape)  # Should be (1, 20)

        # Make prediction using Random Forest
        pred = model.predict_proba(input_data)
        print("Prediction probabilities:", pred.tolist())

        job_idx = pred.argmax()
        print("Predicted job index:", job_idx)

        job_title = le_job.inverse_transform([job_idx])[0]
        print("Predicted job title:", job_title)

        # Get additional info from career_data.csv
        industry = job_info["Industry"].get(job_title, "Unknown")
        salary_range = job_info["Salary_Range"].get(job_title, "Unknown")
        growth_rate = job_info["Growth_Rate"].get(job_title, 0.0)

        result = {
            "job_title": job_title,
            "description": f"A professional role as a {job_title}.",
            "industry": industry,
            "salary_range": salary_range,
            "growth_rate": float(growth_rate)
        }
        return result
    except Exception as e:
        print(f"Error in /predict: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")