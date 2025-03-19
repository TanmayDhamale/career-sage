from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
import pandas as pd
import tensorflow as tf
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

# Load model and encoder
model = tf.keras.models.load_model("models/ann_model.h5")
with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

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
    # Check if user already exists
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user
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

# Predict career (same as before)
@app.post("/predict")
async def predict_career(profile: UserProfile, db: Session = Depends(get_db)):
    gender = le.transform([profile.gender])[0]
    education = le.transform([profile.education])[0]
    skills_count = len(profile.skills.split(", "))
    interests_count = len(profile.interests.split(", "))
    
    input_data = pd.DataFrame([[profile.age, gender, education, skills_count, interests_count]],
                              columns=["Age", "Gender", "Education", "Skills_Count", "Interests_Count"])
    
    pred = model.predict(input_data)
    job_idx = pred.argmax()
    job_title = le.inverse_transform([job_idx])[0]
    
    result = {
        "job_title": job_title,
        "description": f"Description for {job_title}",
        "industry": "Tech",
        "salary_range": "$80k–$120k",
        "growth_rate": 0.07
    }
    return result