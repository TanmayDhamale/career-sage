from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import tensorflow as tf
import pickle

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and encoder
model = tf.keras.models.load_model("models/ann_model.h5")
with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

class UserInput(BaseModel):
    age: int
    gender: str
    education: str
    skills: str
    interests: str

@app.post("/predict")
async def predict_career(user: UserInput):
    gender = le.transform([user.gender])[0]
    education = le.transform([user.education])[0]
    skills_count = len(user.skills.split(", "))
    interests_count = len(user.interests.split(", "))
    
    input_data = pd.DataFrame([[user.age, gender, education, skills_count, interests_count]],
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

# Run: uvicorn main:app --reload