import pandas as pd
from faker import Faker
import random

fake = Faker()
n_samples = 10000

# Define possible values
educations = ["High School", "Bachelor’s", "Master’s", "PhD"]
genders = ["Male", "Female", "Other"]
skills_pool = ["Python", "Java", "Communication", "Data Analysis", "Design", "Management"]
interests_pool = ["Technology", "Art", "Finance", "Healthcare", "Education"]
jobs = [
    {"title": "Software Engineer", "industry": "Tech", "salary": "$80k–$120k", "growth": 0.07},
    {"title": "Graphic Designer", "industry": "Creative", "salary": "$40k–$60k", "growth": 0.03},
    {"title": "Data Scientist", "industry": "Tech", "salary": "$90k–$140k", "growth": 0.08},
    # Add more jobs as needed
]

# Generate data
data = []
for _ in range(n_samples):
    age = random.randint(18, 65)
    gender = random.choice(genders)
    education = random.choice(educations)
    skills = ", ".join(random.sample(skills_pool, random.randint(2, 4)))
    interests = ", ".join(random.sample(interests_pool, random.randint(1, 3)))
    job = random.choice(jobs)
    
    data.append([age, gender, education, skills, interests, job["title"], job["industry"], job["salary"], job["growth"]])

# Create DataFrame
df = pd.DataFrame(data, columns=["Age", "Gender", "Education", "Skills", "Interests", "Job_Title", "Industry", "Salary_Range", "Growth_Rate"])
df.to_csv("career_data.csv", index=False)
print("Dataset generated: career_data.csv")