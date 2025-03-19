from database import Base, engine
from models import User, Profile

# Create all tables
Base.metadata.create_all(bind=engine)

print("Database and tables created successfully!")