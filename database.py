import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

DATABASE_URL = (
    f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Employee(Base):
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    role = Column(String(100))
    department = Column(String(100))
    email = Column(String(150))
    phone = Column(String(30))


class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True)
    name = Column(String(150), nullable=False)
    client = Column(String(150))
    status = Column(String(50))
    current_phase = Column(String(100))
    tech_stack = Column(Text)
    progress_percent = Column(Integer)
    description = Column(Text)


class Department(Base):
    __tablename__ = "departments"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    head = Column(String(100))
    team_size = Column(Integer)
    description = Column(Text)


def init_db():
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()