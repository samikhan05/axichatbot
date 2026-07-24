from database import SessionLocal, init_db, Employee, Project, Department

def seed():
    init_db()
    db = SessionLocal()

    # Clear existing data
    db.query(Employee).delete()
    db.query(Project).delete()
    db.query(Department).delete()

    # Departments
    departments = [
        Department(name="Artificial Intelligence", head="Dr. Ahmed Raza", team_size=8, description="Develops AI systems, RAG applications, and computer vision models."),
        Department(name="Software Engineering", head="Sara Khan", team_size=12, description="Builds enterprise software and backend systems."),
        Department(name="DevOps", head="Usman Ali", team_size=5, description="Manages infrastructure, CI/CD pipelines, and deployments."),
        Department(name="Cloud Infrastructure", head="Bilal Hassan", team_size=4, description="Manages cloud resources and architecture."),
        Department(name="Human Resources", head="Fatima Malik", team_size=3, description="Handles recruitment, policies, and employee welfare."),
    ]
    db.add_all(departments)

    # Employees
    employees = [
        Employee(name="Dr. Ahmed Raza", role="AI Department Lead", department="Artificial Intelligence", email="ahmed.raza@axitech.com", phone="+92-300-1234567"),
        Employee(name="Sara Khan", role="Engineering Lead", department="Software Engineering", email="sara.khan@axitech.com", phone="+92-300-2345678"),
        Employee(name="Usman Ali", role="DevOps Lead", department="DevOps", email="usman.ali@axitech.com", phone="+92-300-3456789"),
        Employee(name="Zain Ahmed", role="Backend Engineer", department="Artificial Intelligence", email="zain.ahmed@axitech.com", phone="+92-300-4567890"),
        Employee(name="Ayesha Siddiqui", role="Frontend Developer", department="Software Engineering", email="ayesha.s@axitech.com", phone="+92-300-5678901"),
        Employee(name="Hassan Tariq", role="ML Engineer", department="Artificial Intelligence", email="hassan.t@axitech.com", phone="+92-300-6789012"),
    ]
    db.add_all(employees)

    # Projects
    projects = [
        Project(
            name="AI Video Reception Chatbot",
            client="Internal",
            status="In Progress",
            current_phase="Phase 2 - Speech and Avatar",
            tech_stack="FastAPI, React, Groq, BGE-M3, Qdrant, Three.js, Rhubarb",
            progress_percent=65,
            description="An AI-powered video receptionist that interacts with visitors using speech and a 3D avatar."
        ),
        Project(
            name="Inventory Management Platform",
            client="RetailCo",
            status="In Progress",
            current_phase="Phase 1 - Core Backend",
            tech_stack="Django, PostgreSQL, React, Redis",
            progress_percent=40,
            description="A real-time inventory tracking and management system for retail operations."
        ),
        Project(
            name="Patient Scheduling System",
            client="MediHealth",
            status="In Progress",
            current_phase="Phase 2 - Appointment Engine",
            tech_stack="FastAPI, PostgreSQL, React, Celery",
            progress_percent=55,
            description="An automated patient scheduling and appointment management system for healthcare."
        ),
    ]
    db.add_all(projects)

    db.commit()
    db.close()
    print("Sample data seeded successfully.")

if __name__ == "__main__":
    seed()