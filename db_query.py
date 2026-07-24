from database import SessionLocal, Employee, Project, Department
from sqlalchemy import or_

def get_all_employees():
    db = SessionLocal()
    employees = db.query(Employee).all()
    db.close()
    return employees

def search_employee(name: str):
    db = SessionLocal()
    results = db.query(Employee).filter(
        or_(
            Employee.name.ilike(f"%{name}%"),
            Employee.role.ilike(f"%{name}%"),
            Employee.department.ilike(f"%{name}%")
        )
    ).all()
    db.close()
    return results

def get_all_projects():
    db = SessionLocal()
    projects = db.query(Project).all()
    db.close()
    return projects

def get_project_status(project_name: str):
    db = SessionLocal()
    project = db.query(Project).filter(
        Project.name.ilike(f"%{project_name}%")
    ).first()
    db.close()
    return project

def get_department_info(dept_name: str):
    db = SessionLocal()
    dept = db.query(Department).filter(
        Department.name.ilike(f"%{dept_name}%")
    ).first()
    db.close()
    return dept

def get_all_departments():
    db = SessionLocal()
    departments = db.query(Department).all()
    db.close()
    return departments

def format_employee_context(employees):
    if not employees:
        return ""
    lines = []
    for e in employees:
        lines.append(f"{e.name} — {e.role}, {e.department} Department. Email: {e.email}, Phone: {e.phone}")
    return "\n".join(lines)

def format_project_context(projects):
    if not projects:
        return ""
    lines = []
    for p in projects:
        lines.append(
            f"Project: {p.name} | Client: {p.client} | Status: {p.status} | "
            f"Phase: {p.current_phase} | Progress: {p.progress_percent}% | "
            f"Tech Stack: {p.tech_stack}"
        )
    return "\n".join(lines)

def format_department_context(departments):
    if not departments:
        return ""
    lines = []
    for d in departments:
        lines.append(f"{d.name} Department — Head: {d.head}, Team Size: {d.team_size}. {d.description}")
    return "\n".join(lines)