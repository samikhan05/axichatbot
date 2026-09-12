import os
import re
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from openai import OpenAI, APITimeoutError, APIConnectionError

from db_query import (
    get_all_employees, search_employee,
    get_all_projects, get_project_status,
    get_all_departments, get_department_info as db_get_department_info,
    format_employee_context, format_project_context, format_department_context
)

load_dotenv()

embed_model = SentenceTransformer("BAAI/bge-m3")
qdrant = QdrantClient(url="http://localhost:6333")

llm = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)

MODEL_NAME = "qwen/qwen3.6-27b"
# NOTE: verify this specific model supports Groq's tool-calling API before
# relying on this in production/demo — not all hosted models do. If it
# errors, switch to a documented tool-calling-capable model on Groq.

SYSTEM_PROMPT = (
    "You are a company receptionist. You have tools to look up company "
    "documents or live company data (employees, projects, departments). "
    "Use a tool whenever the question needs specific information you don't "
    "already have in this conversation. If the question is a greeting or "
    "general chit-chat, you don't need to use a tool. "
    "Answer using only the information you retrieve. Keep answers short — "
    "1 to 2 sentences maximum. Be direct and natural, like a real "
    "receptionist speaking out loud, not a written document. No extra "
    "explanations, no repeating the question, no unnecessary greetings "
    "unless the visitor greets first. Do not use markdown formatting, "
    "bullet points, numbered lists, or bold text — write plain, "
    "spoken-style sentences only, since your response will eventually be "
    "converted to speech. Never guess or state the current date, time, or "
    "any real-world fact not explicitly provided to you. If the answer "
    "isn't found by your tools, say so briefly and suggest asking about "
    "the company, its projects, or its team. When mentioning multiple "
    "items, state them as short separate sentences instead of one long "
    "comma-separated list, since long lists don't sound natural when "
    "spoken aloud."
)


# ---------------------------------------------------------------------------
# Tool implementations — the actual work each tool does when the LLM calls it
# ---------------------------------------------------------------------------

def _tool_search_documents(query: str) -> str:
    query_vector = embed_model.encode(query).tolist()
    results = qdrant.query_points(
        collection_name="company_knowledge",
        query=query_vector,
        limit=3,
    ).points
    if not results:
        return "No relevant documents found."
    return "\n".join(r.payload["text"] for r in results)


def _tool_get_employee_info(name: str = None) -> str:
    employees = search_employee(name) if name else get_all_employees()
    text = format_employee_context(employees)
    return text if text else "No matching employees found."


def _tool_get_project_info(project_name: str = None) -> str:
    if project_name:
        project = get_project_status(project_name)
        projects = [project] if project else []
    else:
        projects = get_all_projects()
    text = format_project_context(projects)
    return text if text else "No matching project found."


def _tool_get_department_info(department_name: str = None) -> str:
    if department_name:
        dept = db_get_department_info(department_name)
        departments = [dept] if dept else []
    else:
        departments = get_all_departments()
    text = format_department_context(departments)
    return text if text else "No matching department found."


TOOL_FUNCTIONS = {
    "search_documents": lambda args: _tool_search_documents(args.get("query", "")),
    "get_employee_info": lambda args: _tool_get_employee_info(args.get("name")),
    "get_project_info": lambda args: _tool_get_project_info(args.get("project_name")),
    "get_department_info": lambda args: _tool_get_department_info(args.get("department_name")),
}

# JSON schema describing each tool to the LLM — this is what lets the LLM
# decide which one fits the question, without your code hard-coding the logic.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the company's document knowledge base for general "
                "information — company overview, history, policies, "
                "services, or anything not tracked as live structured data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, based on the visitor's question.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_employee_info",
            "description": (
                "Get information about company employees — name, role, "
                "department, contact info. Omit 'name' to list all employees."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Employee name, role, or department to search for. Omit for all employees.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_project_info",
            "description": (
                "Get information about company projects — client, status, "
                "current phase, progress, tech stack. Omit 'project_name' "
                "to list all projects."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project_name": {
                        "type": "string",
                        "description": "Project name to search for. Omit for all projects.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_department_info",
            "description": (
                "Get information about company departments — head, team "
                "size, description. Omit 'department_name' to list all "
                "departments."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "department_name": {
                        "type": "string",
                        "description": "Department name to search for. Omit for all departments.",
                    }
                },
                "required": [],
            },
        },
    },
]


def _execute_tool_call(tool_call) -> str:
    name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        args = {}

    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return f"Unknown tool: {name}"

    try:
        return func(args)
    except Exception as e:
        return f"Error running {name}: {e}"


def clean_response(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def ask(question: str, conversation_history: list = None, max_retries: int = 2) -> str:
    if conversation_history is None:
        conversation_history = []

    today_str = datetime.now().strftime("%A, %B %d, %Y")

    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nToday's date is {today_str}."}
    ]
    for turn in conversation_history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    messages.append({"role": "user", "content": question})

    for attempt in range(max_retries + 1):
        try:
            return _run_tool_calling_loop(messages)
        except (APITimeoutError, APIConnectionError) as e:
            print(f"Attempt {attempt + 1} failed ({e}), retrying...")
            if attempt == max_retries:
                return "Sorry, I'm having trouble connecting right now. Please try asking again."


def _run_tool_calling_loop(messages: list, max_rounds: int = 3) -> str:
    """Lets the LLM decide, per question, whether it needs a tool — and
    which one. Vectorization (search_documents) and SQL queries only run
    if the LLM actually asks for that specific tool."""

    for round_num in range(max_rounds):
        t0 = time.time()
        response = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            reasoning_effort="none",
            timeout=15.0,
        )
        print(f"LLM round {round_num + 1} took {time.time() - t0:.2f}s")

        msg = response.choices[0].message

        if not msg.tool_calls:
            # No tool needed (or the LLM is done) — this is the final answer
            return clean_response(msg.content)

        # Record the assistant's tool-call request, then execute each tool
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })

        for tool_call in msg.tool_calls:
            print(f"LLM requested tool: {tool_call.function.name}({tool_call.function.arguments})")
            result_text = _execute_tool_call(tool_call)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_text,
            })

    # Safety net: if the LLM keeps calling tools past max_rounds, force a
    # final answer without offering more tools.
    final_response = llm.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        reasoning_effort="none",
        timeout=15.0,
    )
    return clean_response(final_response.choices[0].message.content)