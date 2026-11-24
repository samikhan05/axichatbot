import os
import re
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion
from openai import OpenAI, APITimeoutError, APIConnectionError

from language import detect_language, LANGUAGE_INSTRUCTIONS, EN
from sparse_encoder import encode_query
from speech import make_speakable

from db_query import (
    get_all_employees, search_employee,
    get_all_projects, get_project_status,
    get_all_departments, get_department_info as db_get_department_info,
    format_employee_context, format_project_context, format_department_context
)

load_dotenv()

COLLECTION_NAME = "company_knowledge"
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

TOP_K = int(os.getenv("RAG_TOP_K", "3"))             # chunks given to the LLM
CANDIDATES = int(os.getenv("RAG_CANDIDATES", "12"))  # per retriever, before fusion/rerank
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "false").lower() in ("1", "true", "yes")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

embed_model = SentenceTransformer("BAAI/bge-m3")
qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

_reranker = None


def get_reranker():
    """Cross-encoder reranker, loaded lazily on first use. bge-reranker-v2-m3
    is multilingual, so it also scores Urdu-script queries correctly."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_MODEL, max_length=512)
    return _reranker

llm = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)

# Must support tool calling on Groq. Check what your key can use with:
#   GET https://api.groq.com/openai/v1/models
MODEL_NAME = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
# gpt-oss models accept low / medium / high; qwen3 models also accept "none"
REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "low")

SYSTEM_PROMPT = (
    "You are the front-desk receptionist of AXI Technologies, speaking to a "
    "visitor in the office lobby. Sound like a polished, warm, professional "
    "human receptionist: speak as the company ('we', 'our team'), confident "
    "and courteous.\n\n"
    "How to answer:\n"
    "- Use your tools whenever the question needs company information you "
    "don't already have in this conversation. Greetings and small talk need "
    "no tool.\n"
    "- Answer only from what the tools return. Never invent facts, and never "
    "guess the current date, time or outside-world facts.\n"
    "- Keep it to 1-2 short, complete, natural sentences. Say each fact once; "
    "never repeat or rephrase the same sentence.\n"
    "- Never mention tools, databases, records, documents, lists, sources or "
    "'listed' / 'according to'. A receptionist just knows: say 'We have 45 "
    "people across eight departments', not 'six employees are listed'.\n"
    "- Headcount questions: the employee directory only contains key "
    "contacts, NOT all staff. For total staff, add up the department team "
    "sizes (from the department info or the company documents).\n"
    "- If the tools don't cover the question, never claim that something "
    "does or doesn't exist. Say politely that you don't have that detail at "
    "hand, and offer to connect them with someone or help with the company, "
    "our projects, our team or office hours.\n"
    "- Plain spoken sentences only: no markdown, bullet points, numbering or "
    "bold, because the reply is spoken aloud. With several items, use short "
    "separate sentences rather than a long comma list.\n"
    "- Write everything the way it is said out loud: no slashes (say "
    "'medical or sick leave'), no brackets, no '&', and no abbreviations "
    "like PKT, approx or e.g. Say '9 AM to 6 PM', not '09:00 AM - 06:00 PM "
    "(PKT)'. For split values say both, e.g. '90 days for maternity and 10 "
    "days for paternity', never '90/10'.\n"
    "- Only greet if the visitor greets first.\n\n"
    "Language: the knowledge base and database are in English. Whatever "
    "language the visitor uses (English, Urdu or Roman Urdu), always write "
    "tool arguments in English: translate the question into a short English "
    "search query and use English spellings of names, departments and "
    "projects."
)


# ---------------------------------------------------------------------------
# Tool implementations — the actual work each tool does when the LLM calls it
# ---------------------------------------------------------------------------

def hybrid_search(query: str, top_k: int = TOP_K, mode: str = "hybrid", rerank: bool | None = None):
    """Retrieve chunks for `query`.

    mode="hybrid": dense (BGE-M3) + sparse (BM25) candidates fused with
                   Reciprocal Rank Fusion inside Qdrant.
    mode="dense":  dense only (the old behaviour, kept for evaluation).
    rerank:        re-score the candidates with a cross-encoder.
    Returns a list of dicts: text, source, section, type, score.
    """
    rerank = RERANK_ENABLED if rerank is None else rerank
    fetch = max(CANDIDATES, top_k) if rerank else top_k
    dense = embed_model.encode(query, normalize_embeddings=True).tolist()

    if mode == "dense":
        points = qdrant.query_points(
            collection_name=COLLECTION_NAME, query=dense, using=DENSE_VECTOR,
            limit=fetch, with_payload=True,
        ).points
    else:
        points = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(query=dense, using=DENSE_VECTOR, limit=CANDIDATES),
                Prefetch(query=encode_query(query), using=SPARSE_VECTOR, limit=CANDIDATES),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=fetch,
            with_payload=True,
        ).points

    hits = [
        {
            "text": p.payload.get("text", ""),
            "source": p.payload.get("source", ""),
            "section": p.payload.get("section", ""),
            "type": p.payload.get("type", "text"),
            "score": float(p.score),
        }
        for p in points
    ]

    if rerank and hits:
        scores = get_reranker().predict([(query, h["text"]) for h in hits])
        for h, sc in zip(hits, scores):
            h["score"] = float(sc)
        hits.sort(key=lambda h: h["score"], reverse=True)

    return hits[:top_k]


def _tool_search_documents(query: str, trace: dict) -> str:
    hits = hybrid_search(query)
    if not hits:
        return "No relevant documents found."
    blocks = []
    for h in hits:
        label = h["source"] + (f" > {h['section']}" if h["section"] else "")
        trace["sources"].append({"source": h["source"], "section": h["section"]})
        blocks.append(f"[{label}]\n{h['text']}")
    return "\n\n".join(blocks)


NOT_IN_DB = (
    "No matching {what} found in the live database. The company documents "
    "may still have it: call search_documents before telling the visitor "
    "it does not exist."
)


def _tool_get_employee_info(name: str = None) -> str:
    employees = search_employee(name) if name else get_all_employees()
    text = format_employee_context(employees)
    return text if text else NOT_IN_DB.format(what="employee")


def _tool_get_project_info(project_name: str = None) -> str:
    if project_name:
        project = get_project_status(project_name)
        projects = [project] if project else []
    else:
        projects = get_all_projects()
    text = format_project_context(projects)
    return text if text else NOT_IN_DB.format(what="project")


def _tool_get_department_info(department_name: str = None) -> str:
    if department_name:
        dept = db_get_department_info(department_name)
        departments = [dept] if dept else []
    else:
        departments = get_all_departments()
    text = format_department_context(departments)
    return text if text else NOT_IN_DB.format(what="department")


TOOL_FUNCTIONS = {
    "search_documents": lambda args, trace: _tool_search_documents(args.get("query", ""), trace),
    "get_employee_info": lambda args, trace: _tool_get_employee_info(args.get("name")),
    "get_project_info": lambda args, trace: _tool_get_project_info(args.get("project_name")),
    "get_department_info": lambda args, trace: _tool_get_department_info(args.get("department_name")),
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
                        "description": "A short search query IN ENGLISH, translated from the visitor's question if needed.",
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
                "Look up key contacts in the staff directory — name, role, "
                "department, email, phone. This is a directory of key "
                "contacts, not the full staff headcount. Omit 'name' to list "
                "all key contacts."
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


def _execute_tool_call(tool_call, trace: dict) -> str:
    name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        args = {}

    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return f"Unknown tool: {name}"

    trace["tools"].append({"name": name, "args": args})
    try:
        return func(args, trace)
    except Exception as e:
        return f"Error running {name}: {e}"


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _words(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)}


def clean_response(text: str) -> str:
    """Strip reasoning tags and fix a model glitch where the answer is
    emitted twice, glued together ("...listed.Six employees are...")."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()

    # sentence end directly followed by a capital letter = two generations glued
    parts = re.split(r"(?<=[a-z0-9][.!?])(?=[A-Z])", text)
    if len(parts) > 1:
        kept = [parts[0]]
        for part in parts[1:]:
            a, b = _words(kept[-1]), _words(part)
            overlap = len(a & b) / max(1, min(len(a), len(b)))
            if overlap < 0.5:  # genuinely new content: keep it, add the space
                kept.append(part)
        text = " ".join(p.strip() for p in kept)

    # drop exact repeated sentences
    seen, out = set(), []
    for sent in re.split(r"(?<=[.!?\u06d4])\s+", text):
        key = " ".join(sorted(_words(sent)))
        if key and key in seen:
            continue
        seen.add(key)
        out.append(sent)
    return " ".join(out).strip()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def ask(question: str, conversation_history: list = None, max_retries: int = 2) -> str:
    """Backwards-compatible: returns only the answer text."""
    return ask_with_meta(question, conversation_history, max_retries=max_retries)["answer"]


def ask_with_meta(question: str, conversation_history: list = None,
                  language: str | None = None, max_retries: int = 2) -> dict:
    """Answer a question and report how: detected language, tools the LLM
    chose, and which document chunks were retrieved (for citations)."""
    if conversation_history is None:
        conversation_history = []

    if language is None:
        previous = conversation_history[-1].get("language") if conversation_history else None
        language = detect_language(question, fallback=previous)

    today_str = datetime.now().strftime("%A, %B %d, %Y")
    system = (
        f"{SYSTEM_PROMPT}\n\nToday's date is {today_str}.\n\n"
        f"{LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS[EN])}"
    )

    messages = [{"role": "system", "content": system}]
    for turn in conversation_history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    messages.append({"role": "user", "content": question})

    trace = {"tools": [], "sources": []}
    answer = "Sorry, I'm having trouble connecting right now. Please try asking again."
    for attempt in range(max_retries + 1):
        try:
            answer = _run_tool_calling_loop(list(messages), trace)
            break
        except (APITimeoutError, APIConnectionError) as e:
            print(f"Attempt {attempt + 1} failed ({e}), retrying...")
            trace = {"tools": [], "sources": []}

    # de-duplicate sources, keep order
    seen, sources = set(), []
    for src in trace["sources"]:
        key = (src["source"], src["section"])
        if key not in seen:
            seen.add(key)
            sources.append(src)

    # same text on screen as the avatar says: no "/", "(PKT)", "&" etc.
    answer = make_speakable(answer, language)
    return {"answer": answer, "language": language, "tools": trace["tools"], "sources": sources}


def _run_tool_calling_loop(messages: list, trace: dict, max_rounds: int = 3) -> str:
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
            reasoning_effort=REASONING_EFFORT,
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
            result_text = _execute_tool_call(tool_call, trace)
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
        reasoning_effort=REASONING_EFFORT,
        timeout=15.0,
    )
    return clean_response(final_response.choices[0].message.content)