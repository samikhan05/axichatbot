import os
import re
import time
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from openai import OpenAI, APITimeoutError, APIConnectionError

load_dotenv()

embed_model = SentenceTransformer("BAAI/bge-m3")
qdrant = QdrantClient(url="http://localhost:6333")

llm = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)

SYSTEM_PROMPT = (
    "You are a company receptionist. Answer using only the information provided. "
    "Keep answers short — 1 to 2 sentences maximum. Be direct and natural, like a "
    "real receptionist speaking out loud, not a written document. No extra "
    "explanations, no repeating the question, no unnecessary greetings unless the "
    "visitor greets first. Do not use markdown formatting, bullet points, numbered "
    "lists, or bold text — write plain, spoken-style sentences only, since your "
    "response will eventually be converted to speech. Never guess or state the "
    "current date, time, or any real-world fact not explicitly provided to you. "
    "If the answer isn't in the provided information, say so briefly and suggest "
    "asking about the company, its projects, or its team."
    "When mentioning multiple items (like multiple projects or names), state them as short separate sentences instead of one long comma-separated list, since long lists in a single sentence don't sound natural when spoken aloud."
)


def clean_response(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def ask(question: str, max_retries: int = 2) -> str:
    t0 = time.time()
    query_vector = embed_model.encode(question).tolist()
    t1 = time.time()
    print(f"Embedding took {t1 - t0:.2f}s")

    results = qdrant.query_points(
        collection_name="company_knowledge",
        query=query_vector,
        limit=3,
    ).points
    t2 = time.time()
    print(f"Qdrant search took {t2 - t1:.2f}s")

    context = "\n".join([r.payload["text"] for r in results])
    today_str = datetime.now().strftime("%A, %B %d, %Y")

    prompt = f"""Today's date is {today_str}.

Company Information:
{context}

Visitor's Question: {question}"""

    for attempt in range(max_retries + 1):
        try:
            response = llm.chat.completions.create(
                model="qwen/qwen3.6-27b",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                reasoning_effort="none",
                timeout=15.0,
            )
            t3 = time.time()
            print(f"LLM response took {t3 - t2:.2f}s")
            return clean_response(response.choices[0].message.content)
        except (APITimeoutError, APIConnectionError) as e:
            print(f"Attempt {attempt + 1} failed ({e}), retrying...")
            if attempt == max_retries:
                return "Sorry, I'm having trouble connecting right now. Please try asking again."