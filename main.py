import os
import re
import uuid
import asyncio
import aiohttp
import traceback  # Added for deep error logging
from urllib.parse import urlparse, urljoin
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from fastapi.middleware.cors import CORSMiddleware

# ================== ENV ==================
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not GOOGLE_API_KEY or not QDRANT_URL:
    raise ValueError("Missing environment variables. Please check your Render environment setup.")

# ================== CLIENTS ==================
gemini_client = genai.Client(api_key=GOOGLE_API_KEY)

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# ================== APP ==================
app = FastAPI(title="RAG Website Chat API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== MODELS ==================
class CrawlRequest(BaseModel):
    url: str
    max_depth: int = 2
    max_pages: int = 10

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    collection_name: str
    question: str
    history: List[Message] = []

class CrawlResponse(BaseModel):
    message: str
    collection_name: str
    pages_indexed: int
    crawled_urls: List[str]
    suggested_questions: List[str]

# ================== HELPERS ==================
def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    scheme = parsed.scheme or "https"
    return f"{scheme}://{parsed.netloc.lower()}"

def sanitize_collection_name(url: str) -> str:
    clean = normalize_url(url).replace("https://", "").replace("http://", "").replace(".", "_")
    return f"site_{clean}"[:63]

def extract_links_from_markdown(text: str, base_url: str, domain: str) -> set:
    links = re.findall(r'\[.*?\]\((.*?)\)', text)
    results = set()
    for link in links:
        try:
            full = urljoin(base_url, link).rstrip("/")
            if domain in urlparse(full).netloc:
                results.add(full)
        except:
            pass
    return results

# ================== FETCH ==================
async def fetch_text(session, url):
    jina_url = f"https://r.jina.ai/{url}"
    try:
        async with session.get(jina_url, timeout=20) as r:
            if r.status == 200:
                text = await r.text()
                return text if len(text) > 200 else None
    except:
        pass
    return None

# ================== PROCESS PAGE ==================
async def process_url(session, url, depth, collection, domain, max_depth):
    print(f"🔍 Fetching [{depth}] {url}")

    text = await fetch_text(session, url)
    if not text:
        print(f"⚠️ Skipped (empty or failed to fetch): {url}")
        return set(), 0, "" 

    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    points = []

    for i, chunk in enumerate(chunks, 1):
        # FIX 1: Updated the deprecated text-embedding-004 model to gemini-embedding-001
        emb = gemini_client.models.embed_content(
            model="gemini-embedding-001",
            contents=chunk
        )
        vector = emb.embeddings[0].values

        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"text": chunk, "url": url}
        ))

    if points:
        qdrant_client.upsert(collection_name=collection, points=points)
        print(f"✅ Indexed {len(points)} chunks from {url}")

    links = set()
    if depth < max_depth:
        links = extract_links_from_markdown(text, url, domain)

    return links, len(points), text 

# ================== QUESTION GENERATOR ==================
async def generate_suggested_questions(text_blob: str) -> List[str]:
    """Generates 5 engagement questions using Gemini."""
    if not text_blob:
        return ["What is this website about?", "What services are offered?", "How can I contact support?", "What are the pricing details?", "Tell me more."]
    
    prompt = f"""Analyze the following website content and generate exactly 5 specific and logical questions a user would likely ask a chatbot about the website content. Return ONLY a plain list of questions, one per line.
    
    CONTENT:
    {text_blob[:6000]}"""
    
    try:
        response = gemini_client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        questions = [q.strip() for q in response.text.strip().split('\n') if q.strip()]
        return questions[:5]
    except Exception as e:
        print(f"⚠️ Failed to generate questions: {e}")
        return ["Tell me more about this site.", "What services are offered?", "How can I contact you?"]

# ================== CRAWLER ==================
async def run_crawler(start_url: str, max_depth: int, max_pages: int):
    base_url = normalize_url(start_url)
    collection = sanitize_collection_name(base_url)
    domain = urlparse(base_url).netloc
    
    print(f"⚙️ Checking if collection '{collection}' exists in Qdrant...")
    collection_exists = qdrant_client.collection_exists(collection_name=collection)
    
    if collection_exists:
        print("⚡ Collection exists. Fetching existing context for questions...")
        existing_points = qdrant_client.scroll(collection_name=collection, limit=3, with_payload=True)[0]
        context = " ".join([p.payload.get('text', '') for p in existing_points])
        questions = await generate_suggested_questions(context)
        return collection, 0, [], questions
    else:
        print(f"⚙️ Collection does NOT exist. Creating new collection: {collection}")
        # Create collection if it doesn't exist
        qdrant_client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )

    visited = set()
    crawled_urls = []
    queue = [(base_url, 1)]
    first_page_text = ""

    async with aiohttp.ClientSession() as session:
        while queue and len(visited) < max_pages:
            url, depth = queue.pop(0)
            url = url.rstrip("/")

            if url in visited: continue
            visited.add(url)
            crawled_urls.append(url)

            new_links, chunks, text = await process_url(
                session, url, depth, collection, domain, max_depth
            )
            
            # Save the text of the first successful page for questions
            if not first_page_text and text:
                first_page_text = text

            for link in new_links:
                if link not in visited and len(visited) < max_pages:
                    queue.append((link, depth + 1))

            await asyncio.sleep(0.3)

    # Generate questions at the end of a new crawl
    suggested_questions = await generate_suggested_questions(first_page_text)

    print("🎉 Crawl completed")
    return collection, len(visited), crawled_urls, suggested_questions

# ================== ROUTES ==================
@app.post("/crawl", response_model=CrawlResponse)
async def crawl_site(req: CrawlRequest):
    try:
        collection, count, urls, questions = await run_crawler(
            req.url, req.max_depth, req.max_pages
        )
        return {
            "message": "Ready",
            "collection_name": collection,
            "pages_indexed": count,
            "crawled_urls": urls,
            "suggested_questions": questions
        }
    except Exception as e:
        print("\n❌ ================= CRITICAL ERROR IN /crawl =================")
        traceback.print_exc()
        print("==============================================================\n")
        raise HTTPException(500, str(e))

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        # FIX 2: Updated the deprecated text-embedding-004 model to gemini-embedding-001
        emb = gemini_client.models.embed_content(
            model="gemini-embedding-001",
            contents=req.question
        )
        query_vector = emb.embeddings[0].values

        results = qdrant_client.query_points(
            collection_name=req.collection_name,
            query=query_vector,
            limit=5,
            with_payload=True
        )

        context = ""
        sources = []

        for hit in results.points:
            context += f"{hit.payload['text']}\n\n"
            if hit.payload["url"] not in sources:
                sources.append(hit.payload["url"])

        prompt = f"""
        CONTENT:
        {context}

        QUESTION:
        {req.question}

        INSTRUCTIONS:
        - Answer only using the above content
        - Keep the response with well-formatted extra text
        - Do not include unnecessary explanations

        ANSWER:"""

        SYSTEM_PROMPT = """
You are an AI assistant for a Retrieval-Augmented Generation (RAG) system.

You must generate answers using ONLY the information provided in the CONTENT.
Do NOT use any prior knowledge or assumptions.

Rules:
- strictly answer in the same language as the question.
- If the answer is not present, say the information is not available.
- Ensure clarity, correctness, and good formatting.
- Do not hallucinate.
"""
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.9,
                max_output_tokens=1024
            )
        )

        return {"answer": response.text, "sources": sources}

    except Exception as e:
        print("\n❌ ================= CRITICAL ERROR IN /chat =================")
        traceback.print_exc()
        print("==============================================================\n")
        raise HTTPException(500, str(e))

# ================== RUN ==================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)