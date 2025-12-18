import os
import re
import uuid
import asyncio
import aiohttp
from urllib.parse import urlparse, urljoin
from typing import List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Body
from pydantic import BaseModel
from dotenv import load_dotenv

from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from fastapi.middleware.cors import CORSMiddleware # <--- IMPORT THIS

# --- LOAD ENV VARS ---
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not GOOGLE_API_KEY or not QDRANT_URL:
    raise ValueError("Missing GOOGLE_API_KEY or QDRANT_URL in environment variables.")

# --- INITIALIZE CLIENTS ---
gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

app = FastAPI(title="RAG Chatbot API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ----------------------

# --- DATA MODELS ---

class CrawlRequest(BaseModel):
    url: str
    max_depth: int = 2
    max_pages: int = 20

class Message(BaseModel):
    role: str # "user" or "assistant" or "AI"
    content: str

class ChatRequest(BaseModel):
    collection_name: str
    question: str
    history: List[Message] = []

class CrawlResponse(BaseModel):
    message: str
    collection_name: str
    pages_indexed: int

# --- HELPER FUNCTIONS ---

def get_domain_from_url(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc

def sanitize_collection_name(url: str) -> str:
    # Create a safe name for Qdrant collection
    clean = url.replace("https://", "").replace("http://", "").replace("/", "_").replace(".", "_")
    return f"jina_{clean}"[:63] # Qdrant has length limits usually

def extract_links_from_markdown(markdown_text: str, base_url: str, target_domain: str) -> set:
    link_pattern = r'\[.*?\]\(([^\)]+)\)'
    links = re.findall(link_pattern, markdown_text)
    clean_links = set()
    for link in links:
        link = link.strip()
        try:
            full_url = urljoin(base_url, link)
            # Only keep links within the same domain
            if target_domain in urlparse(full_url).netloc:
                clean_url = full_url.rstrip('/')
                clean_links.add(clean_url)
        except:
            continue
    return clean_links

async def fetch_text_async(session, url):
    """Fetches text via Jina Reader"""
    jina_url = f"https://r.jina.ai/{url}"
    headers = {'User-Agent': 'FastAPI-RAG-Crawler'}
    try:
        async with session.get(jina_url, headers=headers, timeout=20) as response:
            if response.status == 200:
                text = await response.text()
                # Simple validation
                if len(text) < 200:
                    return None
                return text
            return None
    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
        return None

async def process_url(session, url, current_depth, collection_name, target_domain, max_depth):
    text = await fetch_text_async(session, url)
    if not text:
        return set()

    # Chunking
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    points = []

    # Embedding and Preparing Points
    for chunk in chunks:
        try:
            emb_result = gemini_client.models.embed_content(
                model="text-embedding-004",
                contents=chunk
            )
            vector = emb_result.embeddings[0].values
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": chunk, "url": url}
            ))
        except Exception as e:
            print(f"Embedding error: {e}")

    # Upsert to Qdrant
    if points:
        try:
            qdrant_client.upsert(collection_name=collection_name, points=points)
            print(f"✅ Indexed: {url}")
        except Exception as e:
            print(f"Qdrant Upsert error: {e}")

    # Return new links to crawl
    if current_depth < max_depth:
        return extract_links_from_markdown(text, url, target_domain)
    return set()

# --- CORE LOGIC ---

async def run_crawler(start_url: str, max_depth: int, max_pages: int):
    target_domain = get_domain_from_url(start_url)
    collection_name = sanitize_collection_name(start_url)
    
    # Check if collection exists
    collections = qdrant_client.get_collections()
    existing_names = [c.name for c in collections.collections]
    
    if collection_name not in existing_names:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
    else:
        # Optional: Decide if you want to wipe it or append. 
        # For now, we append/overwrite similar IDs, but let's just proceed.
        print(f"Collection {collection_name} exists.")

    visited = set()
    queue = [(start_url, 1)] # Tuple: (url, depth)
    BATCH_SIZE = 5

    async with aiohttp.ClientSession() as session:
        while queue and len(visited) < max_pages:
            batch_tasks = []
            
            # Create a batch of tasks
            while queue and len(batch_tasks) < BATCH_SIZE and len(visited) + len(batch_tasks) < max_pages:
                url, depth = queue.pop(0)
                url = url.rstrip('/')
                
                if url in visited: 
                    continue
                visited.add(url)
                
                # Add task
                task = process_url(session, url, depth, collection_name, target_domain, max_depth)
                batch_tasks.append((task, depth))

            if not batch_tasks:
                break

            # Run batch
            print(f"🚀 Processing batch of {len(batch_tasks)}...")
            tasks_only = [t[0] for t in batch_tasks]
            depths_only = [t[1] for t in batch_tasks]
            
            results = await asyncio.gather(*tasks_only)

            # Process results (new links)
            for i, new_links in enumerate(results):
                current_depth = depths_only[i]
                if current_depth < max_depth and new_links:
                    for link in new_links:
                        if link not in visited and not any(link == q[0] for q in queue):
                            queue.append((link, current_depth + 1))
            
            await asyncio.sleep(0.5) # Polite delay

    return collection_name, len(visited)

# --- ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "running", "service": "RAG-Backend"}

@app.post("/crawl", response_model=CrawlResponse)
async def start_crawl(request: CrawlRequest):
    """
    Crawls a website and indexes it into Qdrant.
    Returns the collection name to be used in /chat.
    """
    try:
        print(f"Starting crawl for {request.url}")
        collection_name, pages_count = await run_crawler(
            request.url, 
            request.max_depth, 
            request.max_pages
        )
        return CrawlResponse(
            message="Crawling complete",
            collection_name=collection_name,
            pages_indexed=pages_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Asks a question against a specific collection.
    """
    try:
        # 1. Embed Query
        emb_result = gemini_client.models.embed_content(
            model="text-embedding-004",
            contents=request.question
        )
        query_vector = emb_result.embeddings[0].values

        # 2. Retrieve from Qdrant
        search_response = qdrant_client.query_points(
            collection_name=request.collection_name,
            query=query_vector,
            limit=5,
            with_payload=True
        )

        # 3. Build Context
        context_text = ""
        sources = []
        if search_response.points:
            context_parts = []
            for hit in search_response.points:
                url = hit.payload.get('url', 'Unknown')
                text = hit.payload.get('text', '')
                context_parts.append(f"Source: {url}\nContent: {text}")
                if url not in sources:
                    sources.append(url)
            context_text = "\n\n---\n\n".join(context_parts)
        else:
            context_text = "No direct context found on the website for this query."

        # 4. Format History
        history_text = ""
        for msg in request.history[-5:]: # Limit history size
            role = "User" if msg.role.lower() in ["user"] else "AI"
            history_text += f"{role}: {msg.content}\n"

        # 5. System Instruction
        sys_instruction = (
            "You are a helpful AI assistant. "
            "Use the provided 'Context' to answer the 'Current Question'. "
            "Use 'Chat History' for context. "
            "Strictly cite sources from the context provided. "
            "If the answer isn't in the context, say so."
        )

        full_prompt = (
            f"Context from Website:\n{context_text}\n\n"
            f"Chat History:\n{history_text}\n\n"
            f"Current Question: {request.question}\n\n"
            "Answer:"
        )

        # 6. Generate Answer
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=full_prompt,
            config=types.GenerateContentConfig(system_instruction=sys_instruction)
        )

        return {
            "answer": response.text,
            "sources": sources,
            "collection_used": request.collection_name
        }

    except Exception as e:
        # Handle Qdrant missing collection error specifically
        if "Not found: Collection" in str(e):
             raise HTTPException(status_code=404, detail="Collection not found. Please crawl the URL first.")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)