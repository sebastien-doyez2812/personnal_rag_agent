from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import RagAgent
from vectordb import VectorDB
import os

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173")

origins = [url.strip() for url in FRONTEND_URL.split(',')]

# For local dev:
# origins = [
#     "http://localhost:3000",
#     "http://127.0.0.1:3000",
#     "http://localhost:5173" 
# ]

my_vectordb = VectorDB()
my_agent = RagAgent(vector_db= my_vectordb)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Definition of POST request:
class QueryPayload(BaseModel):
    text:str


@app.post("/query")
async def process_agent(payload: QueryPayload):
    # Ask to my agent:
    print(f"[i] received query: {payload.text}")
    rep = my_agent.invoke(query= payload.text)
    print(f"Answer is {rep}")
    return { "query" : payload.text,
        "response": rep 
        }