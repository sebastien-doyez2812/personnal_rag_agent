from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import RagAgent
from vectordb import VectorDB
from fastapi.staticfiles import StaticFiles
import os

# FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173")

# origins = [url.strip() for url in FRONTEND_URL.split(',')]

# For local dev:
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173" 
]

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

# Make image accessible for the frontend:
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img_dir = "C:/Users/doyez/Documents/personnal_rag_agent/data/imgs_only" #os.path.join(BASE_DIR, "data", "imgs_only")
app.mount("/static/images", StaticFiles(directory=img_dir), name="static_images")


@app.post("/query")
async def process_agent(payload: QueryPayload):
    # Ask to my agent:
    print(f"[i] received query: {payload.text}")
    rep, path_img = my_agent.invoke(query= payload.text)
    print(f"Answer is {rep}, image is {path_img}")
    if path_img:
        img_filename = os.path.basename(path_img)
        path_img_for_frontend = f"/static/images/{img_filename}"
        return { "query" : payload.text,
            "response": rep,
            "path_img": path_img_for_frontend
            }
    else:
        return { "query" : payload.text,
            "response": rep,
            }