from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    # Allow all the methods HTTP:
    allow_methods = ["*"],
    allow_headers = ["*"]
)


# Definition of POST request:
class QueryPayload(BaseModel):
    text:str


@app.post("/query")
async def process_agent(payload: QueryPayload):
    