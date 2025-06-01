from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import os

import httpx

app = FastAPI(docs_url="/api/docs")


class InputData(BaseModel):
    domain: str
    timestamp: str


class ReturnData(BaseModel):
    score: float


MODEL_URL: str | None = os.getenv("MODEL_URL")


@app.post("/api/v1/score", response_model=ReturnData, status_code=200)
async def predict(data: list[InputData]) -> ReturnData:
    if MODEL_URL is None:
        raise HTTPException(status_code=400, detail="U need specify MODEL_URL")
    
    payload = {
        "inputs": [
            {
                "name": "input",
                "shape": [1],
                "datatype": "BYTES",
                "data": [{"domain": row.domain, "timestamp": row.timestamp} for row in data]
            }
        ]
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(MODEL_URL, json=payload)

    if response.status_code == 200:
        data = response.json()
        return ReturnData(score=data["outputs"][0]["data"][0])
    
    else:
        raise HTTPException(status_code=502, detail="Error from model")
