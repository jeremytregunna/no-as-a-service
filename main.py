from fastapi import FastAPI, Response
from typing import List
import uvicorn
import time

import json
from markov import MarkovChain

app = FastAPI(
    title="No as a Service",
    description="Generate creative ways to say no using Markov chains",
    version="1.0.0"
)

# Load training data from reasons.json
with open("reasons.json", "r") as f:
    TRAINING_PHRASES = json.load(f)

# Initialize and train the Markov chain
markov = MarkovChain(order=2)
markov.train(TRAINING_PHRASES)


@app.get("/")
async def root():
    return {
        "message": "Welcome to No as a Service!",
        "description": "Generate creative ways to say no",
        "endpoints": {
            "/no": "Get a single creative way to say no",
            "/no/multiple": "Get multiple creative ways to say no",
            "/health": "Check if the service is running"
        }
    }


@app.get("/no")
async def get_no(response: Response):
    start_time = time.perf_counter()
    generated_text = markov.generate(max_length=20)
    end_time = time.perf_counter()
    
    generation_time_ms = (end_time - start_time) * 1000
    response.headers["X-Generation-Time-Ms"] = f"{generation_time_ms:.3f}"
    
    return {"response": generated_text}


@app.get("/no/multiple")
async def get_multiple_nos(response: Response, count: int = 5):
    if count > 20:
        count = 20  # Limit to prevent abuse
    if count < 1:
        count = 1

    start_time = time.perf_counter()
    responses = []
    for _ in range(count):
        responses.append(markov.generate(max_length=20))
    end_time = time.perf_counter()
    
    generation_time_ms = (end_time - start_time) * 1000
    response.headers["X-Generation-Time-Ms"] = f"{generation_time_ms:.3f}"

    return {
        "responses": responses,
        "count": len(responses)
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "markov_trained": markov.is_trained(),
        "training_phrases_count": len(TRAINING_PHRASES)
    }


def main():
    uvicorn.run("main:app", host="0.0.0.0", port=8100, reload=True)


if __name__ == "__main__":
    main()
