from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from Rag import bot

app = FastAPI(title="HeadlinerChatbotAPI")

class QuestionRequest(BaseModel):
    question: str
    player_id: str = "unknown"

class AnswerResponse(BaseModel):
    answer: str

@app.get("/")
def read_root():
    return {"status": "ok", "message": "running"}

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    try:
        print(f"{request.player_id} asked: {request.question}")
        result = bot.ask(request.question)
        return AnswerResponse(answer=result)   
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))