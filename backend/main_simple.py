from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

app = FastAPI(
    title="AI Agent Platform API",
    description="API for creating and managing AI agents",
    version="1.0.0",
)

# Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "AI Agent Platform API", "version": "1.0.0", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "database": "not connected", "ai_services": "configured"}

@app.get("/api/v1/agents")
async def get_agents():
    return {"agents": [], "message": "Agents endpoint working"}

@app.post("/api/v1/agents")
async def create_agent(agent_data: dict):
    return {
        "id": 1,
        "message": "Agent created successfully",
        "agent": {
            "name": agent_data.get("name", "Test Agent"),
            "status": "active"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_simple:app", host="0.0.0.0", port=8000, reload=True)