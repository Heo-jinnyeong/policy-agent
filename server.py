"""Policy Agent API with persistent conversation and validated SSE responses."""
import json
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from pydantic import BaseModel, Field

from agent.deps import SupervisorDeps
from agent.fake_llm import FastLLM, QualityLLM
from agent.graph import build_graph
from agent.mcp_client import MCPClient
from agent.policy_llm import PolicyLLM
from agent.session_store import SqliteSessionStore

CREATION_COUNT = 0


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    thread_id: str = "default"
    domain: str = "academic"


def _build_llms():
    backend = os.getenv("LLM_BACKEND", "fake")
    if backend == "ollama":
        from agent.ollama_llm import OllamaFastLLM, OllamaQualityLLM
        return backend, OllamaFastLLM(os.getenv("OLLAMA_FAST_MODEL", "qwen2.5:0.5b"), 0.00015), OllamaQualityLLM(os.getenv("OLLAMA_QUALITY_MODEL", "qwen3:8b"), 0.005)
    if backend == "openai":
        from agent.openai_llm import OpenAIFastLLM, OpenAIQualityLLM
        return backend, OpenAIFastLLM(os.getenv("OPENAI_FAST_MODEL", "gpt-4o-mini"), 0.00015), OpenAIQualityLLM(os.getenv("OPENAI_QUALITY_MODEL", "gpt-4o"), 0.005)
    return "fake", FastLLM("fake", 0), QualityLLM("fake", 0)


@asynccontextmanager
async def lifespan(application: FastAPI):
    global CREATION_COUNT
    CREATION_COUNT += 1
    mcp = MCPClient()
    await mcp.start()
    backend, fast_llm, quality_llm = _build_llms()
    deps = SupervisorDeps(fast_llm=fast_llm, quality_llm=quality_llm, mcp=mcp,
                          session_store=SqliteSessionStore(), version="policy-v2",
                          policy_llm=PolicyLLM(quality_llm))
    application.state.deps = deps
    application.state.graph = build_graph(deps)
    application.state.llm_backend = backend
    application.state.instance_id = id(deps)
    try:
        yield
    finally:
        await mcp.stop()
        deps.session_store.close()


app = FastAPI(title="Policy Agent", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "instance_id": app.state.instance_id,
            "creation_count": CREATION_COUNT, "llm_backend": app.state.llm_backend,
            "model": app.state.deps.quality_llm.name}


@app.get("/version")
async def version():
    return {"version": "policy-v2"}


def _response(result: dict, run_id=None) -> dict:
    evidence = result.get("evidence", [])
    return {"reply": result.get("final", ""), "analysis": result.get("analysis", {}),
            "sources": [{k: e.get(k) for k in ("id", "document_id", "document_name", "article",
                                              "page", "effective_date", "source", "text")}
                        for e in evidence],
            "events": result.get("events", []),
            "metadata": {"domain": result.get("domain"), "route": result.get("route"),
                         "research_rounds": result.get("research_rounds", 0),
                         "validation_attempts": result.get("validation_attempts", 0),
                         "tokens": result.get("tokens", 0),
                         "estimated_cost_usd": result.get("cost_usd", 0),
                         "langsmith_run_id": run_id}}


@traceable(name="policy_chat_request")
async def _handle(req: ChatRequest):
    deps = app.state.deps
    result = await app.state.graph.ainvoke({"query": req.message, "thread_id": req.thread_id,
        "domain": req.domain if req.domain in {"academic", "corporate"} else "academic",
        "history": deps.session_store.load(req.thread_id), "events": [],
        "cost_usd": 0, "tokens": 0})
    deps.session_store.append(req.thread_id, "user", req.message)
    deps.session_store.append(req.thread_id, "assistant", result["final"])
    run = get_current_run_tree()
    return _response(result, str(run.id) if run else None)


@app.post("/chat")
async def chat(req: ChatRequest):
    return await _handle(req)


@app.get("/chat/stream")
async def chat_stream(message: str, thread_id: str = "default", domain: str = "academic"):
    async def stream():
        deps = app.state.deps
        state = {"query": message, "thread_id": thread_id,
                 "domain": domain if domain in {"academic", "corporate"} else "academic",
                 "history": deps.session_store.load(thread_id), "events": [],
                 "cost_usd": 0, "tokens": 0}
        seen = 0
        try:
            async for delta in app.state.graph.astream(state, stream_mode="updates"):
                for node, update in delta.items():
                    for event in update.get("events", [])[seen:]:
                        yield "data: " + json.dumps({**event, "node": node}, ensure_ascii=False) + "\n\n"
                    if "events" in update:
                        seen = len(update["events"])
                    state.update(update)
            response = _response(state)
            deps.session_store.append(thread_id, "user", message)
            deps.session_store.append(thread_id, "assistant", response["reply"])
            yield "data: " + json.dumps({"type": "FinalAnswer", **response}, ensure_ascii=False) + "\n\n"
        except Exception as exc:
            yield "data: " + json.dumps({"type": "Error", "message": str(exc)}, ensure_ascii=False) + "\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8100)
