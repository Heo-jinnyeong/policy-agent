"""Policy Agent API with persistent conversation and validated SSE responses."""
import base64
import binascii
import json
import os
import zipfile
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from pydantic import BaseModel, Field

from agent.deps import SupervisorDeps
from agent.fake_llm import FastLLM, QualityLLM
from agent.graph import build_graph
from agent.mcp_client import MCPClient
from agent.policy_llm import PolicyLLM
from agent.session_store import SqliteSessionStore
from application_forms.catalog import (EXTENSIONS, FORMS_DIR, MAX_FORM_BYTES,
                                       describe_form, form_path, list_forms)
from application_forms.draft import build_draft

CREATION_COUNT = 0


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    thread_id: str = "default"
    domain: str = "academic"
    llm_backend: str | None = None


class FormUpload(BaseModel):
    filename: str
    domain: str = "academic"
    content_base64: str


class DraftRequest(BaseModel):
    values: dict[str, str]


def _build_llms(backend: str | None = None):
    backend = backend or os.getenv("LLM_BACKEND", "fake")
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
    session_store = SqliteSessionStore()
    configured = ["ollama"]
    if os.getenv("OPENAI_API_KEY"):
        configured.append("openai")
    default_backend = os.getenv("LLM_BACKEND", "ollama")
    if default_backend == "fake":
        configured.append("fake")
    if default_backend not in configured:
        default_backend = "ollama"
    deps_by_backend = {}
    graphs = {}
    for backend in configured:
        _, fast_llm, quality_llm = _build_llms(backend)
        deps = SupervisorDeps(fast_llm=fast_llm, quality_llm=quality_llm, mcp=mcp,
                              session_store=session_store, version="policy-v2",
                              policy_llm=PolicyLLM(quality_llm))
        deps_by_backend[backend] = deps
        graphs[backend] = build_graph(deps)
    application.state.deps_by_backend = deps_by_backend
    application.state.graphs = graphs
    application.state.default_backend = default_backend
    application.state.deps = deps_by_backend[default_backend]
    application.state.graph = graphs[default_backend]
    application.state.llm_backend = default_backend
    application.state.instance_id = id(session_store)
    try:
        yield
    finally:
        await mcp.stop()
        session_store.close()


app = FastAPI(title="Policy Agent", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "instance_id": app.state.instance_id,
            "creation_count": CREATION_COUNT, "llm_backend": app.state.llm_backend,
            "model": app.state.deps.quality_llm.name,
            "backends": {name: {"model": deps.quality_llm.name}
                         for name, deps in app.state.deps_by_backend.items()}}


@app.get("/version")
async def version():
    return {"version": "policy-v2"}


def _response(result: dict, run_id=None, backend=None, model=None) -> dict:
    evidence = result.get("evidence", [])
    analysis = result.get("analysis", {})
    cited = set()
    for field in ("reason_support", "deadline", "department", "related_context"):
        item = analysis.get(field, {})
        if isinstance(item, dict) and item.get("citation_id"):
            cited.add(item["citation_id"])
    for field in ("conditions", "documents", "procedure"):
        for item in analysis.get(field, []):
            if isinstance(item, dict) and item.get("citation_id"):
                cited.add(item["citation_id"])
    return {"reply": result.get("final", ""), "analysis": result.get("analysis", {}),
            "forms": result.get("forms", []),
            "sources": [{k: e.get(k) for k in ("id", "document_id", "document_name", "article",
                                              "page", "effective_date", "source", "text")}
                        for e in evidence if e.get("id") in cited],
            "events": result.get("events", []),
            "metadata": {"domain": result.get("domain"), "route": result.get("route"),
                         "question_type": result.get("question_type", "explanation"),
                         "llm_backend": backend, "model": model,
                         "research_rounds": result.get("research_rounds", 0),
                         "validation_attempts": result.get("validation_attempts", 0),
                         "tokens": result.get("tokens", 0),
                         "estimated_cost_usd": result.get("cost_usd", 0),
                         "langsmith_run_id": run_id}}


@traceable(name="policy_chat_request")
async def _handle(req: ChatRequest):
    backend = _select_backend(req.llm_backend)
    deps = app.state.deps_by_backend[backend]
    result = await app.state.graphs[backend].ainvoke({"query": req.message, "thread_id": req.thread_id,
        "domain": req.domain if req.domain in {"academic", "corporate"} else "academic",
        "history": deps.session_store.load(req.thread_id), "events": [],
        "cost_usd": 0, "tokens": 0})
    deps.session_store.append(req.thread_id, "user", req.message)
    deps.session_store.append(req.thread_id, "assistant", result["final"])
    run = get_current_run_tree()
    return _response(result, str(run.id) if run else None, backend, deps.quality_llm.name)


def _select_backend(requested: str | None) -> str:
    backend = requested or app.state.default_backend
    if backend not in app.state.deps_by_backend:
        detail = ("OpenAI를 사용하려면 서버의 .env에 OPENAI_API_KEY를 설정하고 재시작하세요."
                  if backend == "openai" else "지원하지 않는 LLM입니다.")
        raise HTTPException(status_code=400, detail=detail)
    return backend


@app.post("/chat")
async def chat(req: ChatRequest):
    return await _handle(req)


@app.get("/forms")
async def forms_list(domain: str = "academic"):
    try:
        return {"forms": list_forms(domain, FORMS_DIR)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/forms/upload")
async def forms_upload(req: FormUpload):
    """Store a blank form; applicant values use the separate draft endpoint."""
    if req.domain not in {"academic", "corporate"}:
        raise HTTPException(status_code=400, detail="지원하지 않는 분야입니다.")
    name = req.filename.replace("\\", "/").split("/")[-1]
    if not name or name.startswith(".") or Path(name).suffix.lower() not in EXTENSIONS:
        raise HTTPException(status_code=400, detail="HWP, HWPX, DOCX 양식만 등록할 수 있습니다.")
    if len(req.content_base64) > MAX_FORM_BYTES * 4 // 3 + 8:
        raise HTTPException(status_code=413, detail="양식은 10MB 이하만 등록할 수 있습니다.")
    try:
        content = base64.b64decode(req.content_base64, validate=True)
    except binascii.Error as exc:
        raise HTTPException(status_code=400, detail="파일 데이터가 올바르지 않습니다.") from exc
    if len(content) > MAX_FORM_BYTES:
        raise HTTPException(status_code=413, detail="양식은 10MB 이하만 등록할 수 있습니다.")
    ext = Path(name).suffix.lower()
    if ext == ".hwp" and not content.startswith(bytes.fromhex("d0cf11e0a1b11ae1")):
        raise HTTPException(status_code=400, detail="올바른 HWP 파일이 아닙니다.")
    if ext in {".hwpx", ".docx"}:
        try:
            with zipfile.ZipFile(BytesIO(content)) as archive:
                if sum(item.file_size for item in archive.infolist()) > 60 * 1024 * 1024:
                    raise HTTPException(status_code=413, detail="압축 해제 크기가 너무 큽니다.")
        except zipfile.BadZipFile as exc:
            raise HTTPException(status_code=400, detail="올바른 문서 파일이 아닙니다.") from exc
    directory = FORMS_DIR / req.domain
    directory.mkdir(parents=True, exist_ok=True)
    stem = Path(name).stem
    target = directory / name
    index = 2
    while target.exists():
        target = directory / f"{stem}-{index}{ext}"
        index += 1
    target.write_bytes(content)
    try:
        form = describe_form(target, FORMS_DIR)
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="양식의 내용을 읽을 수 없습니다.") from exc
    return {"form": form}


@app.get("/forms/{form_id}")
async def form_detail(form_id: str):
    try:
        return {"form": describe_form(form_path(form_id, FORMS_DIR), FORMS_DIR)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="양식을 찾을 수 없습니다.") from exc


@app.get("/forms/{form_id}/original")
async def form_original(form_id: str):
    try:
        path = form_path(form_id, FORMS_DIR)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="양식을 찾을 수 없습니다.") from exc
    return FileResponse(path, filename=path.name, media_type="application/octet-stream",
                        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})


@app.post("/forms/{form_id}/draft")
async def form_draft(form_id: str, req: DraftRequest):
    """Applicant values stay in memory and are never added to chat history or traces."""
    try:
        path = form_path(form_id, FORMS_DIR)
        content, extension, filled_original = build_draft(path, req.values, FORMS_DIR)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="양식을 찾을 수 없습니다.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    media_type = ("application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                  if extension == "docx" else "application/octet-stream")
    return Response(content=content, media_type=media_type,
                    headers={"Content-Disposition": f'attachment; filename="application-draft.{extension}"',
                             "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
                             "X-Original-Layout-Filled": str(filled_original).lower()})


@app.get("/chat/stream")
async def chat_stream(message: str, thread_id: str = "default", domain: str = "academic",
                      llm_backend: str | None = None):
    backend = _select_backend(llm_backend)
    async def stream():
        deps = app.state.deps_by_backend[backend]
        state = {"query": message, "thread_id": thread_id,
                 "domain": domain if domain in {"academic", "corporate"} else "academic",
                 "history": deps.session_store.load(thread_id), "events": [],
                 "cost_usd": 0, "tokens": 0}
        seen = 0
        try:
            async for delta in app.state.graphs[backend].astream(state, stream_mode="updates"):
                for node, update in delta.items():
                    for event in update.get("events", [])[seen:]:
                        yield "data: " + json.dumps({**event, "node": node}, ensure_ascii=False) + "\n\n"
                    if "events" in update:
                        seen = len(update["events"])
                    state.update(update)
            response = _response(state, backend=backend, model=deps.quality_llm.name)
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
