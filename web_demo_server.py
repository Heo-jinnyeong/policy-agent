"""Serve the Policy Agent Vue page and proxy API/SSE to port 8100."""
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

BACKEND = "http://127.0.0.1:8100"
HERE = Path(__file__).parent

app = FastAPI()


@app.get("/")
async def index():
    return FileResponse(HERE / "web_chat.html", headers={"Cache-Control": "no-store"})


@app.get("/health")
async def proxy_health():
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{BACKEND}/health")
    return JSONResponse(r.json(), status_code=r.status_code)


@app.post("/chat")
async def proxy_chat(req: Request):
    body = await req.body()
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(f"{BACKEND}/chat", content=body,
                              headers={"content-type": "application/json"})
    return JSONResponse(r.json(), status_code=r.status_code)


async def _form_proxy(method: str, path: str, body: bytes | None = None):
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.request(method, f"{BACKEND}{path}", content=body,
                                        headers={"content-type": "application/json"} if body else None)
    headers = {key: value for key, value in response.headers.items()
               if key.lower() in {"content-disposition", "cache-control",
                                  "x-original-layout-filled", "x-content-type-options"}}
    return Response(content=response.content, status_code=response.status_code,
                    media_type=response.headers.get("content-type"), headers=headers)


@app.get("/forms")
async def proxy_forms(domain: str = "academic"):
    return await _form_proxy("GET", f"/forms?domain={domain}")


@app.post("/forms/upload")
async def proxy_form_upload(req: Request):
    return await _form_proxy("POST", "/forms/upload", await req.body())


@app.get("/forms/{form_id}")
async def proxy_form_detail(form_id: str):
    return await _form_proxy("GET", f"/forms/{form_id}")


@app.get("/forms/{form_id}/original")
async def proxy_form_original(form_id: str):
    return await _form_proxy("GET", f"/forms/{form_id}/original")


@app.post("/forms/{form_id}/draft")
async def proxy_form_draft(form_id: str, req: Request):
    return await _form_proxy("POST", f"/forms/{form_id}/draft", await req.body())


@app.get("/chat/stream")
async def proxy_stream(message: str, thread_id: str = "web-demo", domain: str = "academic",
                       llm_backend: str | None = None):
    async def relay():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "GET", f"{BACKEND}/chat/stream",
                params={"message": message, "thread_id": thread_id, "domain": domain,
                        "llm_backend": llm_backend},
            ) as upstream:
                async for chunk in upstream.aiter_bytes():
                    yield chunk

    return StreamingResponse(relay(), media_type="text/event-stream")


if __name__ == "__main__":
    print("브라우저에서 http://localhost:8600 을 열어보세요 (server.py가 8100에 떠 있어야 합니다)")
    uvicorn.run(app, host="127.0.0.1", port=8600)
