import asyncio
import json
from types import SimpleNamespace

import httpx

from server import app


class MemoryStore:
    def __init__(self):
        self.messages = []

    def load(self, thread_id):
        return []

    def append(self, thread_id, role, content):
        self.messages.append((thread_id, role, content))


class StubGraph:
    def __init__(self, name):
        self.name = name

    async def ainvoke(self, state):
        return {**state, "final": self.name, "analysis": {"eligibility": "need_more_information"}}

    async def astream(self, state, stream_mode):
        yield {"finalize": {"final": self.name,
                            "analysis": {"eligibility": "need_more_information"},
                            "events": [{"type": "Finalize"}]}}


def test_requests_select_backend_without_changing_server_default():
    store = MemoryStore()
    app.state.default_backend = "ollama"
    app.state.llm_backend = "ollama"
    app.state.instance_id = 1
    app.state.deps_by_backend = {
        name: SimpleNamespace(session_store=store,
                              quality_llm=SimpleNamespace(name=name + ":model"))
        for name in ("ollama", "openai")}
    app.state.graphs = {name: StubGraph(name) for name in ("ollama", "openai")}
    app.state.deps = app.state.deps_by_backend["ollama"]

    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://test") as client:
            health = (await client.get("/health")).json()
            assert set(health["backends"]) == {"ollama", "openai"}
            response = await client.post("/chat", json={"message": "학칙", "thread_id": "a",
                                                        "llm_backend": "openai"})
            assert response.json()["reply"] == "openai"
            assert response.json()["metadata"]["llm_backend"] == "openai"
            stream = await client.get("/chat/stream", params={"message": "학칙",
                                      "thread_id": "b", "llm_backend": "ollama"})
            events = [json.loads(line[6:]) for line in stream.text.splitlines()
                      if line.startswith("data: {")]
            assert next(event for event in events if event["type"] == "FinalAnswer")["metadata"]["llm_backend"] == "ollama"
            assert (await client.post("/chat", json={"message": "학칙",
                                      "llm_backend": "unknown"})).status_code == 400

    asyncio.run(check())


def test_openai_requires_server_key():
    app.state.default_backend = "ollama"
    app.state.deps_by_backend = {"ollama": object()}

    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://test") as client:
            response = await client.post("/chat", json={"message": "학칙",
                                                        "llm_backend": "openai"})
            assert response.status_code == 400
            assert "OPENAI_API_KEY" in response.json()["detail"]

    asyncio.run(check())
