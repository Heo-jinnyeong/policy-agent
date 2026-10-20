"""Persistent stdio MCP client shared by FastAPI requests."""
import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from langsmith import traceable
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_DIR = Path(__file__).parent.parent
DEFAULT_SERVER_SCRIPTS = {"policy": SERVER_DIR / "mcp_server.py"}


def _attr(obj, *names):
    """mcp SDK 버전에 따라 pydantic 필드명이 camelCase(예: resourceTemplates,
    uriTemplate, 구버전 1.x)와 snake_case(예: resource_templates, uri_template,
    신버전 2.x)로 갈린다 — 이 프로젝트 안에도 시스템 python3(mcp 1.28.1, camelCase)와
    .venv(mcp 2.2.0, snake_case)가 섞여 있어 실행 환경마다 바뀐다. 둘 다 시도해서
    설치된 버전과 무관하게 동작하게 한다."""
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(f"{type(obj).__name__!r} has none of {names!r}")


class MCPClient:
    def __init__(self, server_scripts: dict[str, Path] | None = None):
        self._server_scripts = server_scripts or DEFAULT_SERVER_SCRIPTS
        self._stack: AsyncExitStack | None = None
        self.sessions: dict[str, ClientSession] = {}
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        self._stack = AsyncExitStack()
        for server_name, script in self._server_scripts.items():
            params = StdioServerParameters(
                command=sys.executable, args=[str(script)],
                env={**os.environ, "FASTMCP_LOG_LEVEL": "ERROR"})
            read, write = await self._stack.enter_async_context(stdio_client(params))
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.sessions[server_name] = session

    async def stop(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self.sessions = {}

    @traceable(run_type="tool", name="mcp.call_tool")
    async def call_tool(self, name: str, arguments: dict, server: str = "policy") -> dict:
        async with self._lock:
            result = await self.sessions[server].call_tool(name, arguments)
        if _attr(result, "isError", "is_error"):
            raise RuntimeError(result.content[0].text)
        structured = getattr(result, "structuredContent", None) or getattr(result, "structured_content", None)
        if structured:
            return structured
        return json.loads(result.content[0].text)

    @traceable(run_type="tool", name="mcp.list_resource_templates")
    async def list_resource_templates(self, server: str) -> list:
        async with self._lock:
            result = await self.sessions[server].list_resource_templates()
        return _attr(result, "resourceTemplates", "resource_templates")

    @traceable(run_type="tool", name="mcp.find_resource_template")
    async def find_resource_template(self, template_name: str) -> tuple[str, str]:
        """연결된 서버들을 순서대로 조회해 template_name을 제공하는 서버를 찾는다."""
        for server_name in self.sessions:
            templates = await self.list_resource_templates(server_name)
            matched = next((t for t in templates if t.name == template_name), None)
            if matched is not None:
                return server_name, _attr(matched, "uriTemplate", "uri_template")
        raise ValueError(
            f"MCP resource template not found on any connected server: {template_name}")

    @traceable(run_type="tool", name="mcp.read_resource")
    async def read_resource(self, uri: str, server: str) -> str:
        async with self._lock:
            result = await self.sessions[server].read_resource(uri)
        return result.contents[0].text

    @traceable(run_type="tool", name="mcp.list_prompts")
    async def list_prompts(self, server: str) -> list:
        async with self._lock:
            result = await self.sessions[server].list_prompts()
        return result.prompts

    @traceable(run_type="tool", name="mcp.find_prompt")
    async def find_prompt(self, prompt_name: str) -> str:
        """연결된 서버들을 순서대로 조회해 prompt_name을 제공하는 서버를 찾는다."""
        for server_name in self.sessions:
            prompts = await self.list_prompts(server_name)
            if any(p.name == prompt_name for p in prompts):
                return server_name
        raise ValueError(
            f"MCP prompt not found on any connected server: {prompt_name}")

    @traceable(run_type="tool", name="mcp.get_prompt")
    async def get_prompt(self, name: str, arguments: dict, server: str = "ops") -> str:
        async with self._lock:
            result = await self.sessions[server].get_prompt(name, arguments)
        return result.messages[0].content.text
