"""
mlangcast_agent/agent.py  —  the compliant Agent Builder + MCP path.

A functional agent powered by **Gemini + Google Cloud Agent Builder (ADK)** that
integrates a partner **MCP server (MongoDB's official `mongodb-mcp-server`)** as a
tool. The agent decides, per event, whether to call the MongoDB MCP tools to fetch
player/team context from `mlangcast.context`, then writes ONE faithful commentary
line in the chosen language.

  * Gemini        — the reasoning model (GEMINI_MODEL).
  * Agent Builder — built with ADK; `adk web` / `adk deploy agent_engine`.
  * Partner MCP   — MongoDB MCP server, called as a tool.

PERFORMANCE: a tool-calling agent is heavy per event (boot the MCP server + connect
to Atlas + 2+ Gemini round-trips ≈ 10-30s). Two ways to call it:
  * generate_commentary(...)   — simple one-off; spins everything up per call.
  * AdkCommentator()           — keeps ONE event loop + runner + MCP connection
                                 WARM, so only the first call pays the boot cost;
                                 later calls are much faster. Use this for the demo.
Both are bounded by a timeout so they never hang silently, and print timing.

SETUP:
    pip install google-adk mcp        # + Node.js on PATH for npx
    # env: MONGODB_URI, GOOGLE_API_KEY, GEMINI_MODEL

RUN / DEPLOY:
    python -m mlangcast_agent.agent          # timed smoke test (persistent path)
    adk web mlangcast_agent                  # chat UI (great for the video)
    adk deploy agent_engine mlangcast_agent  # Vertex AI Agent Engine
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from concurrent.futures import TimeoutError as FutureTimeout

# Load .env for local runs (Cloud Run / Agent Engine inject env vars directly).
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from google.genai import types

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
APP_NAME = "mlangcast"
USER_ID = "demo"

# Read-only MongoDB MCP tools the agent may call.
MONGO_TOOLS = ["find", "aggregate", "count", "list-collections", "list-databases"]

INSTRUCTION = """\
You are MlangCast, a live football commentator. You receive ONE match event plus
the current match state, and you write ONE short, spoken commentary line in the
requested language.

CONTEXT TOOL — you have MongoDB tools. When the player or team in the event is
worth coloring, look them up with ONE `find` on database "mlangcast", collection
"context", matching the `name` field (case-insensitive regex on the player/team
name). Use ONLY the facts you retrieve. Do not call tools more than necessary.

FAITHFULNESS — never invent goals, scores, cards, names, or stats. State the score
only if it is given in the match state. If unsure, leave it out.

STYLE — match energy to the moment. One or two sentences. Output the commentary
line ONLY, natively in the requested language — no labels, no English glosses.
"""


def _mongodb_mcp_toolset() -> McpToolset:
    """The partner MCP integration: MongoDB's official MCP server, as a tool."""
    uri = os.getenv("MONGODB_URI", "")  # empty is import-safe; fails only when used
    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="npx",
                args=["-y", "mongodb-mcp-server@latest", "--readOnly"],
                env={"MDB_MCP_CONNECTION_STRING": uri},
            ),
            timeout=60,
        ),
        tool_filter=MONGO_TOOLS,
    )


# `root_agent` is what the ADK CLI (`adk web` / `adk deploy`) looks for.
root_agent = LlmAgent(
    model=MODEL,
    name="mlangcast_commentator",
    instruction=INSTRUCTION,
    tools=[_mongodb_mcp_toolset()],
)


def _build_user_text(event: dict, state: dict, language: str) -> str:
    return (
        f"Language: {language}\n"
        f"Match state: {json.dumps(state, ensure_ascii=False)}\n"
        f"Event: {json.dumps(event, ensure_ascii=False)[:1500]}\n"
        f"Write ONE faithful commentary line in {language}."
    )


async def _drain(runner, session_id: str, user_text: str) -> str:
    """Run one turn and return the final text."""
    message = types.Content(role="user", parts=[types.Part(text=user_text)])
    final = ""
    async for event in runner.run_async(user_id=USER_ID, session_id=session_id, new_message=message):
        if event.is_final_response() and event.content and event.content.parts:
            final = "".join(p.text or "" for p in event.content.parts)
    return final.strip()


# --------------------------------------------------------------------------- #
# One-off helper (simple; spins up per call). Bounded by a timeout.           #
# --------------------------------------------------------------------------- #
def generate_commentary(event: dict, state: dict, language: str = "en-US", timeout_s: int = 90) -> str:
    """Generate one line via a fresh agent run. Returns "" on timeout/error."""
    async def _run() -> str:
        runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
        session = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
        return await asyncio.wait_for(
            _drain(runner, session.id, _build_user_text(event, state, language)), timeout_s
        )

    t0 = time.time()
    try:
        line = asyncio.run(_run())
    except asyncio.TimeoutError:
        print(f"[adk] timed out after {timeout_s}s (MCP boot + tool loop too slow this run)")
        return ""
    except Exception as exc:
        print(f"[adk] error: {exc}")
        return ""
    print(f"[adk] generated in {time.time() - t0:.1f}s")
    return line


# --------------------------------------------------------------------------- #
# Persistent commentator — keeps the loop + runner + MCP connection WARM.      #
# First call pays the boot cost; later calls reuse it and are much faster.     #
# --------------------------------------------------------------------------- #
class AdkCommentator:
    """
    Reuse one background event loop + runner + MCP connection across calls.

    Usage:
        comm = AdkCommentator(language="es-ES")
        line = comm.generate(event, state)     # first call slow, then fast
        comm.close()
    """

    def __init__(self, language: str = "en-US", timeout_s: int = 90):
        self.language = language
        self.timeout_s = timeout_s
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="adk-loop")
        self._thread.start()
        self._runner = None
        self._session_id = None

    async def _ensure(self) -> None:
        if self._runner is None:
            self._runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
            session = await self._runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
            self._session_id = session.id

    async def _agenerate(self, user_text: str) -> str:
        await self._ensure()
        return await _drain(self._runner, self._session_id, user_text)

    def generate(self, event: dict, state: dict, language: str = None) -> str:
        """Generate one line on the warm agent. Returns "" on timeout/error."""
        user_text = _build_user_text(event, state, language or self.language)
        future = asyncio.run_coroutine_threadsafe(self._agenerate(user_text), self._loop)
        t0 = time.time()
        try:
            line = future.result(timeout=self.timeout_s)
        except FutureTimeout:
            future.cancel()
            print(f"[adk] timed out after {self.timeout_s}s")
            return ""
        except Exception as exc:
            print(f"[adk] error: {exc}")
            return ""
        print(f"[adk] generated in {time.time() - t0:.1f}s")
        return line

    def close(self) -> None:
        """Stop the background loop. Call when done (e.g. on server shutdown)."""
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:
            pass


def make_agent_engine_app():
    """Wrap root_agent for Vertex AI Agent Engine deployment (programmatic path)."""
    from vertexai.preview.reasoning_engines import AdkApp
    return AdkApp(agent=root_agent, enable_tracing=True)


if __name__ == "__main__":
    # Timed smoke test on the PERSISTENT path: two events, so you can see the
    # second call reuse the warm MCP connection and run much faster.
    lang = os.getenv("DEFAULT_LANGUAGE", "es-ES")
    demo = [
        ({"type": {"name": "Shot"}, "minute": 22, "second": 10, "team": {"name": "Argentina"},
          "player": {"name": "Lionel Andrés Messi Cuccittini"},
          "shot": {"outcome": {"name": "Goal"}, "body_part": {"name": "Right Foot"}}},
         {"clock": "22:10", "period": 1, "score": "Argentina 1-0 France"}),
        ({"type": {"name": "Pass"}, "minute": 23, "second": 28, "team": {"name": "France"},
          "player": {"name": "Kylian Mbappé Lottin"}, "pass": {}},
         {"clock": "23:28", "period": 1, "score": "Argentina 1-0 France"}),
    ]
    commentator = AdkCommentator(language=lang)
    try:
        for i, (ev, st) in enumerate(demo, 1):
            print(f"\n--- event {i} ---")
            print("Commentary:", commentator.generate(ev, st) or "(empty — check setup)")
    finally:
        commentator.close()
