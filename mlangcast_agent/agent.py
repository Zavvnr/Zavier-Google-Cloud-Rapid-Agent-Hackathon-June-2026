"""
mlangcast_agent/agent.py  —  the compliant Agent Builder + MCP path.

This is the piece the hackathon requires: a functional agent **powered by Gemini
and Google Cloud Agent Builder (via the Agent Development Kit)** that **integrates a
partner MCP server (MongoDB's official MCP server)** as a tool.

How it satisfies the three requirements:
  * Gemini            — the agent's reasoning model is a Gemini model (GEMINI_MODEL).
  * Agent Builder     — built with ADK (`google.adk`); runnable with `adk web` /
                        deployable to Vertex AI Agent Engine (`adk deploy agent_engine`).
  * Partner MCP server— the agent's tool is MongoDB's MCP server (`mongodb-mcp-server`),
                        which the agent calls to look up player/team context from your
                        `mlangcast.context` collection while writing each line.

The agent decides, per event, whether to call the MongoDB MCP tools to fetch
context, then writes ONE faithful commentary line in the chosen language.

────────────────────────────────────────────────────────────────────────────
SETUP (run once):
    pip install google-adk
    # MCP server runs via npx, so you need Node.js available on PATH:
    #   https://nodejs.org   (then `npx -y mongodb-mcp-server@latest --version`)
    # Env: MONGODB_URI, GOOGLE_API_KEY (or Vertex creds), GEMINI_MODEL

SMOKE TEST (one event through the real agent — calls MCP + Gemini):
    python -m mlangcast_agent.agent

INTERACTIVE / DEPLOY (ADK CLI):
    adk web mlangcast_agent                 # local chat UI
    adk deploy agent_engine mlangcast_agent # push to Vertex AI Agent Engine
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import asyncio
import json
import os

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

# Read-only MongoDB MCP tools the agent is allowed to call.
MONGO_TOOLS = ["find", "aggregate", "count", "list-collections", "list-databases"]

INSTRUCTION = """\
You are MlangCast, a live football commentator. You receive ONE match event plus
the current match state, and you write ONE short, spoken commentary line in the
requested language.

CONTEXT TOOL — you have MongoDB tools. When the player or team in the event is
worth coloring (a shot, goal, key moment, or a quiet build-up around a notable
player), look them up with a `find` on database "mlangcast", collection "context",
matching the `name` field (a case-insensitive regex on the player or team name).
Use ONLY the facts you retrieve.

FAITHFULNESS — never invent goals, scores, cards, names, or stats. State the score
only if it is given in the match state. If you are unsure, leave it out.

STYLE — match energy to the moment (calm in midfield, loud on a goal). One or two
sentences. Output the commentary line ONLY, natively in the requested language —
no labels, no English glosses, no stage directions.
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


# `root_agent` is the name the ADK CLI (`adk web` / `adk deploy`) looks for.
root_agent = LlmAgent(
    model=MODEL,
    name="mlangcast_commentator",
    instruction=INSTRUCTION,
    tools=[_mongodb_mcp_toolset()],
)


# --------------------------------------------------------------------------- #
# Programmatic use — call the agent per event from the Python pipeline.        #
# --------------------------------------------------------------------------- #
async def _run_once(user_text: str) -> str:
    """Run a single turn through the agent (sets up + tears down the MCP server)."""
    runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
    session = await runner.session_service.create_session(app_name=APP_NAME, user_id="demo")
    message = types.Content(role="user", parts=[types.Part(text=user_text)])
    final = ""
    async for event in runner.run_async(user_id="demo", session_id=session.id, new_message=message):
        if event.is_final_response() and event.content and event.content.parts:
            final = "".join(p.text or "" for p in event.content.parts)
    return final.strip()


def generate_commentary(event: dict, state: dict, language: str = "en-US") -> str:
    """
    Generate one commentary line for an event via the ADK agent (which may call the
    MongoDB MCP server for context). Synchronous wrapper around the async runner.

    Drop-in for the pipeline: pass `lambda ev, st: generate_commentary(ev, st, lang)`
    where your current code calls the direct-Gemini generator.
    """
    user_text = (
        f"Language: {language}\n"
        f"Match state: {json.dumps(state, ensure_ascii=False)}\n"
        f"Event: {json.dumps(event, ensure_ascii=False)[:1500]}\n"
        f"Write ONE faithful commentary line in {language}."
    )
    try:
        return asyncio.run(_run_once(user_text))
    except Exception as exc:  # keep the live loop alive (same ethos as the rest)
        print(f"[adk_agent] generation error: {exc}")
        return ""


def make_agent_engine_app():
    """Wrap root_agent for Vertex AI Agent Engine deployment (programmatic path)."""
    from vertexai.preview.reasoning_engines import AdkApp
    return AdkApp(agent=root_agent, enable_tracing=True)


if __name__ == "__main__":
    # One real event through the agent. Needs google-adk + Node (npx) + MONGODB_URI
    # + GOOGLE_API_KEY. Prints the generated line (and the agent will have called the
    # MongoDB MCP server for context if it judged it useful).
    sample_event = {
        "type": {"name": "Shot"}, "minute": 22, "second": 10,
        "team": {"name": "Argentina"}, "player": {"name": "Lionel Andrés Messi Cuccittini"},
        "shot": {"outcome": {"name": "Goal"}, "body_part": {"name": "Right Foot"}},
    }
    sample_state = {"clock": "22:10", "period": 1, "score": "Argentina 1-0 France"}
    line = generate_commentary(sample_event, sample_state, language=os.getenv("DEFAULT_LANGUAGE", "es-ES"))
    print("Commentary:", line or "(empty — check ADK/Node/Mongo/Gemini setup)")
