"""MlangCast ADK agent package (Agent Builder + MongoDB MCP).

`root_agent` is exported here so the ADK CLI can find it:
    adk web mlangcast_agent
    adk deploy agent_engine mlangcast_agent
"""
from .agent import root_agent  # noqa: F401
