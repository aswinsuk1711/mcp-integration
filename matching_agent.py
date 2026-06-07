"""
matching_agent.py
=================
LangGraph Resume Matching Agent — MCP Client Edition
Replaces all direct filesystem tools with MCP server calls via stdio transport.

Architecture:
  Claude (LLM) ←→ LangGraph StateGraph ←→ MCP Client ←→ filesystem_mcp_server.py

Graph Nodes:
  1. initialize        — Set up session, discover server resources
  2. fetch_job         — Load job description via MCP
  3. fetch_resumes     — Batch-load all resumes via MCP (batch_process tool)
  4. analyze_match     — LLM scores each resume against JD
  5. rank_candidates   — Sort and rank by match score
  6. save_results      — Persist ranked results via MCP
  7. watch_new         — Check for newly added resumes (watch_directory)
  8. report            — Generate final human-readable summary
  9. end               — Terminal node

Author: Aswin | MCP Assignment Milestone 3
"""

import os
import json
import asyncio
import sys
from dotenv import load_dotenv

load_dotenv()
from datetime import datetime
from typing import TypedDict, Annotated, Any
import operator

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MCP_SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "filesystem_mcp_server.py")


def get_llm():
    """Create ChatAnthropic instance if API key is available."""
    key = ANTHROPIC_API_KEY
    if not key:
        return None
    return ChatAnthropic(model="claude-3-haiku-20240307", api_key=key, max_tokens=1024)


def mock_llm_score(jd: str, resume: str) -> dict:
    """Fallback mock LLM that provides basic keyword-based scoring."""
    jd_lower = jd.lower()
    resume_lower = resume.lower()

    java_score = 30 if "java" in resume_lower else 0
    spring_score = 15 if "spring" in resume_lower else 0
    api_score = 10 if "rest" in resume_lower or "api" in resume_lower else 0
    db_score = 15 if any(x in resume_lower for x in ["sql", "postgresql", "mysql", "database"]) else 0
    docker_score = 10 if "docker" in resume_lower else 0
    aws_score = 10 if any(x in resume_lower for x in ["aws", "cloud", "microservices"]) else 0
    test_score = 10 if any(x in resume_lower for x in ["junit", "test", "testing"]) else 0

    total = min(100, java_score + spring_score + api_score + db_score + docker_score + aws_score + test_score)

    matched = []
    if "java" in resume_lower:
        matched.append("Java")
    if "spring" in resume_lower:
        matched.append("Spring Boot")
    if "rest" in resume_lower or "api" in resume_lower:
        matched.append("REST APIs")
    if any(x in resume_lower for x in ["sql", "postgresql", "mysql"]):
        matched.append("SQL/Databases")
    if "docker" in resume_lower:
        matched.append("Docker")

    missing = []
    if "java" not in jd_lower:
        missing.append("No specific skills required")
    if "spring" not in resume_lower and "spring" in jd_lower:
        missing.append("Spring Boot")
    if "microservices" in jd_lower and "microservices" not in resume_lower:
        missing.append("Microservices experience")
    if "aws" in jd_lower and not any(x in resume_lower for x in ["aws", "cloud", "gcp", "azure"]):
        missing.append("Cloud platform experience")

    rec = "shortlist" if total >= 70 else ("consider" if total >= 40 else "reject")
    exp_fit = "strong" if total >= 60 else ("moderate" if total >= 30 else "weak")

    return {
        "score": total,
        "matched_skills": matched,
        "missing_skills": missing,
        "experience_fit": exp_fit,
        "reasoning": f"Mock analysis: Found {len(matched)} relevant skills, matches key requirements.",
        "recommendation": rec,
    }


# ─────────────────────────────────────────────
# Agent State Schema
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    # Session
    session_id: str
    jd_filename: str
    status: str

    # Data loaded via MCP
    job_description: str
    resumes: list[dict]          # [{"filename": ..., "content": ...}]
    available_jds: list[str]

    # Analysis outputs
    match_results: list[dict]    # [{"filename", "score", "reasoning", "recommendation"}]
    ranked_candidates: list[dict]
    saved_result_file: str

    # Watch state
    new_files_detected: list[str]

    # Conversation log
    messages: Annotated[list, operator.add]
    final_report: str
    error: str


# ─────────────────────────────────────────────
# MCP Client Helper
# ─────────────────────────────────────────────

async def call_mcp_tool(tool_name: str, arguments: dict = {}) -> str:
    """
    Connect to the MCP server via stdio, call a tool, and return the result.
    Opens a fresh connection per call (stateless; suitable for demo).
    """
    server_params = StdioServerParameters(
        command="python3",
        args=[MCP_SERVER_SCRIPT],
        env=None,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)
            # Result content is a list of TextContent blocks
            if result.content:
                return result.content[0].text
            return json.dumps({"status": "error", "message": "Empty response from MCP server"})


def run_mcp(tool_name: str, arguments: dict = {}) -> dict:
    """Synchronous wrapper around call_mcp_tool for use inside LangGraph nodes."""
    raw = asyncio.run(call_mcp_tool(tool_name, arguments))
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"status": "error", "message": raw}


# ─────────────────────────────────────────────
# Node 1: Initialize
# ─────────────────────────────────────────────

def node_initialize(state: AgentState) -> AgentState:
    """Discover MCP server resources and list available JDs."""
    print("\n[Node 1/8] Initializing MCP session...")

    jd_list = run_mcp("list_job_descriptions")
    available_jds = [f["filename"] for f in jd_list.get("job_descriptions", [])]

    print(f"  [OK] MCP server connected")
    print(f"  [OK] Available JDs: {available_jds}")

    return {
        **state,
        "available_jds": available_jds,
        "status": "initialized",
        "messages": [SystemMessage(content=(
            "You are a professional HR recruiter AI. "
            "Analyze resumes against job descriptions objectively. "
            "Score candidates 0-100 based on skill match, experience, and role fit."
        ))],
    }


# ─────────────────────────────────────────────
# Node 2: Fetch Job Description
# ─────────────────────────────────────────────

def node_fetch_job(state: AgentState) -> AgentState:
    """Load the target job description from the MCP server."""
    jd_filename = state.get("jd_filename", "")
    print(f"\n[Node 2/8] Fetching job description: {jd_filename}")

    if not jd_filename:
        available = state.get("available_jds", [])
        if available:
            jd_filename = available[0]
            print(f"  ℹ No JD specified. Defaulting to: {jd_filename}")
        else:
            return {**state, "error": "No job description files found in MCP server.", "status": "error"}

    result = run_mcp("read_job_description", {"filename": jd_filename})

    if result.get("status") != "ok":
        return {**state, "error": result.get("message", "Failed to load JD"), "status": "error"}

    print(f"  [OK] JD loaded ({result['char_count']} chars)")
    return {
        **state,
        "jd_filename": jd_filename,
        "job_description": result["content"],
        "status": "job_fetched",
    }


# ─────────────────────────────────────────────
# Node 3: Fetch Resumes (Batch via MCP)
# ─────────────────────────────────────────────

def node_fetch_resumes(state: AgentState) -> AgentState:
    """
    Use MCP batch_process tool to load all resumes + JD in one efficient call.
    This demonstrates MCP-specific optimization vs individual file reads.
    """
    print(f"\n[Node 3/8] Batch-fetching resumes via MCP batch_process...")

    result = run_mcp("batch_process", {
        "jd_filename": state["jd_filename"],
        "resume_filenames": "",  # empty = load all
    })

    if result.get("status") != "ok":
        return {**state, "error": result.get("message", "Batch process failed"), "status": "error"}

    resumes = result.get("resumes", [])
    print(f"  [OK] Loaded {len(resumes)} resumes in single MCP call")
    for r in resumes:
        print(f"    - {r['filename']} ({r['char_count']} chars)")

    return {
        **state,
        "resumes": resumes,
        "status": "resumes_fetched",
    }


# ─────────────────────────────────────────────
# Node 4: Analyze Match (LLM)
# ─────────────────────────────────────────────

def node_analyze_match(state: AgentState) -> AgentState:
    """LLM scores each resume against the job description."""
    print(f"\n[Node 4/8] Analyzing {len(state['resumes'])} resumes with LLM...")

    jd = state["job_description"]
    match_results = []

    for resume in state["resumes"]:
        prompt = f"""
You are an expert HR recruiter. Score the following resume against the job description.

--- JOB DESCRIPTION ---
{jd}

--- RESUME ---
{resume['content']}

Respond ONLY with a valid JSON object (no markdown, no explanation outside JSON):
{{
  "score": <integer 0-100>,
  "matched_skills": [<list of matching skills>],
  "missing_skills": [<list of required skills not found>],
  "experience_fit": "<strong|moderate|weak>",
  "reasoning": "<2-3 sentence summary of fit>",
  "recommendation": "<shortlist|consider|reject>"
}}
"""
        messages = [HumanMessage(content=prompt)]
        text = ""
        try:
            llm = get_llm()
            if llm is None:
                raise ValueError("No API key — using mock scoring")
            response = llm.invoke(messages)
            text = response.content.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            analysis = json.loads(text.strip())
        except Exception as e:
            # Fallback to mock scoring if LLM fails (e.g., no credits or no key)
            if not text or "credit" in str(e).lower() or "api key" in str(e).lower():
                analysis = mock_llm_score(jd, resume["content"])
            else:
                analysis = {
                    "score": 0,
                    "matched_skills": [],
                    "missing_skills": [],
                    "experience_fit": "unknown",
                    "reasoning": f"LLM error or failed to parse: {str(e)[:60]}",
                    "recommendation": "consider",
                }

        match_results.append({
            "filename": resume["filename"],
            **analysis,
        })
        print(f"  [OK] {resume['filename']} -> Score: {analysis.get('score', 'N/A')} | {analysis.get('recommendation', 'N/A')}")

    return {**state, "match_results": match_results, "status": "analyzed"}


# ─────────────────────────────────────────────
# Node 5: Rank Candidates
# ─────────────────────────────────────────────

def node_rank_candidates(state: AgentState) -> AgentState:
    """Sort candidates by match score descending."""
    print(f"\n[Node 5/8] Ranking candidates...")

    ranked = sorted(
        state["match_results"],
        key=lambda x: x.get("score", 0),
        reverse=True,
    )

    for i, c in enumerate(ranked, 1):
        print(f"  #{i} {c['filename']} — Score: {c['score']} ({c['recommendation']})")

    return {**state, "ranked_candidates": ranked, "status": "ranked"}


# ─────────────────────────────────────────────
# Node 6: Save Results (via MCP)
# ─────────────────────────────────────────────

def node_save_results(state: AgentState) -> AgentState:
    """Persist ranked results to disk using the MCP save_result tool."""
    print(f"\n[Node 6/8] Saving results via MCP...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"match_result_{state['session_id']}_{timestamp}.json"

    payload = {
        "session_id": state["session_id"],
        "job_description_file": state["jd_filename"],
        "processed_at": datetime.now().isoformat(),
        "total_candidates": len(state["ranked_candidates"]),
        "ranked_candidates": state["ranked_candidates"],
    }

    result = run_mcp("save_result", {
        "result_data": json.dumps(payload),
        "filename": filename,
    })

    if result.get("status") != "ok":
        print(f"  [FAIL] Save failed: {result.get('message')}")
        return {**state, "saved_result_file": "", "status": "save_failed"}

    print(f"  [OK] Results saved: {filename}")
    return {**state, "saved_result_file": filename, "status": "saved"}


# ─────────────────────────────────────────────
# Node 7: Watch for New Resumes (MCP-Specific)
# ─────────────────────────────────────────────

def node_watch_new(state: AgentState) -> AgentState:
    """
    Use MCP watch_directory to detect any newly added resume files
    since the agent last checked. Demonstrates real-time monitoring capability.
    """
    print(f"\n[Node 7/8] Checking for new resumes via watch_directory...")

    result = run_mcp("watch_directory", {"directory": "resumes", "reset": False})
    new_files = result.get("new_files", [])

    if new_files:
        print(f"  [NEW] New resumes detected: {new_files}")
    else:
        print(f"  [OK] No new resumes detected (total: {result.get('total_files', 0)})")

    return {**state, "new_files_detected": new_files, "status": "watched"}


# ─────────────────────────────────────────────
# Node 8: Generate Report
# ─────────────────────────────────────────────

def node_report(state: AgentState) -> AgentState:
    """Generate a human-readable final summary report."""
    print(f"\n[Node 8/8] Generating final report...")

    ranked = state.get("ranked_candidates", [])
    jd_file = state.get("jd_filename", "N/A")
    saved = state.get("saved_result_file", "N/A")
    new_files = state.get("new_files_detected", [])

    lines = [
        "=" * 60,
        "  RESUME MATCHING AGENT — FINAL REPORT",
        f"  Session   : {state.get('session_id', 'N/A')}",
        f"  Job       : {jd_file}",
        f"  Evaluated : {len(ranked)} candidates",
        f"  Saved to  : {saved}",
        "=" * 60,
        "",
        "CANDIDATE RANKINGS:",
        "-" * 60,
    ]

    for i, c in enumerate(ranked, 1):
        lines += [
            f"#{i}  {c['filename']}",
            f"    Score          : {c.get('score', 'N/A')}/100",
            f"    Experience Fit : {c.get('experience_fit', 'N/A')}",
            f"    Recommendation : {c.get('recommendation', 'N/A').upper()}",
            f"    Matched Skills : {', '.join(c.get('matched_skills', [])) or 'None'}",
            f"    Missing Skills : {', '.join(c.get('missing_skills', [])) or 'None'}",
            f"    Reasoning      : {c.get('reasoning', '')}",
            "",
        ]

    if new_files:
        lines += [
            "NEW RESUMES DETECTED (not yet processed):",
            "-" * 60,
            *[f"  • {f}" for f in new_files],
            "",
        ]

    lines += ["=" * 60]
    report = "\n".join(lines)
    print(report)

    return {**state, "final_report": report, "status": "complete"}


# ─────────────────────────────────────────────
# Error Router
# ─────────────────────────────────────────────

def route_after_init(state: AgentState) -> str:
    return "error_end" if state.get("status") == "error" else "fetch_job"

def route_after_job(state: AgentState) -> str:
    return "error_end" if state.get("status") == "error" else "fetch_resumes"

def route_after_resumes(state: AgentState) -> str:
    return "error_end" if state.get("status") == "error" else "analyze_match"


# ─────────────────────────────────────────────
# Build Graph
# ─────────────────────────────────────────────

def build_graph() -> Any:
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("initialize",      node_initialize)
    graph.add_node("fetch_job",       node_fetch_job)
    graph.add_node("fetch_resumes",   node_fetch_resumes)
    graph.add_node("analyze_match",   node_analyze_match)
    graph.add_node("rank_candidates", node_rank_candidates)
    graph.add_node("save_results",    node_save_results)
    graph.add_node("watch_new",       node_watch_new)
    graph.add_node("report",          node_report)

    # Entry point
    graph.set_entry_point("initialize")

    # Conditional edges (with error routing)
    graph.add_conditional_edges("initialize",    route_after_init,    {"fetch_job": "fetch_job", "error_end": END})
    graph.add_conditional_edges("fetch_job",     route_after_job,     {"fetch_resumes": "fetch_resumes", "error_end": END})
    graph.add_conditional_edges("fetch_resumes", route_after_resumes, {"analyze_match": "analyze_match", "error_end": END})

    # Linear edges
    graph.add_edge("analyze_match",   "rank_candidates")
    graph.add_edge("rank_candidates", "save_results")
    graph.add_edge("save_results",    "watch_new")
    graph.add_edge("watch_new",       "report")
    graph.add_edge("report",          END)

    return graph.compile()


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

def run_agent(jd_filename: str = "", session_id: str = ""):
    """Run the matching agent for a given JD file."""
    import uuid
    if not session_id:
        session_id = str(uuid.uuid4())[:8]

    print(f"\n{'='*60}")
    print(f"  Resume Matching Agent (MCP Edition)")
    print(f"  Session: {session_id}")
    print(f"{'='*60}")

    # Initialize watch_directory snapshot before run
    asyncio.run(call_mcp_tool("watch_directory", {"directory": "resumes", "reset": True}))

    initial_state: AgentState = {
        "session_id": session_id,
        "jd_filename": jd_filename,
        "status": "start",
        "job_description": "",
        "resumes": [],
        "available_jds": [],
        "match_results": [],
        "ranked_candidates": [],
        "saved_result_file": "",
        "new_files_detected": [],
        "messages": [],
        "final_report": "",
        "error": "",
    }

    graph = build_graph()
    final_state = graph.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    jd = sys.argv[1] if len(sys.argv) > 1 else "java_backend_jd.txt"
    print(f"\nRunning agent for JD: {jd}")
    print("(Set ANTHROPIC_API_KEY env var for LLM analysis)\n")

    if not ANTHROPIC_API_KEY:
        print("⚠ WARNING: ANTHROPIC_API_KEY not set. LLM nodes will fail.")
        print("  Export your key: export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    result = run_agent(jd_filename=jd)
    sys.exit(0 if result.get("status") == "complete" else 1)
