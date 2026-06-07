"""
filesystem_mcp_server.py
========================
MCP-based Filesystem Server for Resume Matching Agent
Implements JSON-RPC 2.0 compliant MCP protocol using the official MCP SDK.

Tools Exposed:
  - list_resumes()         : List all resume files in the resumes directory
  - read_resume()          : Read content of a specific resume file
  - list_job_descriptions(): List all JD files
  - read_job_description() : Read content of a specific JD file
  - save_result()          : Save a match result to the results directory
  - list_results()         : List previously saved results
  - read_result()          : Read a specific result file
  - watch_directory()      : Monitor a directory for new resume files
  - batch_process()        : Process multiple resumes against a JD efficiently

Author: Aswin | MCP Assignment Milestone 3
"""

import os
import json
import asyncio
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

# ─────────────────────────────────────────────
# Server Initialization
# ─────────────────────────────────────────────

mcp = FastMCP(
    name="filesystem-mcp-server",
    instructions="MCP server exposing filesystem tools for resume matching agent. v1.0.0",
)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
RESUMES_DIR = BASE_DIR / "resumes"
JD_DIR = BASE_DIR / "job_descriptions"
RESULTS_DIR = BASE_DIR / "results"

# Ensure directories exist
for d in [RESUMES_DIR, JD_DIR, RESULTS_DIR]:
    d.mkdir(exist_ok=True)


def _safe_read(path: Path) -> str:
    """Read a file safely and return its content."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.name}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path.name}")
    return path.read_text(encoding="utf-8")


def _list_txt_files(directory: Path) -> list[dict]:
    """List all .txt files in a directory with metadata."""
    files = []
    for f in sorted(directory.glob("*.txt")):
        stat = f.stat()
        files.append({
            "filename": f.name,
            "size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })
    return files


# ─────────────────────────────────────────────
# Tool 1: List Resumes
# ─────────────────────────────────────────────

@mcp.tool()
def list_resumes() -> str:
    """
    List all resume files available in the resumes directory.
    Returns filenames, sizes, and last-modified timestamps.
    """
    files = _list_txt_files(RESUMES_DIR)
    if not files:
        return json.dumps({"status": "ok", "count": 0, "resumes": [], "message": "No resumes found."})
    return json.dumps({"status": "ok", "count": len(files), "resumes": files}, indent=2)


# ─────────────────────────────────────────────
# Tool 2: Read Resume
# ─────────────────────────────────────────────

@mcp.tool()
def read_resume(filename: str) -> str:
    """
    Read the content of a specific resume file.

    Args:
        filename: The resume filename (e.g., 'alice_resume.txt')
    """
    try:
        content = _safe_read(RESUMES_DIR / filename)
        return json.dumps({
            "status": "ok",
            "filename": filename,
            "content": content,
            "char_count": len(content),
        }, indent=2)
    except (FileNotFoundError, ValueError) as e:
        return json.dumps({"status": "error", "message": str(e)})


# ─────────────────────────────────────────────
# Tool 3: List Job Descriptions
# ─────────────────────────────────────────────

@mcp.tool()
def list_job_descriptions() -> str:
    """
    List all job description files available in the job_descriptions directory.
    Returns filenames, sizes, and last-modified timestamps.
    """
    files = _list_txt_files(JD_DIR)
    if not files:
        return json.dumps({"status": "ok", "count": 0, "job_descriptions": [], "message": "No JDs found."})
    return json.dumps({"status": "ok", "count": len(files), "job_descriptions": files}, indent=2)


# ─────────────────────────────────────────────
# Tool 4: Read Job Description
# ─────────────────────────────────────────────

@mcp.tool()
def read_job_description(filename: str) -> str:
    """
    Read the content of a specific job description file.

    Args:
        filename: The JD filename (e.g., 'java_backend_jd.txt')
    """
    try:
        content = _safe_read(JD_DIR / filename)
        return json.dumps({
            "status": "ok",
            "filename": filename,
            "content": content,
            "char_count": len(content),
        }, indent=2)
    except (FileNotFoundError, ValueError) as e:
        return json.dumps({"status": "error", "message": str(e)})


# ─────────────────────────────────────────────
# Tool 5: Save Result
# ─────────────────────────────────────────────

@mcp.tool()
def save_result(result_data: str, filename: str = "") -> str:
    """
    Save a match result (JSON string) to the results directory.

    Args:
        result_data: JSON string containing match result data
        filename: Optional custom filename. Auto-generated if not provided.
    """
    try:
        # Validate JSON
        parsed = json.loads(result_data)
    except json.JSONDecodeError:
        return json.dumps({"status": "error", "message": "result_data must be valid JSON"})

    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"result_{timestamp}.json"

    if not filename.endswith(".json"):
        filename += ".json"

    output_path = RESULTS_DIR / filename
    output_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")

    return json.dumps({
        "status": "ok",
        "message": f"Result saved successfully",
        "filename": filename,
        "path": str(output_path),
    }, indent=2)


# ─────────────────────────────────────────────
# Tool 6: List Results
# ─────────────────────────────────────────────

@mcp.tool()
def list_results() -> str:
    """
    List all previously saved match result files in the results directory.
    """
    files = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        stat = f.stat()
        files.append({
            "filename": f.name,
            "size_bytes": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })
    if not files:
        return json.dumps({"status": "ok", "count": 0, "results": [], "message": "No results saved yet."})
    return json.dumps({"status": "ok", "count": len(files), "results": files}, indent=2)


# ─────────────────────────────────────────────
# Tool 7: Read Result
# ─────────────────────────────────────────────

@mcp.tool()
def read_result(filename: str) -> str:
    """
    Read a specific saved match result file.

    Args:
        filename: The result filename (e.g., 'result_20240101_120000.json')
    """
    try:
        content = _safe_read(RESULTS_DIR / filename)
        return json.dumps({
            "status": "ok",
            "filename": filename,
            "content": json.loads(content),
        }, indent=2)
    except (FileNotFoundError, ValueError) as e:
        return json.dumps({"status": "error", "message": str(e)})


# ─────────────────────────────────────────────
# Tool 8: Watch Directory (MCP-Specific)
# ─────────────────────────────────────────────

# In-memory snapshot store for watch_directory (tracks seen files per session)
_watched_snapshots: dict[str, set[str]] = {}


@mcp.tool()
def watch_directory(directory: str = "resumes", reset: bool = False) -> str:
    """
    Monitor a directory for newly added resume files since last check.
    Maintains a per-session snapshot to detect new arrivals.

    Args:
        directory: Which directory to watch — 'resumes', 'job_descriptions', or 'results'
        reset: If True, resets the snapshot so all current files appear as 'new'
    """
    dir_map = {
        "resumes": RESUMES_DIR,
        "job_descriptions": JD_DIR,
        "results": RESULTS_DIR,
    }

    if directory not in dir_map:
        return json.dumps({
            "status": "error",
            "message": f"Invalid directory '{directory}'. Choose from: {list(dir_map.keys())}"
        })

    target_dir = dir_map[directory]
    current_files = {f.name for f in target_dir.glob("*.txt")} | {f.name for f in target_dir.glob("*.json")}

    if reset or directory not in _watched_snapshots:
        _watched_snapshots[directory] = current_files
        return json.dumps({
            "status": "ok",
            "action": "snapshot_initialized",
            "directory": directory,
            "tracked_files": sorted(current_files),
            "message": "Snapshot taken. Future calls will report new files.",
        }, indent=2)

    previous = _watched_snapshots[directory]
    new_files = current_files - previous
    removed_files = previous - current_files

    # Update snapshot
    _watched_snapshots[directory] = current_files

    return json.dumps({
        "status": "ok",
        "directory": directory,
        "checked_at": datetime.now().isoformat(),
        "new_files": sorted(new_files),
        "removed_files": sorted(removed_files),
        "total_files": len(current_files),
        "changes_detected": len(new_files) + len(removed_files) > 0,
    }, indent=2)


# ─────────────────────────────────────────────
# Tool 9: Batch Process (MCP-Specific)
# ─────────────────────────────────────────────

@mcp.tool()
def batch_process(jd_filename: str, resume_filenames: str = "") -> str:
    """
    Read multiple resumes and one job description in a single efficient call.
    Returns all content batched together so the agent can perform matching
    without making separate read calls per file.

    Args:
        jd_filename: The job description filename to match against
        resume_filenames: Comma-separated list of resume filenames.
                          Leave empty to batch ALL resumes in the directory.
    """
    # Load JD
    try:
        jd_content = _safe_read(JD_DIR / jd_filename)
    except (FileNotFoundError, ValueError) as e:
        return json.dumps({"status": "error", "message": f"JD error: {e}"})

    # Resolve resume list
    if resume_filenames.strip():
        filenames = [f.strip() for f in resume_filenames.split(",") if f.strip()]
    else:
        filenames = [f.name for f in sorted(RESUMES_DIR.glob("*.txt"))]

    if not filenames:
        return json.dumps({"status": "error", "message": "No resume files found to process."})

    # Load all resumes
    batch = []
    errors = []
    for fname in filenames:
        try:
            content = _safe_read(RESUMES_DIR / fname)
            checksum = hashlib.md5(content.encode()).hexdigest()[:8]
            batch.append({
                "filename": fname,
                "content": content,
                "char_count": len(content),
                "checksum": checksum,
            })
        except (FileNotFoundError, ValueError) as e:
            errors.append({"filename": fname, "error": str(e)})

    return json.dumps({
        "status": "ok",
        "job_description": {
            "filename": jd_filename,
            "content": jd_content,
        },
        "resumes": batch,
        "resume_count": len(batch),
        "errors": errors,
        "processed_at": datetime.now().isoformat(),
    }, indent=2)


# ─────────────────────────────────────────────
# Resource Discovery Endpoint
# ─────────────────────────────────────────────

@mcp.resource("info://server")
def server_info() -> str:
    """MCP resource discovery endpoint — describes all available tools."""
    return json.dumps({
        "server": "filesystem-mcp-server",
        "version": "1.0.0",
        "description": "Filesystem MCP server for resume-to-job matching",
        "directories": {
            "resumes": str(RESUMES_DIR),
            "job_descriptions": str(JD_DIR),
            "results": str(RESULTS_DIR),
        },
        "tools": [
            {"name": "list_resumes",          "description": "List all resume files"},
            {"name": "read_resume",            "description": "Read a specific resume"},
            {"name": "list_job_descriptions",  "description": "List all JD files"},
            {"name": "read_job_description",   "description": "Read a specific JD"},
            {"name": "save_result",            "description": "Save a match result"},
            {"name": "list_results",           "description": "List saved results"},
            {"name": "read_result",            "description": "Read a saved result"},
            {"name": "watch_directory",        "description": "Detect new files in a directory"},
            {"name": "batch_process",          "description": "Batch-load resumes + JD in one call"},
        ],
    }, indent=2)


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Filesystem MCP Server — Resume Matching Agent")
    print("  Transport : stdio (JSON-RPC 2.0)")
    print(f"  Resumes   : {RESUMES_DIR}")
    print(f"  JDs       : {JD_DIR}")
    print(f"  Results   : {RESULTS_DIR}")
    print("=" * 55)
    mcp.run(transport="stdio")
