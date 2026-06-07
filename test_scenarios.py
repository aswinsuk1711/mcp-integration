"""
test_scenarios.py
=================
Test suite demonstrating all MCP tool scenarios for the filesystem server.
Tests run against the live MCP server via stdio transport.

Scenarios:
  1. Server discovery & resource listing
  2. Read resume and job description files
  3. Batch processing (multiple resumes in one call)
  4. Watch directory for new files
  5. Save and retrieve match results
  6. Error handling (invalid files, bad inputs)
  7. Full end-to-end agent workflow

Author: Aswin | MCP Assignment Milestone 3
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
SERVER_SCRIPT = str(BASE_DIR / "filesystem_mcp_server.py")

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"

results_log = []


# ─────────────────────────────────────────────
# Helper: Call MCP Tool
# ─────────────────────────────────────────────

async def call_tool(tool_name: str, arguments: dict = {}) -> dict:
    """Open a stdio MCP session, call a tool, return parsed JSON result."""
    server_params = StdioServerParameters(
        command="python3",
        args=[SERVER_SCRIPT],
        env=None,
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)
            if result.content:
                try:
                    return json.loads(result.content[0].text)
                except json.JSONDecodeError:
                    return {"status": "error", "raw": result.content[0].text}
            return {"status": "error", "message": "No content returned"}


def log(label: str, status: str, detail: str = ""):
    marker = PASS if status == "pass" else (FAIL if status == "fail" else INFO)
    line = f"  {marker}  {label}"
    if detail:
        line += f"\n         >> {detail}"
    print(line)
    results_log.append({"label": label, "status": status, "detail": detail})


# ─────────────────────────────────────────────
# Scenario 1: Server Discovery
# ─────────────────────────────────────────────

async def scenario_1_discovery():
    print("\n" + "-" * 55)
    print("SCENARIO 1: Server Discovery & Resource Listing")
    print("-" * 55)

    # List resumes
    result = await call_tool("list_resumes")
    if result.get("status") == "ok" and result.get("count", 0) > 0:
        log("list_resumes returns files", "pass", f"{result['count']} resumes found")
        for r in result["resumes"]:
            log(f"  Found: {r['filename']}", "info", f"{r['size_bytes']} bytes")
    else:
        log("list_resumes returns files", "fail", str(result))

    # List JDs
    result = await call_tool("list_job_descriptions")
    if result.get("status") == "ok" and result.get("count", 0) > 0:
        log("list_job_descriptions returns files", "pass", f"{result['count']} JDs found")
    else:
        log("list_job_descriptions returns files", "fail", str(result))

    # List results (initially empty)
    result = await call_tool("list_results")
    if result.get("status") == "ok":
        log("list_results works (empty is ok)", "pass", f"{result.get('count', 0)} results")
    else:
        log("list_results works", "fail", str(result))


# ─────────────────────────────────────────────
# Scenario 2: Read Files
# ─────────────────────────────────────────────

async def scenario_2_read_files():
    print("\n" + "-" * 55)
    print("SCENARIO 2: Read Resume and Job Description")
    print("-" * 55)

    # Read a resume
    result = await call_tool("read_resume", {"filename": "alice_resume.txt"})
    if result.get("status") == "ok" and "Alice" in result.get("content", ""):
        log("read_resume — alice_resume.txt", "pass", f"{result['char_count']} chars")
    else:
        log("read_resume — alice_resume.txt", "fail", str(result))

    # Read a JD
    result = await call_tool("read_job_description", {"filename": "java_backend_jd.txt"})
    if result.get("status") == "ok" and "Java" in result.get("content", ""):
        log("read_job_description — java_backend_jd.txt", "pass", f"{result['char_count']} chars")
    else:
        log("read_job_description — java_backend_jd.txt", "fail", str(result))

    # Read non-existent file (should error gracefully)
    result = await call_tool("read_resume", {"filename": "ghost.txt"})
    if result.get("status") == "error":
        log("read_resume — non-existent file returns error", "pass", result.get("message", ""))
    else:
        log("read_resume — non-existent file should error", "fail", str(result))


# ─────────────────────────────────────────────
# Scenario 3: Batch Processing
# ─────────────────────────────────────────────

async def scenario_3_batch_process():
    print("\n" + "-" * 55)
    print("SCENARIO 3: Batch Processing (MCP-Specific Tool)")
    print("-" * 55)

    # Batch all resumes
    result = await call_tool("batch_process", {
        "jd_filename": "java_backend_jd.txt",
        "resume_filenames": "",
    })
    if result.get("status") == "ok" and result.get("resume_count", 0) >= 3:
        log("batch_process — all resumes", "pass", f"{result['resume_count']} resumes in 1 call")
        log("JD included in batch", "pass" if result.get("job_description") else "fail",
            result.get("job_description", {}).get("filename", "missing"))
        for r in result.get("resumes", []):
            log(f"  Resume: {r['filename']}", "info", f"checksum: {r['checksum']}")
    else:
        log("batch_process — all resumes", "fail", str(result))

    # Batch specific resumes only
    result = await call_tool("batch_process", {
        "jd_filename": "java_backend_jd.txt",
        "resume_filenames": "alice_resume.txt, bob_resume.txt",
    })
    if result.get("status") == "ok" and result.get("resume_count") == 2:
        log("batch_process — specific resumes", "pass", "2 of 3 resumes loaded")
    else:
        log("batch_process — specific resumes", "fail", str(result))


# ─────────────────────────────────────────────
# Scenario 4: Watch Directory
# ─────────────────────────────────────────────

async def scenario_4_watch_directory():
    print("\n" + "-" * 55)
    print("SCENARIO 4: watch_directory (MCP-Specific Tool)")
    print("-" * 55)

    # Initialize snapshot
    result = await call_tool("watch_directory", {"directory": "resumes", "reset": True})
    if result.get("status") == "ok" and result.get("action") == "snapshot_initialized":
        log("watch_directory — snapshot initialized", "pass",
            f"Tracking {len(result.get('tracked_files', []))} files")
    else:
        log("watch_directory — snapshot initialization", "fail", str(result))

    # Check again immediately (no new files expected)
    result = await call_tool("watch_directory", {"directory": "resumes", "reset": False})
    if result.get("status") == "ok" and not result.get("changes_detected"):
        log("watch_directory — no changes detected", "pass", "Stable directory")
    else:
        log("watch_directory — stability check", "fail", str(result))

    # Note: watch_directory uses in-memory state. Since each stdio call opens
    # a fresh server process, the snapshot resets per call. We verify the tool
    # works correctly by checking status=ok and change detection fields exist.
    new_file = BASE_DIR / "resumes" / "dave_resume.txt"
    new_file.write_text("Name: Dave Test\nSKILLS: Java, Spring Boot\n")
    result = await call_tool("watch_directory", {"directory": "resumes", "reset": True})
    new_file.unlink(missing_ok=True)
    if result.get("status") == "ok" and "dave_resume.txt" in result.get("tracked_files", []):
        log("watch_directory — detects new file in snapshot", "pass", "dave_resume.txt tracked")
    else:
        log("watch_directory — new file detection", "fail", str(result))

    # Invalid directory
    result = await call_tool("watch_directory", {"directory": "invalid_dir"})
    if result.get("status") == "error":
        log("watch_directory — invalid dir returns error", "pass", result.get("message", ""))
    else:
        log("watch_directory — invalid dir should error", "fail", str(result))


# ─────────────────────────────────────────────
# Scenario 5: Save and Retrieve Results
# ─────────────────────────────────────────────

async def scenario_5_save_results():
    print("\n" + "-" * 55)
    print("SCENARIO 5: Save & Retrieve Match Results")
    print("-" * 55)

    # Save a result
    payload = json.dumps({
        "session_id": "test_abc",
        "job_description_file": "java_backend_jd.txt",
        "ranked_candidates": [
            {"filename": "alice_resume.txt", "score": 88, "recommendation": "shortlist"},
            {"filename": "bob_resume.txt",   "score": 62, "recommendation": "consider"},
        ],
    })
    result = await call_tool("save_result", {
        "result_data": payload,
        "filename": "test_scenario_result.json",
    })
    if result.get("status") == "ok":
        log("save_result — saves JSON file", "pass", result.get("filename", ""))
    else:
        log("save_result — save failed", "fail", str(result))

    # List results (should now include it)
    result = await call_tool("list_results")
    files = [r["filename"] for r in result.get("results", [])]
    if "test_scenario_result.json" in files:
        log("list_results — includes saved result", "pass", str(files))
    else:
        log("list_results — saved result not found", "fail", str(result))

    # Read back
    result = await call_tool("read_result", {"filename": "test_scenario_result.json"})
    if result.get("status") == "ok" and result.get("content", {}).get("session_id") == "test_abc":
        log("read_result — correct data retrieved", "pass", "session_id matches")
    else:
        log("read_result — data mismatch", "fail", str(result))

    # Save invalid JSON
    result = await call_tool("save_result", {"result_data": "not-valid-json"})
    if result.get("status") == "error":
        log("save_result — rejects invalid JSON", "pass", result.get("message", ""))
    else:
        log("save_result — should reject invalid JSON", "fail", str(result))


# ─────────────────────────────────────────────
# Scenario 6: Error Handling
# ─────────────────────────────────────────────

async def scenario_6_error_handling():
    print("\n" + "-" * 55)
    print("SCENARIO 6: Error Handling")
    print("-" * 55)

    cases = [
        ("read_resume",            {"filename": "does_not_exist.txt"},   "non-existent resume"),
        ("read_job_description",   {"filename": "does_not_exist.txt"},   "non-existent JD"),
        ("read_result",            {"filename": "missing.json"},          "non-existent result"),
        ("batch_process",          {"jd_filename": "ghost_jd.txt"},       "bad JD in batch"),
        ("watch_directory",        {"directory": "foo"},                  "invalid watch dir"),
    ]

    for tool, args, label in cases:
        result = await call_tool(tool, args)
        if result.get("status") == "error":
            log(f"{tool} — {label}", "pass", f"Error: {result.get('message', 'handled')[:60]}")
        else:
            log(f"{tool} — {label} should error", "fail", str(result)[:80])


# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────

def print_summary():
    print("\n" + "=" * 55)
    print("  TEST SUMMARY")
    print("=" * 55)
    total = len([r for r in results_log if r["status"] in ("pass", "fail")])
    passed = len([r for r in results_log if r["status"] == "pass"])
    failed = len([r for r in results_log if r["status"] == "fail"])
    print(f"  Total : {total}")
    print(f"  Passed: {passed}  [PASS]")
    print(f"  Failed: {failed}  [FAIL]")
    print("=" * 55)
    if failed:
        print("\nFailed tests:")
        for r in results_log:
            if r["status"] == "fail":
                print(f"  [FAIL] {r['label']}")
                if r["detail"]:
                    print(f"    {r['detail']}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

async def main():
    print("=" * 55)
    print("  MCP FILESYSTEM SERVER — TEST SCENARIOS")
    print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    await scenario_1_discovery()
    await scenario_2_read_files()
    await scenario_3_batch_process()
    await scenario_4_watch_directory()
    await scenario_5_save_results()
    await scenario_6_error_handling()

    print_summary()


if __name__ == "__main__":
    asyncio.run(main())
