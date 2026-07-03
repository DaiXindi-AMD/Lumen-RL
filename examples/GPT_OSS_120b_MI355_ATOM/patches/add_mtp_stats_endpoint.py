"""Patch: add /debug/mtp_stats endpoint to official ATOM.

Strategy: the scheduler's SpecStats writes JSON to /tmp/mtp_stats.json on every
log interval AND on explicit request. The API server adds a GET endpoint that:
1. Triggers a utility command to refresh the file
2. Reads and returns the JSON
"""
import os

# --- Patch 1: engine_core.py — write stats to file on get_mtp_stats ---
EC_FILE = "/app/ATOM/atom/model_engine/engine_core.py"

with open(EC_FILE) as f:
    ec_src = f.read()

OLD_MTP = '''                        elif cmd == "get_mtp_stats":
                            self.print_mtp_statistics()'''

NEW_MTP = '''                        elif cmd == "get_mtp_stats":
                            self.print_mtp_statistics()
                            self._write_mtp_stats_file()'''

if "_write_mtp_stats_file" not in ec_src:
    ec_src = ec_src.replace(OLD_MTP, NEW_MTP)

    WRITE_METHOD = '''
    def _write_mtp_stats_file(self):
        """Dump SpecStats to /tmp/mtp_stats.json for the API endpoint."""
        import json as _json
        ss = getattr(self.scheduler, "spec_stats", None)
        if ss is None:
            data = {"enabled": False}
        else:
            s = ss.get_statistics()
            ts = ss.total_steps
            avg = 1 + s["total_accepted_tokens"] / ts if ts > 0 else 0
            total = sum(s["distribution"].values())
            dist_pct = {k: round(v / total * 100, 1) if total > 0 else 0 for k, v in s["distribution"].items()}
            data = {
                "enabled": True,
                "total_draft_tokens": s["total_draft_tokens"],
                "total_accepted_tokens": s["total_accepted_tokens"],
                "acceptance_rate": round(s["acceptance_rate"], 4),
                "average_tokens_per_forward": round(avg, 3),
                "distribution": {str(k): v for k, v in s["distribution"].items()},
                "distribution_percent": {str(k): v for k, v in dist_pct.items()},
            }
        with open("/tmp/mtp_stats.json", "w") as _f:
            _json.dump(data, _f)

'''
    # Insert before process_output_sockets
    anchor = "    def process_output_sockets("
    if anchor in ec_src:
        ec_src = ec_src.replace(anchor, WRITE_METHOD + anchor)
        with open(EC_FILE, "w") as f:
            f.write(ec_src)
        print(f"Patched {EC_FILE}: added _write_mtp_stats_file")
    else:
        print(f"WARNING: anchor not found in {EC_FILE}")
else:
    print(f"{EC_FILE}: already patched")


# --- Patch 2: api_server.py — add /debug/mtp_stats endpoint ---
API_FILE = "/app/ATOM/atom/entrypoints/openai/api_server.py"

with open(API_FILE) as f:
    api_src = f.read()

ENDPOINT = '''
@app.get("/debug/mtp_stats")
async def get_mtp_stats():
    """Return MTP/Eagle3 acceptance statistics."""
    import json as _json
    global engine
    if engine is not None:
        try:
            engine.print_mtp_statistics()
        except Exception:
            pass
    stats_path = "/tmp/mtp_stats.json"
    try:
        import asyncio
        await asyncio.sleep(0.2)
        with open(stats_path) as _f:
            return _json.load(_f)
    except FileNotFoundError:
        return {"enabled": False, "total_draft_tokens": 0, "total_accepted_tokens": 0,
                "acceptance_rate": 0.0, "average_tokens_per_forward": 0.0,
                "distribution": {}, "distribution_percent": {}}


'''

if "/debug/mtp_stats" not in api_src:
    # Insert after @app.get("/health") block
    anchor = '@app.get("/health")'
    idx = api_src.find(anchor)
    if idx >= 0:
        # Find end of health function
        next_decorator = api_src.find("\n@app.", idx + 1)
        if next_decorator >= 0:
            api_src = api_src[:next_decorator] + "\n" + ENDPOINT + api_src[next_decorator:]
        else:
            api_src += ENDPOINT
        with open(API_FILE, "w") as f:
            f.write(api_src)
        print(f"Patched {API_FILE}: added /debug/mtp_stats endpoint")
    else:
        print(f"WARNING: health endpoint not found in {API_FILE}")
else:
    print(f"{API_FILE}: already patched")


# --- Patch 3: llm_engine.py — expose print_mtp_statistics ---
LLM_FILE = "/app/ATOM/atom/model_engine/llm_engine.py"

with open(LLM_FILE) as f:
    llm_src = f.read()

# print_mtp_statistics already exists, just make sure it's there
if "print_mtp_statistics" not in llm_src:
    print(f"WARNING: print_mtp_statistics not found in {LLM_FILE}")
else:
    print(f"{LLM_FILE}: print_mtp_statistics already exists, OK")
