"""Web app for the Enterprise Knowledge Intelligence & Decision Copilot.

    python app.py                      # http://127.0.0.1:5000
    python app.py --samples            # preload the bundled sample documents
    ANTHROPIC_API_KEY=... python app.py   # answer with Claude instead of the offline extractive mode

One shared copilot instance serves one team/demo: uploaded documents are visible to everyone who
opens the page. Requests are serialised with a lock so the search index is never rebuilt mid-query.
"""

from __future__ import annotations

import argparse
import os
import threading
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from kc import LLM, KnowledgeCopilot
from kc.loaders import SUPPORTED

ROOT = Path(__file__).parent
SAMPLES = ROOT / "sample_docs"
MAX_UPLOAD_MB = 25

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

copilot = KnowledgeCopilot()
lock = threading.RLock()

SUGGESTIONS = [
    "How many days of annual leave do I get?",
    "What changed in the 2025 leave policy?",
    "How fast must a phishing email be reported?",
    "What is the hotel limit for travel in metro cities?",
]


def _source(s) -> dict:
    return {"id": s.id, "doc": s.doc, "locator": s.locator, "text": s.text, "score": s.score, "label": s.label}


def _status() -> dict:
    llm = copilot.llm
    docs = [
        {
            "name": r["Document"], "type": r["Type"], "pages": r["Pages"], "words": r["Words"],
            "chunks": r["Chunks"], "key_terms": r["Key terms"], "added": r["Added"],
        }
        for r in copilot.stats()
    ]
    has_samples = any(d["name"].startswith("HR_Leave_Policy") for d in docs)
    return {
        "docs": docs,
        "llm": {"available": llm.available, "model": llm.model, "reason": llm.reason},
        "suggestions": SUGGESTIONS if has_samples else [],
        "supported": list(SUPPORTED),
    }


def _error(message: str, code: int = 400):
    return jsonify({"error": message}), code


@app.errorhandler(413)
def too_large(_):
    return _error(f"File too large. The limit is {MAX_UPLOAD_MB} MB per upload.", 413)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    with lock:
        return jsonify(_status())


@app.post("/api/upload")
def upload():
    files = request.files.getlist("files")
    if not files:
        return _error("No files received.")
    results = []
    with lock:
        for f in files:
            name = os.path.basename((f.filename or "").replace("\\", "/")).strip()
            if not name:
                continue
            if Path(name).suffix.lower() not in SUPPORTED:
                results.append({"name": name, "ok": False, "error": f"Unsupported type. Use {', '.join(SUPPORTED)}."})
                continue
            try:
                rec = copilot.add_document(name, f.read())
                results.append({"name": name, "ok": True, "chunks": rec.chunk_count})
            except Exception as exc:  # unreadable / scanned / corrupt files are reported per file
                results.append({"name": name, "ok": False, "error": str(exc)})
        return jsonify({"results": results, **_status()})


@app.post("/api/samples")
def samples():
    with lock:
        loaded, errors = copilot.load_folder(SAMPLES)
        return jsonify({"loaded": loaded, "errors": errors, **_status()})


@app.delete("/api/docs/<path:name>")
def delete_doc(name: str):
    with lock:
        copilot.remove_document(name)
        return jsonify(_status())


@app.post("/api/ask")
def ask():
    body = request.get_json(silent=True) or {}
    question = str(body.get("question", "")).strip()
    if not question:
        return _error("Type a question first.")
    with lock:
        a = copilot.ask(question, body.get("history") or [], body.get("docs") or None)
    return jsonify({
        "text": a.text, "sources": [_source(s) for s in a.sources], "grounded": a.grounded,
        "confidence": a.confidence, "mode": a.mode, "standalone": a.standalone_question, "note": a.note,
    })


@app.post("/api/summarize")
def summarize():
    body = request.get_json(silent=True) or {}
    with lock:
        try:
            s = copilot.summarize(str(body.get("doc", "")), str(body.get("style", "executive")))
        except KeyError:
            return _error("Choose a document to summarize.")
    return jsonify({
        "text": s.text, "doc": s.doc, "mode": s.mode, "key_terms": s.key_terms, "words": s.words, "note": s.note,
    })


@app.post("/api/compare")
def compare():
    body = request.get_json(silent=True) or {}
    with lock:
        try:
            c = copilot.compare(body.get("docs") or [], str(body.get("aspect", "")))
        except ValueError as exc:
            return _error(str(exc))
    return jsonify({
        "text": c.text, "docs": c.docs, "sources": [_source(s) for s in c.sources], "mode": c.mode, "note": c.note,
    })


@app.post("/api/actions")
def actions():
    body = request.get_json(silent=True) or {}
    with lock:
        p = copilot.action_plan(str(body.get("goal", "")), body.get("docs") or None)
    return jsonify({
        "summary": p.summary, "recommendation": p.recommendation, "items": p.items, "risks": p.risks,
        "sources": [_source(s) for s in p.sources], "mode": p.mode, "note": p.note,
    })


@app.post("/api/settings")
def settings():
    """Connect Claude from the UI. The key is held in memory only and never echoed back."""
    body = request.get_json(silent=True) or {}
    key = str(body.get("api_key", "")).strip()
    model = str(body.get("model", "")).strip() or None
    with lock:
        if body.get("clear"):  # disconnect: back to offline mode regardless of environment variables
            copilot.llm = LLM(api_key="x", model=model or copilot.llm.model)
            copilot.llm._client, copilot.llm.api_key = None, ""
            copilot.llm.reason = "Disconnected from Claude"
        else:
            copilot.llm = LLM(api_key=key or copilot.llm.api_key, model=model or copilot.llm.model)
        return jsonify(_status())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to share on your network (no login!)")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--samples", action="store_true", help="preload the bundled sample documents")
    args = ap.parse_args()
    if args.samples or os.environ.get("COPILOT_SAMPLES") == "1":
        loaded, _ = copilot.load_folder(SAMPLES)
        print(f"Loaded {len(loaded)} sample documents")
    mode = f"Claude ({copilot.llm.model})" if copilot.llm.available else f"offline extractive mode ({copilot.llm.reason})"
    print(f"Answering with: {mode}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
