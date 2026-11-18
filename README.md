# Knowledge Copilot

An Enterprise Knowledge Intelligence & Decision Copilot. Employees add company documents (policies, SOPs, HR guidelines, project documents, reports) and then ask questions, get summaries, compare documents and turn them into action items. Every answer shows the passages it came from.

It runs in two modes, and the whole app works in both:

| | Offline (default) | Connected to Claude |
|---|---|---|
| Needs | nothing beyond `pip install` | `ANTHROPIC_API_KEY` |
| Answers | quotes the most relevant sentences, with citations | written answers grounded in the retrieved passages, with citations |
| Summaries | key sentences extracted from the document | executive / section-by-section / bullet summaries (map-reduce for long files) |
| Comparison | sentence-by-sentence table that flags changed figures | structured comparison: overview, table, similarities, conflicts, recommendation |
| Decisions | obligations, owners and deadlines pulled from the text | grounded recommendation, action items, risks |
| Follow-ups | resolved with retrieval heuristics | question rewritten by the model using the conversation |

If a Claude call fails (network, quota), the app falls back to offline mode for that request and says so.

## Run it

```bash
pip install -r requirements.txt
python app.py --samples          # then open http://127.0.0.1:5000
```

`--samples` preloads the bundled documents (two versions of an HR leave policy, remote work, IT security, travel and expense, a project charter, a quarterly report). Without it, drag your own files into the left panel. Supported: PDF, Word (`.docx`), Markdown, text, CSV.

To use Claude, either set the key before starting (`export ANTHROPIC_API_KEY=sk-ant-...`; optionally `COPILOT_MODEL`) or click the mode pill at the top right and paste it. A key entered in the page lives only in the server's memory.

## What to try

| Goal | Try |
|---|---|
| Answer with sources | Ask "How many days of annual leave do I get?" The evidence panel shows the exact passages, with matching words highlighted. Select a citation number to jump to its passage. |
| Conflicting documents | The same question returns both the 2024 and 2025 figures and warns that they differ. |
| Follow-ups | Ask "What about sick leave?" or "Who needs to approve it?" right after another question. |
| Honest "I don't know" | Ask "Who is the CEO?" The copilot says it can't find it rather than guessing. |
| Summarize | Summarize tab, pick `IT_Security_SOP.md`. |
| Compare | Tick the two leave policies in the library, open Compare. Changed figures are listed first. |
| Recommendations | Decide tab: "Prepare Project Phoenix for rollout", or leave the goal blank to list every obligation and deadline. |

## How it works

```
upload ─► loaders ─► chunker ─► hybrid retriever ─► answer / summary / comparison / plan
          pdf,docx   overlapping   BM25 + TF-IDF       Claude (grounded prompt)
          md,txt,csv chunks that   cosine, with        or offline extractive fallback
                     keep page &   IDF-weighted
                     heading       relevance gate
```

- **Loading** (`kc/loaders.py`): keeps page numbers for PDFs, and heading breadcrumbs for Markdown and Word (`Policy > Leave > Sick leave`). Word tables are kept as rows.
- **Chunking** (`kc/chunker.py`): about 900 characters with a small overlap, split on paragraph and sentence boundaries. Each chunk remembers its document, page and section, which become the citation.
- **Retrieval** (`kc/retriever.py`): BM25 (exact terms, numbers, acronyms) fused with TF-IDF cosine (word and bigram overlap), with light stemming and a small thesaurus (`vacation` finds `annual leave`). It runs entirely locally; no embedding model download and no data leaves the machine except the retrieved passages sent to Claude when connected.
- **Relevance gate**: a question is only answered when the retrieved passages actually cover its terms. Words the documents never use ("CEO") lower confidence, so unanswerable questions are declined instead of answered with the nearest unrelated text.
- **Grounding** (`kc/copilot.py`): Claude is told to answer only from the numbered passages, cite each claim as `[S1]`, say what is missing instead of guessing, and flag disagreements between documents. Citations in the reply are checked against the passages that were sent, and only cited passages are shown as sources.

## Project layout

```
app.py              Flask server and JSON API
kc/                 the copilot (no web code)
  copilot.py        ask, summarize, compare, action_plan; prompts and offline fallbacks
  retriever.py      hybrid BM25 + TF-IDF search
  loaders.py chunker.py text.py llm.py
templates/ static/  the web interface
sample_docs/        demo documents
tests/              37 offline tests (the Claude path is tested with a scripted fake model)
```

Run the tests with `python -m unittest discover -s tests -v` (or `pytest`).

## Limits and next steps

- **Scanned PDFs** have no text layer and are rejected with a message; add OCR (for example `pytesseract`) in `loaders.py` to support them.
- **Retrieval is lexical.** The thesaurus in `kc/text.py` covers common HR, IT and finance wording; add your own jargon there. For heavier paraphrase, swap the TF-IDF half of `HybridRetriever` for embeddings. The rest of the pipeline does not change.
- **One shared library, no login.** Everyone who opens the page sees the same documents, and the index lives in memory (restarting clears it). Bind to `127.0.0.1` (the default) unless you add authentication and persistence. Do not expose it with `--host 0.0.0.0` on an untrusted network.
- The built-in Flask server is for demos. Use a production WSGI server behind authentication for real deployment.
