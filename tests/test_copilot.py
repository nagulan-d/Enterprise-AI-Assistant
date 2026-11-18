"""Tests for the Knowledge Copilot. Run with:  python -m unittest discover -s tests -v   (or pytest)

All tests run offline. The Claude path is exercised with a scripted fake model, so no API key is needed.
"""

from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kc import KnowledgeCopilot  # noqa: E402
from kc.chunker import chunk_sections  # noqa: E402
from kc.llm import LLM, LLMError  # noqa: E402
from kc.loaders import Section, load_bytes  # noqa: E402

SAMPLES = ROOT / "sample_docs"


def offline() -> KnowledgeCopilot:
    llm = LLM(api_key="")
    llm._client = None  # never call out, even if ANTHROPIC_API_KEY is set in the environment
    llm.reason = "offline for tests"
    return KnowledgeCopilot(llm=llm)


def loaded() -> KnowledgeCopilot:
    c = offline()
    names, errors = c.load_folder(SAMPLES)
    assert not errors and len(names) >= 7, (names, errors)
    return c


class FakeLLM:
    """Stands in for Claude: returns scripted replies and records every prompt it was given."""

    available = True
    model = "fake"
    reason = ""

    def __init__(self, *replies: str | Exception) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    def complete(self, system, messages, max_tokens=1500, temperature=0.2):
        self.calls.append({"system": system, "messages": messages})
        r = self.replies.pop(0) if self.replies else "ok"
        if isinstance(r, Exception):
            raise r
        return r


class Loading(unittest.TestCase):
    def test_markdown_headings_become_breadcrumbs(self):
        secs = load_bytes("a.md", b"# Policy\n\n## Leave\n\n### Sick\nTen days.\n")
        self.assertEqual(secs[0].heading, "Policy > Leave > Sick")

    def test_txt_and_csv(self):
        self.assertEqual(load_bytes("a.txt", b"hello world")[0].text, "hello world")
        csv = load_bytes("a.csv", b"item,cost\npen,5\n")[0].text
        self.assertIn("item: pen | cost: 5", csv)

    def test_docx_keeps_headings_and_tables(self):
        from docx import Document

        d = Document()
        d.add_heading("Checklist", 1)
        d.add_paragraph("Collect the laptop on day one.")
        t = d.add_table(rows=1, cols=2)
        t.cell(0, 0).text, t.cell(0, 1).text = "Badge", "Facilities"
        buf = io.BytesIO()
        d.save(buf)
        sec = load_bytes("a.docx", buf.getvalue())[0]
        self.assertEqual(sec.heading, "Checklist")
        self.assertIn("Collect the laptop", sec.text)
        self.assertIn("Badge | Facilities", sec.text)

    def test_pdf_records_page_numbers(self):
        from reportlab.pdfgen import canvas

        buf = io.BytesIO()
        c = canvas.Canvas(buf)
        c.drawString(72, 760, "First page text")
        c.showPage()
        c.drawString(72, 760, "Second page text")
        c.save()
        secs = load_bytes("a.pdf", buf.getvalue())
        self.assertEqual([s.page for s in secs], [1, 2])

    def test_rejects_unsupported_and_empty(self):
        with self.assertRaises(ValueError):
            load_bytes("a.exe", b"x")
        with self.assertRaises(ValueError):
            load_bytes("a.txt", b"   \n  ")


class Chunking(unittest.TestCase):
    def test_chunks_respect_size_and_keep_locator(self):
        text = "\n\n".join(f"Sentence number {i} is about leave policy." for i in range(80))
        chunks = chunk_sections("d.md", [Section(text, None, "Leave")], max_chars=300, overlap=60)
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(len(c.text) <= 300 + 60 for c in chunks))
        self.assertTrue(all(c.locator == "Leave" for c in chunks))
        self.assertEqual([c.index for c in chunks], list(range(len(chunks))))

    def test_one_huge_line_is_hard_wrapped(self):
        chunks = chunk_sections("d.txt", [Section("word " * 1000)], max_chars=200, overlap=0)
        self.assertTrue(all(len(c.text) <= 200 for c in chunks))


class Retrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = loaded()

    def top_doc(self, q):
        return self.c.ask(q).sources[0].doc

    def test_finds_the_right_document(self):
        cases = {
            "How fast must a phishing email be reported?": "IT_Security_SOP.md",
            "What is the hotel limit in metro cities?": "Travel_and_Expense_Policy.md",
            "How many days a week must I be in the office?": "Remote_Work_Policy.md",
            "What was the churn rate in Q3?": "Q3_2025_Business_Report.md",
            "Who is the project manager of Phoenix?": "Project_Phoenix_Charter.md",
        }
        for q, doc in cases.items():
            with self.subTest(q=q):
                self.assertEqual(self.top_doc(q), doc)

    def test_answers_contain_the_exact_figure(self):
        cases = {
            "What is the password length requirement?": "14 characters",
            "When must expense claims be submitted?": "30 days",
            "What is the internet allowance for remote workers?": "1,000",
            "How do I report a lost laptop?": "2 hours",
            "How many wellbeing days do I get?": "2 paid wellbeing",
        }
        for q, fig in cases.items():
            with self.subTest(q=q):
                a = self.c.ask(q)
                self.assertTrue(a.grounded)
                self.assertIn(fig, a.text + " ".join(s.text for s in a.sources))

    def test_everyday_wording_maps_to_policy_wording(self):
        a = self.c.ask("How many vacation days do I get?")
        self.assertIn("annual leave", a.text)
        a = self.c.ask("How many weeks of maternity leave?")
        self.assertIn("parental", a.text.lower())

    def test_unanswerable_questions_are_declined_not_guessed(self):
        for q in [
            "Who is the CEO of the company?",
            "What is the capital of France?",
            "How do I apply for a visa?",
            "What is the dress code?",
            "What is the stock price?",
        ]:
            with self.subTest(q=q):
                a = self.c.ask(q)
                self.assertFalse(a.grounded, a.text)
                self.assertEqual(a.sources, [])

    def test_every_citation_points_at_a_returned_source(self):
        a = self.c.ask("How many days of annual leave do I get?")
        import re

        cited = {int(n) for n in re.findall(r"\[S(\d+)\]", a.text)}
        self.assertTrue(cited)
        self.assertTrue(cited <= {s.id for s in a.sources})

    def test_scope_restricts_search(self):
        a = self.c.ask("How many days of annual leave do I get?", docs=["HR_Leave_Policy_2025.md"])
        self.assertEqual({s.doc for s in a.sources}, {"HR_Leave_Policy_2025.md"})

    def test_conflicting_versions_are_flagged(self):
        a = self.c.ask("How many days of sick leave do I get?")
        self.assertEqual({s.doc for s in a.sources}, {"HR_Leave_Policy_2024.md", "HR_Leave_Policy_2025.md"})
        self.assertIn("differ", a.text.lower())

    def test_no_documents(self):
        a = offline().ask("anything?")
        self.assertFalse(a.grounded)
        self.assertIn("No documents", a.text)

    def test_remove_document_removes_it_from_search(self):
        c = loaded()
        c.remove_document("Travel_and_Expense_Policy.md")
        self.assertFalse(c.ask("What is the hotel limit in metro cities?").grounded)


class Conversation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = loaded()

    def chat(self, *qs):
        hist, answers = [], []
        for q in qs:
            a = self.c.ask(q, hist)
            answers.append(a)
            hist += [{"role": "user", "content": q}, {"role": "assistant", "content": a.text}]
        return answers

    def test_pronoun_follow_up_resolves_to_previous_topic(self):
        a = self.chat("What is the process to request unpaid leave?", "Who needs to approve it?")[1]
        self.assertIn("unpaid", a.standalone_question.lower())
        self.assertIn("line manager and HR", a.text)

    def test_new_topic_is_not_polluted_by_previous_question(self):
        a = self.chat("How many days of annual leave do I get?", "What about sick leave?")[1]
        self.assertEqual(a.standalone_question, "What about sick leave?")
        self.assertIn("sick leave", a.text)

    def test_unrelated_follow_up_is_still_declined(self):
        a = self.chat("What is the hotel cap in metro cities?", "Who is the CEO?")[1]
        self.assertFalse(a.grounded)


class Summaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = loaded()

    def test_all_styles_return_text_with_figures(self):
        for style in ("executive", "detailed", "bullets"):
            with self.subTest(style=style):
                s = self.c.summarize("IT_Security_SOP.md", style)
                self.assertEqual(s.mode, "extractive")
                self.assertTrue(len(s.text) > 100)
                self.assertTrue(s.key_terms)

    def test_unknown_document(self):
        with self.assertRaises(KeyError):
            self.c.summarize("nope.md")


class Comparing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = loaded()

    def test_detects_changed_figures_between_versions(self):
        r = self.c.compare(["HR_Leave_Policy_2024.md", "HR_Leave_Policy_2025.md"])
        differs = [row for row in r.table if row["status"] == "Differs"]
        self.assertGreaterEqual(len(differs), 5)
        annual = next(row for row in differs if "18 days" in row["a"])
        self.assertIn("22 days", annual["b"])
        self.assertEqual(annual["topic"], "Annual Leave")

    def test_focus_topic(self):
        r = self.c.compare(["HR_Leave_Policy_2024.md", "HR_Leave_Policy_2025.md"], "parental leave")
        self.assertTrue(r.table)
        self.assertTrue(all("Parental" in row["topic"] for row in r.table if row["status"] == "Differs"))

    def test_needs_two_documents(self):
        with self.assertRaises(ValueError):
            self.c.compare(["HR_Leave_Policy_2024.md"])


class ActionPlans(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = loaded()

    def test_extracts_owner_deadline_and_priority(self):
        p = self.c.action_plan("Prepare Project Phoenix for rollout", ["Project_Phoenix_Charter.md"])
        by_text = {i["action"]: i for i in p.items}
        report = next(v for k, v in by_text.items() if "status report" in k)
        self.assertEqual(report["owner"], "Project Manager")  # subject of the sentence, not the recipient
        self.assertEqual(report["deadline"], "every Friday")
        self.assertEqual(report["priority"], "High")
        legal = next(v for k, v in by_text.items() if "CFO approval" in k)
        self.assertEqual(legal["deadline"], "by 15 September 2025")

    def test_facts_are_not_action_items(self):
        p = self.c.action_plan("", ["Project_Phoenix_Charter.md"])
        self.assertFalse(any(i["action"] == "The approved budget is 4.2 crore INR." for i in p.items))

    def test_plan_cites_only_passages_it_used(self):
        p = self.c.action_plan("Prepare Project Phoenix for rollout")
        self.assertEqual({s.doc for s in p.sources}, {"Project_Phoenix_Charter.md"})

    def test_without_goal_scans_all_selected_documents(self):
        p = self.c.action_plan("", ["IT_Security_SOP.md", "Travel_and_Expense_Policy.md"])
        self.assertGreaterEqual(len(p.items), 5)


class ClaudePath(unittest.TestCase):
    """The LLM branch, driven by a scripted fake model."""

    def make(self, *replies):
        fake = FakeLLM(*replies)
        c = KnowledgeCopilot(llm=fake)
        c.load_folder(SAMPLES)
        return c, fake

    def test_ask_sends_only_retrieved_passages_and_keeps_cited_sources(self):
        c, fake = self.make("Employees get 22 days [S1].")
        a = c.ask("How many days of annual leave do I get?")
        self.assertEqual(a.mode, "llm")
        prompt = fake.calls[0]["messages"][-1]["content"]
        self.assertIn("[S1]", prompt)
        self.assertIn("Answer ONLY", fake.calls[0]["system"])
        self.assertEqual([s.id for s in a.sources], [1])

    def test_follow_up_is_condensed_by_the_model_and_history_is_passed(self):
        c, fake = self.make("What is the sick leave entitlement?", "10 days [S1].")
        hist = [{"role": "user", "content": "How many annual leave days?"}, {"role": "assistant", "content": "18 [S1]."}]
        a = c.ask("and sick?", hist)
        self.assertEqual(a.standalone_question, "What is the sick leave entitlement?")
        self.assertEqual(fake.calls[1]["messages"][0]["role"], "user")
        self.assertEqual(len(fake.calls[1]["messages"]), 3)  # prior turn + new question

    def test_model_failure_falls_back_to_extractive(self):
        c, _ = self.make(LLMError("boom"))
        a = c.ask("How many days of annual leave do I get?")
        self.assertEqual(a.mode, "extractive")
        self.assertIn("boom", a.note)
        self.assertTrue(a.grounded)

    def test_no_model_call_when_nothing_relevant(self):
        c, fake = self.make()
        self.assertFalse(c.ask("What is the capital of France?").grounded)
        self.assertEqual(fake.calls, [])

    def test_action_plan_parses_json_and_resolves_source_labels(self):
        reply = json.dumps({
            "summary": "Rollout needs sign-off [S1].",
            "recommendation": "Get security sign-off first [S1].",
            "action_items": [{"action": "Obtain security sign-off", "owner": "Security Team",
                              "priority": "high", "deadline": "Before Rollout", "source": "S1"}],
            "risks": ["SSO contract unsigned"],
        })
        c, _ = self.make("```json\n" + reply + "\n```")
        p = c.action_plan("Prepare Project Phoenix for rollout", ["Project_Phoenix_Charter.md"])
        self.assertEqual(p.mode, "llm")
        self.assertEqual(p.items[0]["priority"], "High")
        self.assertIn("Project_Phoenix_Charter.md", p.items[0]["source"])
        self.assertEqual(p.risks, ["SSO contract unsigned"])

    def test_long_document_is_summarised_map_reduce(self):
        c, fake = self.make()
        c.add_document("big.txt", ("Employees must file the report on time. " * 1500).encode())
        s = c.summarize("big.txt")
        self.assertEqual(s.mode, "llm")
        self.assertGreater(len(fake.calls), 2)  # several partial summaries, then the final one


class WebApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app as webapp

        webapp.copilot = offline()
        cls.webapp = webapp
        cls.client = webapp.app.test_client()

    def test_full_flow(self):
        c = self.client
        self.assertEqual(c.get("/").status_code, 200)
        r = c.post("/api/samples").get_json()
        self.assertGreaterEqual(len(r["docs"]), 7)
        a = c.post("/api/ask", json={"question": "How many wellbeing days do I get?"}).get_json()
        self.assertTrue(a["grounded"])
        self.assertTrue(a["sources"][0]["label"])
        self.assertEqual(c.post("/api/ask", json={"question": "  "}).status_code, 400)
        s = c.post("/api/summarize", json={"doc": "IT_Security_SOP.md"}).get_json()
        self.assertTrue(s["text"])
        self.assertEqual(c.post("/api/summarize", json={"doc": "missing.md"}).status_code, 400)
        cmp = c.post("/api/compare", json={"docs": ["HR_Leave_Policy_2024.md", "HR_Leave_Policy_2025.md"]})
        self.assertIn("Differs", cmp.get_json()["text"])
        self.assertEqual(c.post("/api/compare", json={"docs": ["HR_Leave_Policy_2024.md"]}).status_code, 400)
        p = c.post("/api/actions", json={"goal": "Prepare Project Phoenix for rollout"}).get_json()
        self.assertTrue(p["items"])
        left = c.delete("/api/docs/IT_Security_SOP.md").get_json()
        self.assertNotIn("IT_Security_SOP.md", [d["name"] for d in left["docs"]])

    def test_upload_rejects_bad_files_but_keeps_good_ones(self):
        data = {"files": [(io.BytesIO(b"Visitors must sign in."), "../../evil/visitors.txt"),
                          (io.BytesIO(b"x"), "bad.exe"), (io.BytesIO(b""), "empty.txt")]}
        r = self.client.post("/api/upload", data=data, content_type="multipart/form-data").get_json()
        res = {x["name"]: x for x in r["results"]}
        self.assertTrue(res["visitors.txt"]["ok"])  # path components stripped
        self.assertFalse(res["bad.exe"]["ok"])
        self.assertFalse(res["empty.txt"]["ok"])

    def test_api_key_is_never_echoed(self):
        r = self.client.post("/api/settings", json={"api_key": "sk-ant-secret-123"})
        self.assertNotIn("secret", r.get_data(as_text=True))
        self.client.post("/api/settings", json={"clear": True})
        self.assertFalse(self.client.get("/api/status").get_json()["llm"]["available"])


if __name__ == "__main__":
    unittest.main()
