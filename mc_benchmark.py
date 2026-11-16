"""
Star Wars Multiple-Choice AI Benchmark

Gives every question in questions.csv to each AI model, grades the A/B/C/D answers,
and produces a report card for each model:
  - final grade (%) and letter grade
  - best and worst categories
  - every graded question, with the wrong ones marked

Standard Python 3 only, no packages to install.

USAGE (in Terminal, from this folder):
  export OPENROUTER_API_KEY="sk-or-v1-..."   # one key, all models (see MODELS below)
  python3 mc_benchmark.py               # run every model that has a key
  python3 mc_benchmark.py --dry-run     # test everything offline with a fake model
  python3 mc_benchmark.py --file my_questions.csv

OPENROUTER (one key for every model):
  export OPENROUTER_API_KEY="sk-or-v1-..."
  python3 mc_benchmark.py
  Check the key without showing it:  echo ${OPENROUTER_API_KEY:0:6}   -> should print sk-or-
  A model that errors (e.g. retired model id, no credits) is skipped with the reason shown.

TROUBLESHOOTING "Please pass a valid API key":
  - Get the key from https://aistudio.google.com/apikey (Gemini keys start with "AIza").
  - Check it without showing it:   echo ${GEMINI_API_KEY:0:4} ${#GEMINI_API_KEY}
    should print "AIza 39". If not, run the export line again.
  - Use straight quotes "..." around the key, not curly ones, and no spaces inside.
  - The export only lasts for that Terminal window; run it again in a new window.

QUESTIONS FILE (CSV; in Excel use File > Save As > CSV UTF-8):
  id, category, question, A, B, C, D, answer
  answer is just the letter: A, B, C or D

OUTPUT (in the reports/ folder, stamped with the date and time):
  report_<time>.html            open in any browser: report cards + all graded questions
  graded_questions_<time>.csv   every question x model, with CORRECT / INCORRECT
  summary_<time>.csv            grade per model and per category (for charts in the paper)
"""

import csv, html, json, os, re, sys, time, random, threading, urllib.request, urllib.error
from collections import defaultdict
from datetime import datetime

# ---------------------------------------------------------------------------
# MODELS: all use the same OpenAI-compatible format. A model runs only if its
# key is set. Model names change over time; check the provider if one errors.
# ---------------------------------------------------------------------------
MODELS = [
    # The newest flagship model from each of the four big AI labs, all on OpenRouter
    # (checked against openrouter.ai on Nov 11, 2026). One OpenRouter key runs all four.
    {"label": "Grok 4.7 (xAI)", "model": "x-ai/grok-4.7",
     "base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY"},
    {"label": "GPT-6 Astra (OpenAI)", "model": "openai/gpt-6-astra",
     "base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY"},
    {"label": "Claude Fable 5.2 (Anthropic)", "model": "anthropic/claude-fable-5.2",
     "base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY"},
    {"label": "Gemini 4 Carbon (Google)", "model": "google/gemini-4-carbon",
     "base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY"},
    # an open-weight model from a non-US lab, and a small model from a family already tested.
    {"label": "DeepSeek V4.1 Flash (DeepSeek)", "model": "deepseek/deepseek-v4.1-flash-1017",
     "base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY"},
    {"label": "Claude Haiku 5.5 (Anthropic)", "model": "anthropic/claude-haiku-5.5",
     "base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY"},
    # a second open-weight model, and a European lab.
    {"label": "GPT-6 Luna (OpenAI)", "model": "openai/gpt-6-luna",
     "base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY"},
    {"label": "Gemini 3.8 Flash (Google)", "model": "google/gemini-3.8-flash",
     "base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY"},
    {"label": "GLM 5.3 (Z.ai)", "model": "z-ai/glm-5.3",
     "base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY"},
    {"label": "Mistral Large 4 (Mistral)", "model": "mistralai/mistral-large-4-0",
     "base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY"},
]

SECONDS_BETWEEN_CALLS = 0.5
REASONING_EFFORT = "low"   # low / medium / high: same for every model
REQUEST_TIMEOUT = 120        # seconds to wait for one answer before giving up

# ---------------------------------------------------------------------------
# ENERGY & WATER ESTIMATES
# No AI company publishes energy use per model or per token, so these are ESTIMATES.
# They scale the per-prompt figures the companies did publish by how many tokens
# each AI actually used in this run (OpenRouter reports the token counts):
#   - Google (Aug 2025): median Gemini text prompt = 0.24 Wh and 0.26 mL of water
#   - OpenAI (Jun 2025): average ChatGPT query = 0.34 Wh and about 0.32 mL of water
# A typical chat reply is assumed to be about 500 output tokens, so
#   ~0.3 Wh / 500 tokens = 0.0006 Wh per output token (reasoning tokens included).
# Reading the prompt is far cheaper than writing the answer, so input tokens count 10%.
# Water is on-site data-center cooling: Google's 0.26 mL / 0.24 Wh = about 1.1 mL per Wh.
# The same rate is used for every model, because per-model figures aren't published.
WH_PER_OUTPUT_TOKEN = 0.0006
INPUT_TOKEN_WEIGHT = 0.1
ML_WATER_PER_WH = 1.1
   # raise to 2-4 if a free tier rate-limits you

# Identical instructions for every model, so the comparison is fair.
SYSTEM_PROMPT = (
    "You are taking a multiple-choice test about Star Wars. "
    "Each question has exactly one correct answer: A, B, C, or D. "
    "Unless a question says otherwise, answer according to current Disney canon. "
    "Reply with the single letter of your answer and nothing else."
)

LETTERS = ["A", "B", "C", "D"]

# ---------------------------------------------------------------------------
# Asking a model
# ---------------------------------------------------------------------------
_LAST = threading.local()      # token usage of the most recent call, per thread

def estimate_resources(prompt_tokens, completion_tokens):
    wh = (completion_tokens + INPUT_TOKEN_WEIGHT * prompt_tokens) * WH_PER_OUTPUT_TOKEN
    return wh, wh * ML_WATER_PER_WH

def ask_model(cfg, prompt, retries=2):
    if cfg.get("anthropic"):
        url = cfg["base_url"] + "/messages"
        headers = {"Content-Type": "application/json", "x-api-key": os.environ[cfg["key_env"]],
                   "anthropic-version": "2023-06-01"}
        body = {"model": cfg["model"], "max_tokens": 20, "temperature": 0,
                "system": SYSTEM_PROMPT, "messages": [{"role": "user", "content": prompt}]}
    else:
        url = cfg["base_url"].rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if cfg.get("key_env"):
            headers["Authorization"] = "Bearer " + os.environ[cfg["key_env"]]
        body = {"model": cfg["model"], "temperature": 0, "max_tokens": 4000,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": prompt}]}
        if "openrouter.ai" in cfg["base_url"]:
            # Every model gets the same LOW thinking effort: fast, cheap, and a fair comparison.
            body["reasoning"] = {"effort": REASONING_EFFORT, "exclude": True}
            body["usage"] = {"include": True}      # token counts and cost for the Ethics tab
    data = json.dumps(body).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
                out = json.loads(r.read())
            if cfg.get("anthropic"):
                return "".join(b.get("text", "") for b in out.get("content", [])).strip()
            _LAST.usage = out.get("usage") or {}
            return (out["choices"][0]["message"]["content"] or "").strip()
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="ignore")[:200]
            if e.code in (429, 500, 502, 503, 529) and attempt < retries - 1:
                wait = 5 * (attempt + 1)
                print(f"     (rate limit / server busy, retrying in {wait}s)")
                time.sleep(wait)
                continue
            return f"ERROR {e.code}: {msg}"
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3)
                continue
            return f"ERROR: {e}"

def fake_model(cfg, row):
    """--dry-run: right about 70% of the time, so the reports have something to show."""
    rng = random.Random(f"{cfg['label']}-{row['id']}")
    if rng.random() < cfg.get("skill", 0.7):
        return row["answer"]
    return rng.choice([l for l in LETTERS if l != row["answer"]] + ["I'm not sure"])

# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------
def extract_letter(response):
    """Pulls A/B/C/D out of answers like 'B', 'B)', '(C)', 'Answer: D', 'The answer is A.'"""
    if not response or response.startswith("ERROR"):
        return ""
    r = response.strip()
    m = re.match(r"^\W*([ABCD])\b", r, re.I)                       # starts with the letter
    if m: return m.group(1).upper()
    m = re.search(r"answer\s*(?:is|:)?\s*\(?([ABCD])\b", r, re.I)  # "answer is C"
    if m: return m.group(1).upper()
    found = set(re.findall(r"\b([ABCD])\)", r)) | set(re.findall(r"\(([ABCD])\)", r))
    if len(found) == 1: return found.pop()
    return ""                                                     # no clear single answer

def build_prompt(row):
    return (row["question"].strip() + "\n" +
            "\n".join(f"{l}) {row[l].strip()}" for l in LETTERS) +
            "\nAnswer with A, B, C, or D.")

def letter_grade(pct):
    for cutoff, g in [(93,"A"),(90,"A-"),(87,"B+"),(83,"B"),(80,"B-"),(77,"C+"),
                      (73,"C"),(70,"C-"),(67,"D+"),(60,"D")]:
        if pct >= cutoff: return g
    return "F"

# ---------------------------------------------------------------------------
# Loading questions
# ---------------------------------------------------------------------------
def load_questions(path):
    if not os.path.exists(path):
        sys.exit(f"Can't find {path}. Put your questions in that file (see top of this script).")
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    rows = [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in rows]
    rows = [r for r in rows if r.get("question")]
    need = ["id", "category", "question", "A", "B", "C", "D", "answer"]
    missing = [c for c in need if c not in rows[0]]
    if missing:
        sys.exit(f"{path} is missing column(s): {', '.join(missing)}\nNeeded: {', '.join(need)}")
    problems = []
    for r in rows:
        a = r["answer"].upper().strip(" ).(")
        if a not in LETTERS:
            problems.append(f"  question {r['id']}: answer '{r['answer']}' is not A/B/C/D")
        r["answer"] = a
        r["category"] = r["category"] or "uncategorized"
    if problems:
        sys.exit("Fix these before running:\n" + "\n".join(problems))
    return rows

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def analyze(results, models):
    stats = {}
    for m in models:
        rs = [r for r in results if r["model"] == m]
        cats = defaultdict(lambda: [0, 0])
        for r in rs:
            cats[r["category"]][1] += 1
            cats[r["category"]][0] += r["correct"]
        cat_pct = {c: 100 * ok / n for c, (ok, n) in cats.items()}
        best_v, worst_v = max(cat_pct.values()), min(cat_pct.values())
        correct = sum(r["correct"] for r in rs)
        pct = 100 * correct / len(rs)
        stats[m] = {
            "correct": correct, "total": len(rs), "pct": pct, "grade": letter_grade(pct),
            "cats": {c: (cats[c][0], cats[c][1], cat_pct[c]) for c in sorted(cats)},
            "best": sorted(c for c, v in cat_pct.items() if v == best_v), "best_pct": best_v,
            "worst": sorted(c for c, v in cat_pct.items() if v == worst_v), "worst_pct": worst_v,
            "wrong": [r for r in rs if not r["correct"]],
            "no_answer": sum(1 for r in rs if not r["model_answer"]),
        }
    return stats

def print_report(stats):
    for m, s in stats.items():
        print("\n" + "=" * 62)
        print(f" REPORT CARD: {m}")
        print("=" * 62)
        print(f" Final grade:   {s['pct']:.1f}%  ({s['grade']})   {s['correct']}/{s['total']} correct")
        print(f" Best category:  {', '.join(s['best'])}  ({s['best_pct']:.0f}%)")
        print(f" Worst category: {', '.join(s['worst'])}  ({s['worst_pct']:.0f}%)")
        print(" By category:")
        for c, (ok, n, p) in s["cats"].items():
            print(f"   {c:<28} {ok:>3}/{n:<3} {p:5.1f}%")
        print(f" Incorrect ({len(s['wrong'])}):")
        for r in s["wrong"]:
            got = r["model_answer"] or "no clear answer"
            print(f"   Q{r['id']}: answered {got}, correct {r['answer']}  - {r['question'][:55]}")
    if len(stats) > 1:
        print("\n" + "=" * 62 + "\n RANKING\n" + "=" * 62)
        for i, (m, s) in enumerate(sorted(stats.items(), key=lambda x: -x[1]["pct"]), 1):
            print(f" {i}. {m:<30} {s['pct']:5.1f}%  ({s['grade']})")

def write_html(path, stats, questions, results, stamp):
    e = html.escape
    by_q = defaultdict(dict)
    for r in results:
        by_q[r["id"]][r["model"]] = r
    models = list(stats)
    parts = [f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Star Wars AI Benchmark</title><style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f6f7f9;color:#1d2330}}
main{{max-width:1050px;margin:auto;padding:24px 16px}}
h1{{margin:0 0 4px}} .sub{{color:#5b6475;margin-bottom:24px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}}
.card{{background:#fff;border-radius:12px;padding:18px;box-shadow:0 1px 3px #0002}}
.grade{{font-size:44px;font-weight:700}} .letter{{font-size:20px;color:#5b6475;margin-left:6px}}
.good{{color:#17803d}} .bad{{color:#c0262d}}
table{{border-collapse:collapse;width:100%;font-size:14px}}
td,th{{padding:6px 8px;border-bottom:1px solid #e3e6ea;text-align:left;vertical-align:top}}
.bar{{height:8px;background:#e3e6ea;border-radius:4px;min-width:60px}}
.bar>div{{height:8px;border-radius:4px;background:#3a6ff7}}
.ok{{background:#e5f6ea;color:#17803d;font-weight:600}} .no{{background:#fde8e8;color:#c0262d;font-weight:600}}
.wrap{{overflow-x:auto;background:#fff;border-radius:12px;box-shadow:0 1px 3px #0002;margin-top:24px}}
h2{{margin-top:36px}}
</style></head><body><main>
<h1>Star Wars AI Benchmark Results</h1>
<div class="sub">{len(questions)} multiple-choice questions &middot; {len(models)} model(s) &middot; run {e(stamp)}</div>
<div class="cards">"""]
    for m, s in sorted(stats.items(), key=lambda x: -x[1]["pct"]):
        rows = "".join(
            f"<tr><td>{e(c)}</td><td>{ok}/{n}</td><td>{p:.0f}%</td>"
            f"<td><div class='bar'><div style='width:{p:.0f}%'></div></div></td></tr>"
            for c, (ok, n, p) in s["cats"].items())
        parts.append(f"""<div class="card"><h3 style="margin-top:0">{e(m)}</h3>
<div><span class="grade">{s['pct']:.1f}%</span><span class="letter">{s['grade']}</span></div>
<p>{s['correct']} of {s['total']} correct</p>
<p><b class="good">Best:</b> {e(', '.join(s['best']))} ({s['best_pct']:.0f}%)<br>
<b class="bad">Worst:</b> {e(', '.join(s['worst']))} ({s['worst_pct']:.0f}%)</p>
<table><tr><th>Category</th><th>Score</th><th>%</th><th></th></tr>{rows}</table></div>""")
    parts.append("</div>")

    # Incorrect answers per model
    parts.append("<h2>Questions answered incorrectly</h2>")
    for m, s in stats.items():
        parts.append(f"<h3>{e(m)} &mdash; {len(s['wrong'])} wrong</h3>")
        if not s["wrong"]:
            parts.append("<p>None, perfect score.</p>"); continue
        parts.append("<div class='wrap'><table><tr><th>#</th><th>Category</th><th>Question</th>"
                     "<th>AI answered</th><th>Correct</th></tr>")
        q_lookup = {q["id"]: q for q in questions}
        for r in s["wrong"]:
            q = q_lookup[r["id"]]
            got = r["model_answer"]
            got_txt = f"{got}) {e(q[got])}" if got else f"<i>no clear answer</i>: {e(r['raw_response'][:80])}"
            parts.append(f"<tr><td>{e(r['id'])}</td><td>{e(r['category'])}</td><td>{e(r['question'])}</td>"
                         f"<td class='no'>{got_txt}</td><td class='ok'>{r['answer']}) {e(q[r['answer']])}</td></tr>")
        parts.append("</table></div>")

    # Full graded sheet
    parts.append("<h2>All graded questions</h2><div class='wrap'><table><tr><th>#</th><th>Category</th>"
                 "<th>Question</th><th>Key</th>" + "".join(f"<th>{e(m)}</th>" for m in models) + "</tr>")
    for q in questions:
        cells = ""
        for m in models:
            r = by_q[q["id"]].get(m)
            if r:
                cls = "ok" if r["correct"] else "no"
                mark = "&#10003;" if r["correct"] else "&#10007;"
                cells += f"<td class='{cls}'>{mark} {r['model_answer'] or '?'}</td>"
            else:
                cells += "<td></td>"
        parts.append(f"<tr><td>{e(q['id'])}</td><td>{e(q['category'])}</td><td>{e(q['question'])}</td>"
                     f"<td><b>{q['answer']}</b></td>{cells}</tr>")
    parts.append("</table></div></main></body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(parts))


# ---------------------------------------------------------------------------
# TEST_ANSWERS folder: every answer, organized by AI
# ---------------------------------------------------------------------------
def export_test_answers(results, questions, folder="TEST_ANSWERS"):
    """Writes (overwriting the previous run's files):
         answers_by_AI.csv          one block per AI, every question, with its grade
         answers_side_by_side.csv   one row per question, one column per AI
         <AI name>.csv              a separate file for each AI
    """
    os.makedirs(folder, exist_ok=True)
    q_by_id = {q["id"]: q for q in questions}
    order = {q["id"]: i for i, q in enumerate(questions)}
    models = []
    for r in results:
        if r["model"] not in models:
            models.append(r["model"])

    def choice(qid, letter):
        q = q_by_id.get(qid)
        return f"{letter}) {q[letter]}" if (q and letter in LETTERS) else "no clear answer"

    header = ["AI", "Question #", "Category", "Question", "AI Answer", "Correct Answer", "Result"]
    def rows_for(m):
        rs = sorted([r for r in results if r["model"] == m], key=lambda r: order.get(r["id"], 0))
        return [[m, r["id"], r["category"], r["question"], choice(r["id"], r["model_answer"]),
                 choice(r["id"], r["answer"]), "CORRECT" if int(r["correct"]) else "INCORRECT"]
                for r in rs]
    def score_line(m):
        rs = [r for r in results if r["model"] == m]
        ok = sum(int(r["correct"]) for r in rs)
        pct = 100 * ok / len(rs)
        return [m, "", "", f"FINAL GRADE: {pct:.1f}% ({letter_grade(pct)})", f"{ok}/{len(rs)} correct", "", ""]

    with open(os.path.join(folder, "answers_by_AI.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        for m in models:
            w.writerows(rows_for(m))
            w.writerow(score_line(m))
            w.writerow([])
    for m in models:
        safe = re.sub(r"[^A-Za-z0-9.]+", "_", m).strip("_")
        with open(os.path.join(folder, f"{safe}.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows_for(m))
            w.writerow(score_line(m))
    with open(os.path.join(folder, "answers_side_by_side.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Question #", "Category", "Question", "Correct Answer"] + models)
        got = {(r["model"], r["id"]): r for r in results}
        for q in questions:
            cells = []
            for m in models:
                r = got.get((m, q["id"]))
                cells.append("" if not r else
                             f"{r['model_answer'] or '?'} - {'CORRECT' if int(r['correct']) else 'WRONG'}")
            w.writerow([q["id"], q["category"], q["question"], f"{q['answer']}) {q[q['answer']]}"] + cells)
    return folder

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = sys.argv[1:]
    dry = "--dry-run" in args
    qfile = args[args.index("--file") + 1] if "--file" in args else "questions.csv"
    questions = load_questions(qfile)
    print(f"Loaded {len(questions)} questions from {qfile}")

    if dry:
        active = [{"label": "Fake Model A", "skill": 0.8}, {"label": "Fake Model B", "skill": 0.55}]
    else:
        active = [m for m in MODELS if m.get("enabled", True)
                  and (m["key_env"] is None or os.environ.get(m["key_env"]))]
        if not active:
            sys.exit("No models set up. Set an API key (see top of script), or try --dry-run.")
    print("Models:", ", ".join(m["label"] for m in active))

    import threading
    from concurrent.futures import ThreadPoolExecutor
    lock = threading.Lock()
    def say(msg):
        with lock:
            print(msg, flush=True)

    def run_one_model(cfg):
        out = []
        tag = cfg["label"][:22].ljust(22)
        for i, q in enumerate(questions, 1):
            if not dry:
                say(f"  {tag} Q{q['id']:<3} asking...")
            t0 = time.time()
            _LAST.usage = {}
            raw = fake_model(cfg, q) if dry else ask_model(cfg, build_prompt(q))
            secs = time.time() - t0
            u = _LAST.usage or {}
            if dry:   # made-up token counts so the Ethics tab can be tested offline
                rng = random.Random(f"{cfg['label']}-{q['id']}-tokens")
                u = {"prompt_tokens": 120, "completion_tokens": rng.randint(5, 600), "cost": 0}
            ptok = int(u.get("prompt_tokens") or 0)
            ctok = int(u.get("completion_tokens") or 0)
            rtok = int(((u.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0)
            wh, ml = estimate_resources(ptok, ctok) if (ptok or ctok) else (None, None)
            if i == 1 and raw.startswith("ERROR"):
                try:
                    detail = json.loads(raw.split(":", 1)[1])
                    detail = (detail[0] if isinstance(detail, list) else detail)["error"]["message"]
                except Exception:
                    detail = raw[:300]
                say(f"  {tag} SKIPPED: the API returned an error before answering anything.\n"
                    f"  {tag} Reason: {detail}")
                return []
            got = extract_letter(raw)
            ok = int(got == q["answer"])
            out.append({"id": q["id"], "model": cfg["label"], "category": q["category"],
                        "question": q["question"], "answer": q["answer"],
                        "model_answer": got, "raw_response": raw, "correct": ok,
                        "prompt_tokens": ptok, "completion_tokens": ctok, "reasoning_tokens": rtok,
                        "cost_usd": u.get("cost"), "seconds": round(secs, 2),
                        "energy_wh": None if wh is None else round(wh, 5),
                        "water_ml": None if ml is None else round(ml, 5)})
            say(f"  {tag} Q{q['id']:<3} {'correct  ' if ok else 'INCORRECT'} answered {got or '?'} "
                f"(key {q['answer']})  {secs:4.1f}s" + ("" if got else f"  raw: {raw[:40]!r}"))
            if not dry:
                time.sleep(SECONDS_BETWEEN_CALLS)
        say(f"  {tag} DONE")
        return out

    print(f"\nAsking all {len(active)} models at the same time "
          f"({len(questions)} questions each). Press Ctrl+C to stop.\n")
    with ThreadPoolExecutor(max_workers=len(active)) as pool:
        per_model = list(pool.map(run_one_model, active))
    results = [r for model_results in per_model for r in model_results]

    models = [m["label"] for m in active if any(r["model"] == m["label"] for r in results)]
    if not results:
        sys.exit("\nNo model answered any questions, so no report was made. See the error above.")
    stats = analyze(results, models)
    print_report(stats)

    os.makedirs("reports", exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    tag = "_DRYRUN" if dry else ""
    with open(f"reports/graded_questions_{stamp}{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "id", "category", "question", "correct_answer", "ai_answer",
                    "result", "raw_response", "prompt_tokens", "completion_tokens",
                    "reasoning_tokens", "cost_usd", "seconds", "energy_wh", "water_ml"])
        for r in results:
            w.writerow([r["model"], r["id"], r["category"], r["question"], r["answer"],
                        r["model_answer"] or "none", "CORRECT" if r["correct"] else "INCORRECT",
                        r["raw_response"], r["prompt_tokens"], r["completion_tokens"],
                        r["reasoning_tokens"], "" if r["cost_usd"] is None else r["cost_usd"],
                        r["seconds"], "" if r["energy_wh"] is None else r["energy_wh"],
                        "" if r["water_ml"] is None else r["water_ml"]])
    # Energy, water, tokens and cost per AI (estimates; method at the top of this script)
    with open(f"reports/resources_{stamp}{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "prompt_tokens", "completion_tokens", "reasoning_tokens",
                    "cost_usd", "seconds", "energy_wh_est", "water_ml_est"])
        print("\nEstimated energy and water (see the method at the top of mc_benchmark.py):")
        for m in models:
            rs = [r for r in results if r["model"] == m]
            tot = lambda k: sum((r[k] or 0) for r in rs)
            w.writerow([m, tot("prompt_tokens"), tot("completion_tokens"), tot("reasoning_tokens"),
                        round(tot("cost_usd"), 6), round(tot("seconds"), 1),
                        round(tot("energy_wh"), 4), round(tot("water_ml"), 4)])
            print(f"  {m:<32} {tot('energy_wh'):7.3f} Wh   {tot('water_ml'):7.3f} mL water   "
                  f"{tot('completion_tokens'):>6} output tokens")
    with open(f"reports/summary_{stamp}{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        cats = sorted({r["category"] for r in results})
        w.writerow(["model", "final_grade_%", "letter", "correct", "total", "best_category",
                    "worst_category"] + [f"{c}_%" for c in cats])
        for m, s in stats.items():
            w.writerow([m, f"{s['pct']:.1f}", s["grade"], s["correct"], s["total"],
                        " / ".join(s["best"]), " / ".join(s["worst"])] +
                       [f"{s['cats'][c][2]:.1f}" if c in s["cats"] else "" for c in cats])
    if not dry:
        export_test_answers(results, questions)
        print("  TEST_ANSWERS/ updated (answers organized by AI)")
    html_path = f"reports/report_{stamp}{tag}.html"
    write_html(html_path, stats, questions, results, stamp.replace("_", " "))
    print(f"\nSaved to the reports folder:\n  {html_path}  (open this in your browser)"
          f"\n  graded_questions_{stamp}{tag}.csv\n  summary_{stamp}{tag}.csv")

if __name__ == "__main__":
    main()
