"""
Full RAG + Model accuracy test suite (query variant).

Run:
    cd backend
    python test_rag_full.py

Requirements:
    - Backend running at http://localhost:8000
    - Empty or fresh finance_memory/ (tests seed their own data)
    - pip install requests
"""

import requests
import json
import time
import sys
import os
import random
from collections import Counter
from typing import Optional

BASE = os.getenv("BASE_URL", "http://localhost:8000")
PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "
UNSTABLE = "🟡"

results = {"pass": 0, "fail": 0, "warn": 0, "unstable": 0}
saved_ids = []  # track all saved IDs for cleanup
RNG = random.Random(42)


# ── Helpers ────────────────────────────────────────────────────────────────────


def debug(message: str) -> dict:
    r = requests.post(f"{BASE}/debug", json={"message": message}, timeout=30)
    assert r.status_code == 200, f"debug failed: {r.text}"
    payload = r.json()

    # Top-k debug logging for RAG analysis
    top_k = (payload.get("rag_top_candidates") or [])[:3]
    if top_k:
        merchants = [str(c.get("merchant") or "None") for c in top_k]
        scores = [str(c.get("similarity") or 0.0) for c in top_k]
        print(
            f"    [RAG TOP3] query='{message}' | merchants={merchants} | scores={scores}"
        )

    return payload


def chat(message: str, session_data=None, followup_field=None) -> dict:
    payload = {"message": message}
    if session_data:
        payload["session_data"] = session_data
    if followup_field:
        payload["followup_field"] = followup_field
    r = requests.post(f"{BASE}/chat", json=payload, timeout=30)
    assert r.status_code == 200, f"chat failed: {r.text}"
    return r.json()


def save(entry: dict) -> str:
    r = requests.post(f"{BASE}/save", json={"entry": entry}, timeout=10)
    assert r.status_code == 200
    tx_id = r.json().get("id", "")
    if tx_id:
        saved_ids.append(tx_id)
    return tx_id


def delete(tx_id: str):
    requests.delete(f"{BASE}/transactions/{tx_id}", timeout=10)


def seed_via_chat(message: str) -> str:
    """Save a transaction via chat flow — fully realistic path."""
    # Step 1: send message
    resp = chat(message)
    session = resp.get("extracted", {})
    session["original_query"] = message

    # Step 2: answer any mandatory followups (shouldn't need any with complete inputs)
    for _ in range(5):
        if resp.get("status") == "complete":
            break
        if resp.get("status") == "followup" and not resp.get("is_optional_followup"):
            print(
                f"    ⚠️  Unexpected mandatory followup for: '{message}' → {resp.get('question')}"
            )
            return ""
        if resp.get("status") == "followup" and resp.get("is_optional_followup"):
            # Skip all optional followups during seeding
            field = resp.get("followup_field")
            session = resp.get("extracted", session)
            resp = chat("skip", session_data=session, followup_field=field)
            session = resp.get("extracted", session)

    # Save
    entry = resp.get("extracted", session)
    if not entry.get("items"):
        return ""
    r = requests.post(f"{BASE}/save", json={"entry": entry}, timeout=10)
    if r.status_code == 200:
        tx_id = r.json().get("id", "")
        if tx_id:
            saved_ids.append(tx_id)
        return tx_id
    return ""


def get_extracted(message: str) -> dict:
    d = debug(message)
    return d.get("normalized_output") or {}


def get_item(extracted: dict) -> dict:
    return (extracted.get("items") or [{}])[0]


def get_rag_chip(message: str) -> tuple[Optional[str], Optional[str]]:
    """Returns (merchant_chip, payment_chip) from RAG assumptions."""
    d = debug(message)
    assumptions = d.get("rag_assumptions") or {}
    return assumptions.get("merchant"), assumptions.get("payment_method")


def get_chat_chip(message: str) -> tuple[Optional[str], Optional[str]]:
    """Get what chip the chat UI would show for merchant and payment."""
    resp = chat(message)
    merchant_chip = None
    payment_chip = None

    # Walk through the optional followup sequence
    session = resp.get("extracted", {})
    session["original_query"] = message

    for _ in range(6):
        if resp.get("status") == "complete":
            break
        if resp.get("status") != "followup":
            break
        if not resp.get("is_optional_followup"):
            break

        field = resp.get("followup_field")
        assumption = resp.get("assumption_value")

        if field == "merchant" and assumption:
            merchant_chip = assumption
        elif field == "payment_method" and assumption:
            payment_chip = assumption

        session = resp.get("extracted", session)
        # Skip to next field
        resp = chat("skip", session_data=session, followup_field=field)
        session = resp.get("extracted", session)

    return merchant_chip, payment_chip


def check(
    test_id: str,
    description: str,
    passed: bool,
    detail: str = "",
    warn_only=False,
    unstable_only=False,
):
    global results
    if passed:
        results["pass"] += 1
        print(f"  {PASS} {test_id}: {description}")
    elif unstable_only:
        results["unstable"] += 1
        print(f"  {UNSTABLE} {test_id}: {description} {detail}")
    elif warn_only:
        results["warn"] += 1
        print(f"  {WARN} {test_id}: {description} {detail}")
    else:
        results["fail"] += 1
        print(f"  {FAIL} {test_id}: {description} {detail}")


def section(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def subsection(title: str):
    print(f"\n  ── {title} ──")


# ══════════════════════════════════════════════════════════════════════════════
# SETUP — check backend is up
# ══════════════════════════════════════════════════════════════════════════════

print("\n🔍 Checking backend...")
try:
    r = requests.get(f"{BASE}/health", timeout=5)
    assert r.status_code == 200
    health = r.json()
    model = health.get("model", "unknown")
    backend_runtime = health.get("backend", "unknown")
    model_path = health.get("model_path", "unknown")
    print(f"   Backend OK | Model: {model}")
    print(f"   Runtime: {backend_runtime} | Model Path: {model_path}")
except Exception as e:
    print(f"   ❌ Backend not reachable: {e}")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — MODEL EXTRACTION ACCURACY
# Tests what Qwen returns before any RAG.
# If these fail → fine-tuning problem.
# ══════════════════════════════════════════════════════════════════════════════

section("PHASE 1 — MODEL EXTRACTION ACCURACY")

subsection("1A: Intent detection — expense")
expense_inputs = [
    ("E1", "spent 300 on breakfast today"),
    ("E2", "paid 500 for dinner today"),
    ("E3", "bought medicine 300 today"),
    ("E4", "filled petrol 1500 today"),
    ("E5", "bought groceries 600 today"),
    ("E6", "Netflix subscription 649 today"),
    ("E7", "gym membership 1000 today"),
    ("E8", "Uber 200 today"),
]
for tid, inp in expense_inputs:
    e = get_extracted(inp)
    check(tid, inp, e.get("intent") == "expense", f"→ got intent='{e.get('intent')}'")

subsection("1B: Intent detection — income")
income_inputs = [
    ("I1", "got salary 45000 today"),
    ("I2", "received salary 50000 today"),
    ("I3", "salary credited 72000 today"),
    ("I4", "got bonus 10000 today"),
    ("I5", "received freelance payment 8000 today"),
    ("I6", "Rahul paid me back 2000 today"),
    ("I7", "Amazon refunded 499 today"),
    ("I8", "got cashback 150 today"),
    ("I9", "interest credited 500 today"),
    ("I10", "rent received 12000 today"),
]
for tid, inp in income_inputs:
    e = get_extracted(inp)
    check(tid, inp, e.get("intent") == "income", f"→ got intent='{e.get('intent')}'")

subsection("1C: Must NOT be income (false positive guard)")
not_income = [
    ("NI1", "paid salary to cook 3000 today", "expense"),
    ("NI2", "gave dad 5000 today", "expense"),
    ("NI3", "sent Ravi 2000 via UPI today", "expense"),
]
for tid, inp, expected in not_income:
    e = get_extracted(inp)
    check(tid, inp, e.get("intent") == expected, f"→ got intent='{e.get('intent')}'")

subsection("1D: Category extraction — must be correct")
category_cases = [
    ("C1", "bought tablets 200 today", "Health"),
    ("C2", "bought medicine 300 today", "Health"),
    ("C3", "paid gym fees 1000 today", "Health"),
    ("C4", "bought supplements 400 today", "Health"),
    ("C5", "filled petrol 1500 today", "Transport"),
    ("C6", "filled diesel 2000 today", "Transport"),
    ("C7", "Uber cab 250 today", "Transport"),
    ("C8", "bought vegetables 400 today", "Groceries"),
    ("C9", "bought rice 600 today", "Groceries"),
    ("C10", "bought groceries 500 today", "Groceries"),
    ("C11", "Netflix subscription 649 today", "Entertainment"),
    ("C12", "bought subscription 799 today", "Entertainment"),
    ("C13", "Spotify 119 today", "Entertainment"),
    ("C14", "ordered breakfast from dominos 300 today", "Food"),
    ("C15", "coffee at starbucks 150 today", "Food"),
    ("C16", "electricity bill 1200 today", "Bills"),
    ("C17", "Jio recharge 239 today", "Bills"),
]
for tid, inp, expected_cat in category_cases:
    e = get_extracted(inp)
    item = get_item(e)
    got = str(item.get("category") or "").strip()
    check(
        tid,
        f"{inp} → category={expected_cat}",
        got.lower() == expected_cat.lower(),
        f"got '{got}'",
    )

subsection("1E: payment_method must be null when NOT mentioned")
no_payment_cases = [
    ("P1", "bought medicine 300 today"),
    ("P2", "filled petrol 1500 today"),
    ("P3", "bought groceries 500 today"),
    ("P4", "got salary 45000 today"),
    ("P5", "Netflix subscription 649 today"),
    ("P6", "gym membership 1000 today"),
]
for tid, inp in no_payment_cases:
    e = get_extracted(inp)
    item = get_item(e)
    pm = item.get("payment_method")
    check(tid, f"{inp} → payment=null", pm is None, f"got payment='{pm}'")

subsection("1F: payment_method correctly extracted when mentioned")
payment_cases = [
    ("PM1", "bought medicine 300 today via UPI", "UPI"),
    ("PM2", "filled petrol 1500 today cash", "Cash"),
    ("PM3", "Netflix 649 today card", "Card"),
    ("PM4", "received salary 45000 via bank transfer today", "Bank Transfer"),
    ("PM5", "paid 500 via gpay today", "UPI"),
]
for tid, inp, expected_pm in payment_cases:
    e = get_extracted(inp)
    item = get_item(e)
    got = str(item.get("payment_method") or "")
    check(
        tid,
        f"{inp} → payment={expected_pm}",
        got.lower() == expected_pm.lower(),
        f"got '{got}'",
    )

subsection("1G: amount must be null when NOT in input")
no_amount_cases = [
    ("A1", "bought medicine today"),
    ("A2", "got salary today"),
    ("A3", "filled petrol today"),
]
for tid, inp in no_amount_cases:
    e = get_extracted(inp)
    item = get_item(e)
    amt = item.get("amount")
    check(tid, f"{inp} → amount=null", amt is None, f"got amount={amt}")

subsection("1H: datetime must be null when NOT in input")
no_datetime_cases = [
    ("D1", "bought medicine 300"),
    ("D2", "got salary 45000"),
    ("D3", "filled petrol 1500"),
]
for tid, inp in no_datetime_cases:
    e = get_extracted(inp)
    item = get_item(e)
    dt = item.get("datetime")
    check(tid, f"{inp} → datetime=null", dt is None, f"got datetime='{dt}'")

subsection("1I: income source extraction")
source_cases = [
    ("S1", "got salary 45000 today", "Salary"),
    ("S2", "received freelance payment 8000 today", "Freelance"),
    ("S3", "got bonus 10000 today", "Bonus"),
    ("S4", "interest credited 500 today", "Interest"),
    ("S5", "got cashback 200 today", "Cashback"),
    ("S6", "received refund 499 from Amazon today", "Refund"),
    ("S7", "got rent 12000 from tenant today", "Rent"),
]
for tid, inp, expected_src in source_cases:
    e = get_extracted(inp)
    item = get_item(e)
    got = str(item.get("source") or item.get("category") or "")
    check(
        tid,
        f"{inp} → source={expected_src}",
        got.lower() == expected_src.lower(),
        f"got '{got}'",
    )

subsection("1J: income payer extraction")
payer_cases = [
    ("PY1", "salary credited from TCS 65000 today", "TCS"),
    ("PY2", "received 50000 salary from Infosys today", "Infosys"),
    ("PY3", "Rahul paid me back 2000 today", "Rahul"),
    ("PY4", "got 5000 from Priya today", "Priya"),
    ("PY5", "Amazon refunded 499 today", "Amazon"),
]
for tid, inp, expected_payer in payer_cases:
    e = get_extracted(inp)
    item = get_item(e)
    got = str(item.get("payer") or "")
    check(
        tid,
        f"{inp} → payer={expected_payer}",
        got.lower() == expected_payer.lower(),
        f"got '{got}'",
    )


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — RAG ACCURACY (seed data first)
# Tests whether correct merchants/payments are suggested.
# If Phase 1 passes but Phase 2 fails → RAG logic problem.
# ══════════════════════════════════════════════════════════════════════════════

section("PHASE 2 — RAG ACCURACY")
print("  Dynamic large-scale seeding enabled")

CATEGORIES = [
    "Food",
    "Groceries",
    "Health",
    "Transport",
    "Bills",
    "Entertainment",
    "Income",
]

CATEGORY_SPECS = {
    "Food": {
        "items": ["breakfast", "dinner", "coffee", "pizza", "burger"],
        "merchants": ["swiggy", "zomato", "dominos", "starbucks"],
        "payments": ["UPI", "Cash", "Card"],
        "phrases": [
            "spent {amt} on {item} at {merchant} today via {pm}",
            "ordered {item} from {merchant} {amt} today",
            "had {item} at {merchant} {amt} today using {pm}",
        ],
    },
    "Groceries": {
        "items": ["vegetables", "rice", "fruits", "milk", "groceries"],
        "merchants": ["bigbazaar", "dmart", "reliance fresh"],
        "payments": ["Cash", "UPI", "Card"],
        "phrases": [
            "bought {item} from {merchant} for {amt} today via {pm}",
            "purchased {item} {amt} today at {merchant}",
            "got {item} from {merchant} {amt} today",
        ],
    },
    "Health": {
        "items": ["medicine", "tablets", "supplements", "gym fees", "checkup"],
        "merchants": ["apollo pharmacy", "medcart", "gold gym"],
        "payments": ["Card", "Cash", "UPI"],
        "phrases": [
            "bought {item} at {merchant} {amt} today via {pm}",
            "paid {amt} for {item} at {merchant} today",
            "got {item} from {merchant} for {amt} today",
        ],
    },
    "Transport": {
        "items": ["petrol", "diesel", "cab", "metro", "parking"],
        "merchants": ["hp pump", "indian oil", "uber", "ola"],
        "payments": ["Cash", "UPI", "Card"],
        "phrases": [
            "filled {item} at {merchant} for {amt} today via {pm}",
            "spent {amt} on {item} today",
            "paid {amt} for {item} using {pm} today",
        ],
    },
    "Bills": {
        "items": ["electricity bill", "wifi bill", "mobile recharge", "water bill"],
        "merchants": ["jio", "airtel", "bescom", "act fibernet"],
        "payments": ["UPI", "Card", "Bank Transfer"],
        "phrases": [
            "paid {item} of {amt} today via {pm}",
            "{item} {amt} paid today",
            "cleared {item} {amt} today using {pm}",
        ],
    },
    "Entertainment": {
        "items": ["netflix subscription", "spotify", "movie tickets", "gaming"],
        "merchants": ["netflix", "spotify", "pvr", "inox"],
        "payments": ["Card", "UPI", "Cash"],
        "phrases": [
            "paid {amt} for {item} today via {pm}",
            "{item} {amt} today",
            "spent {amt} on {item} at {merchant} today",
        ],
    },
    "Income": {
        "items": ["salary", "bonus", "freelance payment", "refund", "cashback"],
        "merchants": ["TCS", "Infosys", "Amazon", "client", "bank"],
        "payments": ["Bank Transfer", "UPI"],
        "phrases": [
            "received {item} {amt} today via {pm}",
            "got {item} of {amt} today",
            "{item} {amt} credited today through {pm}",
        ],
    },
}

MERCHANT_TO_CATEGORY = {}
for _cat, _spec in CATEGORY_SPECS.items():
    if _cat == "Income":
        continue
    for _m in _spec["merchants"]:
        MERCHANT_TO_CATEGORY[_m.lower()] = _cat


def clamp_seed_count(value: int) -> int:
    return max(20, min(50, value))


def generate_seeds(category: str, count: int) -> list[str]:
    spec = CATEGORY_SPECS[category]
    seeds = []
    for _ in range(count):
        amt = RNG.randint(100, 2000)
        phrase = RNG.choice(spec["phrases"])
        item = RNG.choice(spec["items"])
        merchant = RNG.choice(spec["merchants"])
        pm = RNG.choice(spec["payments"])
        msg = phrase.format(amt=amt, item=item, merchant=merchant, pm=pm)
        seeds.append(msg)
    return seeds


def infer_merchant_category(merchant: Optional[str]) -> Optional[str]:
    if not merchant:
        return None
    m = str(merchant).lower().strip()
    for key, cat in MERCHANT_TO_CATEGORY.items():
        if key in m:
            return cat
    return None


def generate_rag_queries(category: str, count: int) -> list[str]:
    spec = CATEGORY_SPECS[category]
    queries = []
    noisy_prefix = ["", "umm ", "i think ", "maybe ", "like "]
    for _ in range(count):
        amt = RNG.randint(120, 1800)
        item = RNG.choice(spec["items"])
        prefix = RNG.choice(noisy_prefix)
        if category == "Income":
            template = RNG.choice(
                [
                    "{p}received {item} {amt} today",
                    "{p}got {item} {amt} today",
                    "{p}{item} {amt} credited today",
                ]
            )
        else:
            template = RNG.choice(
                [
                    "{p}spent {amt} on {item} today",
                    "{p}paid {amt} for {item} today",
                    "{p}{item} {amt} today",
                ]
            )
        queries.append(template.format(p=prefix, item=item, amt=amt))
    return queries


subsection("2A: Large-scale dynamic seeding (20-50 per category)")
seed_counts = {
    cat: clamp_seed_count(int(os.getenv(f"SEED_{cat.upper()}", "20")))
    for cat in CATEGORIES
}

print(f"  Seed plan: {seed_counts}")
seed_batches = {}
for cat in CATEGORIES:
    batch = generate_seeds(cat, seed_counts[cat])
    seed_batches[cat] = batch
    print(f"    {cat}: generating {len(batch)} seeds")
    for msg in batch:
        tx_id = seed_via_chat(msg)
        if tx_id:
            saved_ids.append(tx_id)

time.sleep(1.0)

section("RAG STRESS TEST")

subsection("2B: 200+ diversified RAG queries")
query_count_per_category = int(os.getenv("RAG_QUERY_COUNT_PER_CATEGORY", "30"))
all_rag_queries = []
for cat in CATEGORIES:
    for q in generate_rag_queries(cat, query_count_per_category):
        all_rag_queries.append((cat, q))

for idx, (cat, q) in enumerate(all_rag_queries, start=1):
    merchant, payment = get_rag_chip(q)
    if cat == "Income":
        got = str(payment or "").lower()
        ok = any(pm in got for pm in ["bank transfer", "upi"])
        check(
            f"Q{idx}",
            f"{cat} query payment suggestion",
            ok,
            f"query='{q}', payment='{payment}'",
            warn_only=True,
        )
    else:
        predicted_cat = infer_merchant_category(merchant)
        ok = predicted_cat == cat if merchant else False
        check(
            f"Q{idx}",
            f"{cat} query merchant category match",
            ok,
            f"query='{q}', merchant='{merchant}', predicted={predicted_cat}",
            warn_only=True,
        )

subsection("2C: RAG DOMINANCE TEST")
dominance_plan = {"Food": 50, "Groceries": 5, "Health": 5, "Transport": 5}
for cat, cnt in dominance_plan.items():
    for msg in generate_seeds(cat, cnt):
        tx_id = seed_via_chat(msg)
        if tx_id:
            saved_ids.append(tx_id)

dominance_queries = [
    ("DT1", "bought vegetables 300 today"),
    ("DT2", "bought medicine 200 today"),
]
food_merchants = {"swiggy", "zomato", "dominos", "starbucks"}
for tid, q in dominance_queries:
    merchant, _ = get_rag_chip(q)
    got = str(merchant or "").lower()
    leak = any(m in got for m in food_merchants)
    check(
        tid,
        f"{q} should not leak food merchants",
        not leak,
        f"got merchant='{merchant}'",
    )

subsection("2D: Merchant dominance test")
for _ in range(40):
    tx_id = seed_via_chat(
        f"ordered dinner from swiggy {RNG.randint(200, 900)} today via UPI"
    )
    if tx_id:
        saved_ids.append(tx_id)
for _ in range(5):
    tx_id = seed_via_chat(
        f"ordered dinner from zomato {RNG.randint(200, 900)} today via UPI"
    )
    if tx_id:
        saved_ids.append(tx_id)

merchant_runs = []
for _ in range(20):
    merchant, _ = get_rag_chip("ordered dinner 400 today")
    merchant_runs.append(str(merchant or "").lower())

counts = Counter(m for m in merchant_runs if m)
top_merchant = counts.most_common(1)[0][0] if counts else ""
top_share = (counts[top_merchant] / len(merchant_runs)) if counts else 0.0
has_zomato = any("zomato" in m for m in merchant_runs)

check(
    "MD1",
    "Dominance should not exceed 80% for one merchant",
    top_share <= 0.80,
    f"top_merchant='{top_merchant}', share={top_share:.2f}, runs={merchant_runs}",
    unstable_only=top_share > 0.80,
)
check(
    "MD2",
    "Dinner retrieval should still allow zomato",
    has_zomato,
    f"runs={merchant_runs}",
    unstable_only=not has_zomato,
)

subsection("2E: Noise and realistic queries")
noise_queries = [
    ("NQ1", "umm spent like 300 on dinner today"),
    ("NQ2", "i think i paid 200 for coffee"),
    ("NQ3", "maybe 500 for groceries today"),
    ("NQ4", "uhh paid around 350 for breakfast"),
]
for tid, q in noise_queries:
    e = get_extracted(q)
    merchant, _ = get_rag_chip(q)
    intent_ok = e.get("intent") == "expense"
    merchant_cat = infer_merchant_category(merchant)
    check(
        tid,
        "Noisy query should remain expense and in-domain",
        intent_ok and merchant_cat in {"Food", "Groceries"},
        f"query='{q}', intent='{e.get('intent')}', merchant='{merchant}', cat='{merchant_cat}'",
        warn_only=True,
    )

subsection("2F: Unseen merchant generalization")
unseen_queries = [
    ("UG1", "had lunch from subway 300 today", "Food"),
    ("UG2", "bought medicine from local chemist 200 today", "Health"),
]
for tid, q, expected_cat in unseen_queries:
    e = get_extracted(q)
    item = get_item(e)
    cat = str(item.get("category") or "")
    merchant, _ = get_rag_chip(q)
    merchant_cat = infer_merchant_category(merchant)
    no_hallucination = merchant_cat in {None, expected_cat}
    check(
        tid,
        f"Unseen merchant should fallback to {expected_cat} category",
        cat.lower() == expected_cat.lower() and no_hallucination,
        f"category='{cat}', rag_merchant='{merchant}', rag_category='{merchant_cat}'",
        warn_only=True,
    )

subsection("2G: Multiple-run consistency test")
consistency_runs = []
for _ in range(5):
    merchant, _ = get_rag_chip("ordered dinner 400 today")
    consistency_runs.append(str(merchant or "").lower())

run_counts = Counter(m for m in consistency_runs if m)
top_freq = run_counts.most_common(1)[0][1] if run_counts else 0
too_unstable = top_freq < 3
check(
    "CS1",
    "Repeated dinner query should be reasonably consistent",
    not too_unstable,
    f"runs={consistency_runs}",
    unstable_only=too_unstable,
)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — CHAT FLOW (end-to-end followup sequence)
# Tests the full user experience including followup questions.
# ══════════════════════════════════════════════════════════════════════════════

section("PHASE 3 — CHAT FLOW (end-to-end)")

subsection("3A: Mandatory fields triggering followup")


def test_mandatory_followup(tid: str, message: str, expected_missing: str):
    resp = chat(message)
    field = resp.get("followup_field")
    is_optional = resp.get("is_optional_followup", True)
    check(
        tid,
        f"'{message}' asks for '{expected_missing}'",
        field == expected_missing and not is_optional,
        f"got followup_field='{field}', is_optional={is_optional}",
    )


test_mandatory_followup("FL1", "bought medicine at apollo", "amount")
test_mandatory_followup("FL2", "bought medicine 300", "datetime")
test_mandatory_followup("FL3", "spent on groceries today", "amount")

subsection("3B: Optional followup shows RAG chip (context-aware)")


def test_optional_chip(
    tid: str,
    message: str,
    expected_field: str,
    expected_chip_contains: str,
):
    """Checks that the expected optional followup field shows the right chip."""
    resp = chat(message)
    session = resp.get("extracted", {})
    session["original_query"] = message

    # Walk through to optional followup
    for _ in range(6):
        if resp.get("status") == "complete":
            break

        if (
            resp.get("status") == "followup"
            and resp.get("is_optional_followup")
            and resp.get("followup_field") == expected_field
        ):
            chip = resp.get("assumption_value")
            got = str(chip or "").lower()
            check(
                tid,
                f"'{message}' → {expected_field} chip contains '{expected_chip_contains}'",
                expected_chip_contains.lower() in got,
                f"got chip='{chip}'",
            )
            return

        if resp.get("status") == "followup" and not resp.get("is_optional_followup"):
            break

        field = resp.get("followup_field")
        if not field:
            break
        session = resp.get("extracted", session)
        resp = chat("skip", session_data=session, followup_field=field)
        session = resp.get("extracted", session)

    check(
        tid,
        f"'{message}' → {expected_field} chip shown",
        False,
        "(chip never appeared)",
    )


test_optional_chip(
    "OC1", "paid 300 for breakfast at dominos today", "payment_method", "UPI"
)
test_optional_chip(
    "OC2", "purchased vegetables at bigbazaar 400 today", "payment_method", "Cash"
)
test_optional_chip(
    "OC3", "petrol filled at hp pump 1500 today", "payment_method", "Cash"
)
test_optional_chip("OC4", "paid 300 for breakfast today", "merchant", "dominos")

subsection("3C: Yes/No chip response")


def test_chip_yes(
    tid: str, message: str, expected_field: str, expected_value_contains: str
):
    """Test that accepting a chip stores the correct value."""
    resp = chat(message)
    session = resp.get("extracted", {})
    session["original_query"] = message

    for _ in range(8):
        if resp.get("status") == "complete":
            break
        field = resp.get("followup_field")
        if field == expected_field and resp.get("assumption_value"):
            # Accept the chip
            session = resp.get("extracted", session)
            resp = chat("yes", session_data=session, followup_field=field)
            session = resp.get("extracted", session)
            # Check value was set
            item = (session.get("items") or [{}])[0]
            got = str(item.get(expected_field) or "").lower()
            check(
                tid,
                f"Accepting chip → {expected_field}='{expected_value_contains}'",
                expected_value_contains.lower() in got,
                f"got {expected_field}='{got}'",
            )
            return
        session = resp.get("extracted", session)
        resp = chat("skip", session_data=session, followup_field=field)
        session = resp.get("extracted", session)

    check(tid, f"Chip acceptance test for '{message}'", False, "(chip never appeared)")


test_chip_yes("CY1", "paid 300 for breakfast at dominos today", "payment_method", "UPI")
test_chip_yes(
    "CY2", "purchased vegetables at bigbazaar 400 today", "payment_method", "Cash"
)

subsection("3D: Reject chip → plain followup")


def test_chip_no(tid: str, message: str, reject_field: str):
    """Test that rejecting a chip gives a plain text followup."""
    resp = chat(message)
    session = resp.get("extracted", {})
    session["original_query"] = message

    for _ in range(8):
        if resp.get("status") == "complete":
            break
        field = resp.get("followup_field")
        if field == reject_field and resp.get("assumption_value"):
            session = resp.get("extracted", session)
            resp = chat("no", session_data=session, followup_field=field)
            # After rejection, should ask again with no chip
            chip = resp.get("assumption_value")
            check(
                tid,
                f"Rejecting chip → plain followup (no chip)",
                chip is None,
                f"still got chip='{chip}'",
            )
            return
        session = resp.get("extracted", session)
        resp = chat("skip", session_data=session, followup_field=field)
        session = resp.get("extracted", session)

    check(tid, f"Chip rejection test for '{message}'", False, "(chip never appeared)")


test_chip_no("CR1", "paid 300 for breakfast at dominos today", "payment_method")

subsection("3E: Skip optional followup")


def test_skip(tid: str, message: str, skip_field: str):
    resp = chat(message)
    session = resp.get("extracted", {})
    session["original_query"] = message

    for _ in range(8):
        if resp.get("status") == "complete":
            break
        field = resp.get("followup_field")
        if field == skip_field:
            session = resp.get("extracted", session)
            resp = chat("skip", session_data=session, followup_field=field)
            session = resp.get("extracted", session)
            item = (session.get("items") or [{}])[0]
            got = item.get(skip_field)
            check(
                tid,
                f"Skipping '{skip_field}' → field=null",
                got is None,
                f"got {skip_field}='{got}'",
            )
            return
        session = resp.get("extracted", session)
        resp = chat("skip", session_data=session, followup_field=field)
        session = resp.get("extracted", session)

    check(tid, f"Skip test for field '{skip_field}'", False, "(field never appeared)")


test_skip("SK1", "spent 300 on breakfast today", "merchant")
test_skip("SK2", "bought medicine 300 today", "merchant")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — EDGE CASES
# ══════════════════════════════════════════════════════════════════════════════

section("PHASE 4 — EDGE CASES")

subsection("4A: Typos and alternate phrasing")
typo_cases = [
    ("TY1", "recieved salary 45000 today", "income"),  # typo: recieved
    ("TY2", "payed 300 at zomato today", "expense"),  # typo: payed
    ("TY3", "bout shoes 2000 today", "expense"),  # typo: bout
    ("TY4", "spent 300 on grocieries today", "expense"),  # typo: grocieries
]
for tid, inp, expected_intent in typo_cases:
    e = get_extracted(inp)
    check(
        tid,
        f"Typo: '{inp}' → intent={expected_intent}",
        e.get("intent") == expected_intent,
        f"got '{e.get('intent')}'",
    )

subsection("4B: Amount formats")
amount_cases = [
    ("AM1", "spent 1.5k on shopping today", 1500.0),
    ("AM2", "spent 1 lakh on car today", 100000.0),
    ("AM3", "paid Rs. 500 for food today", 500.0),
    ("AM4", "bought ₹300 medicine today", 300.0),
    ("AM5", "spent 1,500 on clothes today", 1500.0),
]
for tid, inp, expected_amt in amount_cases:
    e = get_extracted(inp)
    item = get_item(e)
    got = item.get("amount")
    check(
        tid,
        f"Amount format: '{inp}' → {expected_amt}",
        got == expected_amt,
        f"got {got}",
    )

subsection("4C: Multi-item input (model should extract first item)")
multi_tests = [
    ("MI1", "Zomato pizza 400 and Uber 180 today", "expense"),
    ("MI2", "bought groceries 500 and paid electricity 1200 today", "expense"),
]
for tid, inp, expected_intent in multi_tests:
    e = get_extracted(inp)
    check(
        tid,
        f"Multi-item: '{inp}' → intent={expected_intent}",
        e.get("intent") == expected_intent,
        f"got '{e.get('intent')}'",
    )

subsection("4D: Hindi/mixed inputs")
hindi_cases = [
    ("HI1", "aaj 500 rupaye khane pe kharch kiye", "expense"),
    ("HI2", "dawa liya 450 ka today", "expense"),
    ("HI3", "petrol bharwaya 500 mein today", "expense"),
    ("HI4", "got salary aaj 45000", "income"),
]
for tid, inp, expected_intent in hindi_cases:
    e = get_extracted(inp)
    check(
        tid,
        f"Hindi: '{inp}' → intent={expected_intent}",
        e.get("intent") == expected_intent,
        f"got '{e.get('intent')}'",
        warn_only=True,
    )

subsection("4E: Unsupported inputs must NOT save")
unsupported = [
    ("UN1", "what is 2+2"),
    ("UN2", "hello"),
    ("UN3", "help me plan my budget"),
    ("UN4", "who is elon musk"),
    ("UN5", "100 rs"),  # just amount, no context
]
for tid, inp in unsupported:
    e = get_extracted(inp)
    intent = e.get("intent")
    check(
        tid,
        f"Unsupported: '{inp}' → not expense/income",
        intent not in {"expense", "income"},
        f"got intent='{intent}'",
        warn_only=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — EXTRA REGRESSION USE CASES
# Broader real-world chat/followup behavior and parser robustness checks.
# ══════════════════════════════════════════════════════════════════════════════

section("PHASE 5 — EXTRA REGRESSION USE CASES")

subsection("5A: RAG should stay in-domain (not exact-merchant brittle)")
rag_domain_cases = [
    (
        "RG1",
        "bought rice 450 today",
        ["hp pump", "dominos", "zomato", "gold gym"],
    ),
    (
        "RG2",
        "bought tablets 250 today",
        ["hp pump", "dominos", "zomato"],
    ),
]
for tid, inp, forbidden in rag_domain_cases:
    merchant, _ = get_rag_chip(inp)
    got = str(merchant or "").lower()
    ok = not any(token in got for token in forbidden)
    check(
        tid,
        f"'{inp}' → merchant suggestion should avoid unrelated domains",
        ok,
        f"got chip='{merchant}', forbidden={forbidden}",
    )

subsection("5B: Followup parser robustness (amount/datetime/payment normalization)")


def test_amount_followup_parse(tid: str):
    resp = chat("bought medicine at apollo today")
    if resp.get("followup_field") != "amount":
        check(
            tid,
            "Amount followup appears",
            False,
            f"got followup_field='{resp.get('followup_field')}'",
        )
        return
    session = resp.get("extracted", {})
    resp2 = chat("1.5k", session_data=session, followup_field="amount")
    item = (resp2.get("extracted", {}).get("items") or [{}])[0]
    amt = item.get("amount")
    check(tid, "Amount followup parses 1.5k → 1500", amt == 1500.0, f"got {amt}")


def test_datetime_followup_parse(tid: str):
    resp = chat("bought medicine 300 at apollo")
    if resp.get("followup_field") != "datetime":
        check(
            tid,
            "Datetime followup appears",
            False,
            f"got followup_field='{resp.get('followup_field')}'",
        )
        return
    session = resp.get("extracted", {})
    resp2 = chat("2 days ago", session_data=session, followup_field="datetime")
    item = (resp2.get("extracted", {}).get("items") or [{}])[0]
    dt = item.get("datetime")
    check(tid, "Datetime followup parses relative date", dt is not None, f"got '{dt}'")


def test_payment_normalization_followup(tid: str):
    resp = chat("spent 320 on breakfast today")
    session = resp.get("extracted", {})
    session["original_query"] = "spent 320 on breakfast today"

    for _ in range(8):
        if resp.get("status") == "complete":
            break
        field = resp.get("followup_field")
        if field == "payment_method":
            session = resp.get("extracted", session)
            resp2 = chat("gpay", session_data=session, followup_field="payment_method")
            item = (resp2.get("extracted", {}).get("items") or [{}])[0]
            pm = str(item.get("payment_method") or "")
            check(
                tid,
                "Payment followup normalizes gpay → UPI",
                pm.lower() == "upi",
                f"got '{pm}'",
            )
            return
        if not field:
            break
        session = resp.get("extracted", session)
        resp = chat("skip", session_data=session, followup_field=field)
        session = resp.get("extracted", session)

    check(
        tid,
        "Payment followup appeared for normalization test",
        False,
        "(payment followup never appeared)",
    )


test_amount_followup_parse("FR1")
test_datetime_followup_parse("FR2")
test_payment_normalization_followup("FR3")

subsection("5C: Fully-specified input should not trigger mandatory followup")
full_inputs = [
    ("FU1", "spent 450 on dinner at zomato today via UPI"),
    ("FU2", "received salary 55000 today via bank transfer from TCS"),
]
for tid, inp in full_inputs:
    resp = chat(inp)
    is_followup = resp.get("status") == "followup"
    is_optional = bool(resp.get("is_optional_followup")) if is_followup else False
    passed = (not is_followup) or is_optional
    check(
        tid,
        f"'{inp}' should not ask mandatory followup",
        passed,
        f"status='{resp.get('status')}', followup_field='{resp.get('followup_field')}', optional={resp.get('is_optional_followup')}",
    )


# ══════════════════════════════════════════════════════════════════════════════
# CLEANUP
# ══════════════════════════════════════════════════════════════════════════════

section("CLEANUP")
print(f"  Deleting {len(saved_ids)} seeded transactions...")
for tx_id in saved_ids:
    delete(tx_id)
print(f"  Done.")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════

total = results["pass"] + results["fail"] + results["warn"] + results["unstable"]
section("FINAL REPORT")
print(
    f"""
  Total tests : {total}
  ✅ Pass     : {results['pass']}
  ❌ Fail     : {results['fail']}
  ⚠️  Warn     : {results['warn']}
    🟡 Unstable : {results['unstable']}

    Score: {results['pass']}/{results['pass'] + results['fail'] + results['unstable']} 
                 ({results['pass']/(results['pass']+results['fail']+results['unstable'])*100:.1f}% pass rate, excluding warnings)
"""
)

if results["fail"] == 0:
    print("  🎉 All tests passed!")
elif results["fail"] <= 3:
    print("  ⚠️  Almost there — check the ❌ failures above.")
else:
    print("  🔧 Several failures — check which phase failed most.")

print(
    f"""
  Reading the results:
  - Phase 1 failures → model problem (fix fine-tuning)
  - Phase 2 failures → RAG problem (check embed text / thresholds)
  - Phase 3 failures → followup flow bug (check /chat route)
  - Phase 4 failures → edge case handling (check normalization)
"""
)
