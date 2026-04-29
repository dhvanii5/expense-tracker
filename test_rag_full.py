"""
Full RAG + Model accuracy test suite.

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
from typing import Optional

BASE = os.getenv("BASE_URL", "http://localhost:8000")
PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

results = {"pass": 0, "fail": 0, "warn": 0}
saved_ids = []  # track all saved IDs for cleanup


# ── Helpers ────────────────────────────────────────────────────────────────────


def debug(message: str) -> dict:
    r = requests.post(f"{BASE}/debug", json={"message": message}, timeout=30)
    assert r.status_code == 200, f"debug failed: {r.text}"
    return r.json()


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
    test_id: str, description: str, passed: bool, detail: str = "", warn_only=False
):
    global results
    if passed:
        results["pass"] += 1
        print(f"  {PASS} {test_id}: {description}")
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
    ("E1", "spent 300 on lunch today"),
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
    ("C14", "ordered lunch from swiggy 300 today", "Food"),
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
print("  Seeding 10 transactions...")

SEEDS = [
    "spent 300 on lunch at swiggy today via UPI",
    "spent 500 on dinner at zomato yesterday via UPI",
    "spent 150 on coffee at starbucks today via card",
    "bought vegetables at bigbazaar 400 today via cash",
    "bought rice at dmart 600 today via cash",
    "bought fruits at reliance fresh 200 today via cash",
    "bought medicine at apollo pharmacy 300 today via card",
    "paid gym fees at gold gym 1000 today via cash",
    "filled petrol at hp pump 1500 today via cash",
    "received salary 45000 today via bank transfer",
]

seed_ids = []
for i, seed in enumerate(SEEDS):
    tx_id = seed_via_chat(seed)
    if tx_id:
        seed_ids.append(tx_id)
        print(f"    ✓ Seed {i+1}: {seed[:50]}...")
    else:
        print(f"    ✗ Seed {i+1} FAILED: {seed[:50]}")

time.sleep(0.5)  # let chroma settle

subsection("2A: Food — item-specific merchant suggestion")
food_tests = [
    ("F1", "spent 200 on lunch today", "swiggy", ["zomato", "starbucks"]),
    ("F2", "spent 400 on dinner today", "zomato", ["swiggy", "starbucks"]),
    ("F3", "spent 100 on coffee today", "starbucks", ["swiggy", "zomato"]),
    (
        "F4",
        "ordered food 350 today",
        None,
        ["starbucks"],
    ),  # any food merchant ok, not starbucks
]
for tid, inp, expected, wrong in food_tests:
    merchant, _ = get_rag_chip(inp)
    got = str(merchant or "").lower()

    if expected is None:
        # Just check wrong ones not shown
        no_wrong = not any(w in got for w in wrong)
        check(tid, f"'{inp}' → not {wrong}", no_wrong, f"got chip='{merchant}'")
    else:
        correct = expected in got
        no_wrong = not any(w in got for w in wrong)
        check(
            tid,
            f"'{inp}' → chip={expected}",
            correct and no_wrong,
            f"got chip='{merchant}'",
        )

subsection("2B: Groceries — item-specific merchant suggestion")
grocery_tests = [
    ("G1", "spent 300 on vegetables today", "bigbazaar", ["dmart", "reliance"]),
    ("G2", "bought rice 500 today", "dmart", ["bigbazaar", "reliance"]),
    ("G3", "spent 150 on fruits today", "reliance", ["dmart", "bigbazaar"]),
    ("G4", "bought groceries 800 today", None, ["hp pump", "swiggy"]),  # any grocery ok
    (
        "G5",
        "bought milk 100 today",
        None,
        ["hp pump", "swiggy", "apollo"],
    ),  # groceries merchant
]
for tid, inp, expected, wrong in grocery_tests:
    merchant, _ = get_rag_chip(inp)
    got = str(merchant or "").lower()

    if expected is None:
        no_wrong = not any(w in got for w in wrong)
        check(
            tid,
            f"'{inp}' → groceries merchant (not food/transport)",
            no_wrong,
            f"got chip='{merchant}'",
        )
    else:
        correct = expected in got
        no_wrong = not any(w in got for w in wrong)
        check(
            tid,
            f"'{inp}' → chip contains '{expected}'",
            correct and no_wrong,
            f"got chip='{merchant}'",
        )

subsection("2C: Health — item-specific merchant suggestion")
health_tests = [
    ("H1", "bought tablets 200 today", "apollo", ["gold gym", "hp pump"]),
    ("H2", "paid gym membership 1200 today", "gold gym", ["apollo", "hp pump"]),
    ("H3", "bought medicine 300 today", "apollo", ["gold gym", "hp pump"]),
    (
        "H4",
        "bought supplements 400 today",
        None,
        ["hp pump", "swiggy", "dmart"],
    ),  # health merchant ok
]
for tid, inp, expected, wrong in health_tests:
    merchant, _ = get_rag_chip(inp)
    got = str(merchant or "").lower()

    if expected is None:
        no_wrong = not any(w in got for w in wrong)
        check(
            tid,
            f"'{inp}' → health merchant (not transport/food)",
            no_wrong,
            f"got chip='{merchant}'",
        )
    else:
        correct = expected in got
        no_wrong = not any(w in got for w in wrong)
        check(
            tid,
            f"'{inp}' → chip contains '{expected}'",
            correct and no_wrong,
            f"got chip='{merchant}'",
        )

subsection("2D: Transport — merchant suggestion")
transport_tests = [
    ("T1", "filled diesel 2000 today", "hp pump", []),
    ("T2", "filled petrol 1000 today", "hp pump", []),
    ("T3", "paid for cab 300 today", None, ["hp pump"]),  # NO chip — no cab in DB
]
for tid, inp, expected, wrong in transport_tests:
    merchant, _ = get_rag_chip(inp)
    got = str(merchant or "").lower()

    if expected is None:
        no_wrong = not any(w in got for w in wrong) if got else True
        check(
            tid,
            f"'{inp}' → no chip (no match in DB)",
            (not got) and no_wrong,
            f"got chip='{merchant}'",
        )
    else:
        correct = expected in got
        check(
            tid,
            f"'{inp}' → chip contains '{expected}'",
            correct,
            f"got chip='{merchant}'",
        )

subsection("2E: Cross-category isolation — no merchant leak")
isolation_tests = [
    ("X1", "bought medicine 300 today", ["hp pump", "dmart", "swiggy", "zomato"]),
    ("X2", "filled petrol 1500 today", ["apollo", "dmart", "swiggy", "gold gym"]),
    ("X3", "bought groceries 500 today", ["hp pump", "apollo", "swiggy", "gold gym"]),
    ("X4", "ordered food 200 today", ["hp pump", "apollo", "dmart", "gold gym"]),
]
for tid, inp, should_not_contain in isolation_tests:
    merchant, _ = get_rag_chip(inp)
    got = str(merchant or "").lower()
    no_leak = not any(w in got for w in should_not_contain)
    check(
        tid,
        f"'{inp}' → no cross-category merchant",
        no_leak,
        f"got chip='{merchant}' (should not contain {should_not_contain})",
    )

subsection("2F: Payment method suggestion")
payment_rag_tests = [
    ("PR1", "spent 300 on lunch at swiggy today", "UPI"),  # swiggy seeded with UPI
    (
        "PR2",
        "bought vegetables at bigbazaar 400 today",
        "Cash",
    ),  # bigbazaar seeded with cash
    ("PR3", "filled petrol at hp pump 1500 today", "Cash"),  # hp seeded with cash
    (
        "PR4",
        "paid gym fees at gold gym 1000 today",
        "Cash",
    ),  # gold gym seeded with cash
    (
        "PR5",
        "bought medicine at apollo pharmacy 300 today",
        "Card",
    ),  # apollo seeded with card
]
for tid, inp, expected_pm in payment_rag_tests:
    _, payment = get_rag_chip(inp)
    got = str(payment or "").lower()
    check(
        tid,
        f"'{inp}' → payment suggestion={expected_pm}",
        expected_pm.lower() in got,
        f"got payment='{payment}'",
    )

subsection("2G: Income RAG — payment method suggestion")
income_rag_tests = [
    ("IR1", "received salary 50000 today", "Bank Transfer"),
    ("IR2", "got salary 60000 today", "Bank Transfer"),
]
for tid, inp, expected_pm in income_rag_tests:
    d = debug(inp)
    assumptions = d.get("rag_assumptions") or {}
    payment = assumptions.get("payment_method")
    got = str(payment or "").lower()
    check(
        tid,
        f"'{inp}' → income payment suggestion={expected_pm}",
        expected_pm.lower() in got,
        f"got payment='{payment}'",
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


test_optional_chip("OC1", "spent 300 on lunch at swiggy today", "payment_method", "UPI")
test_optional_chip(
    "OC2", "bought vegetables at bigbazaar 400 today", "payment_method", "Cash"
)
test_optional_chip(
    "OC3", "filled petrol at hp pump 1500 today", "payment_method", "Cash"
)
test_optional_chip("OC4", "spent 300 on lunch today", "merchant", "swiggy")

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


test_chip_yes("CY1", "spent 300 on lunch at swiggy today", "payment_method", "UPI")
test_chip_yes(
    "CY2", "bought vegetables at bigbazaar 400 today", "payment_method", "Cash"
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


test_chip_no("CR1", "spent 300 on lunch at swiggy today", "payment_method")

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


test_skip("SK1", "spent 300 on lunch today", "merchant")
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
        ["hp pump", "swiggy", "zomato", "gold gym"],
    ),
    (
        "RG2",
        "bought tablets 250 today",
        ["hp pump", "swiggy", "zomato"],
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
    resp = chat("spent 320 on lunch today")
    session = resp.get("extracted", {})
    session["original_query"] = "spent 320 on lunch today"

    for _ in range(8):
        if resp.get("status") == "complete":
            break
        field = resp.get("followup_field")
        if field == "payment_method":
            session = resp.get("extracted", session)
            resp2 = chat("gpay", session_data=session, followup_field="payment_method")
            item = (resp2.get("extracted", {}).get("items") or [{}])[0]
            pm = str(item.get("payment_method") or "")
            check(tid, "Payment followup normalizes gpay → UPI", pm.lower() == "upi", f"got '{pm}'")
            return
        if not field:
            break
        session = resp.get("extracted", session)
        resp = chat("skip", session_data=session, followup_field=field)
        session = resp.get("extracted", session)

    check(tid, "Payment followup appeared for normalization test", False, "(payment followup never appeared)")


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

total = results["pass"] + results["fail"] + results["warn"]
section("FINAL REPORT")
print(
    f"""
  Total tests : {total}
  ✅ Pass     : {results['pass']}
  ❌ Fail     : {results['fail']}
  ⚠️  Warn     : {results['warn']}

  Score: {results['pass']}/{results['pass'] + results['fail']} 
         ({results['pass']/(results['pass']+results['fail'])*100:.1f}% pass rate, excluding warnings)
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
