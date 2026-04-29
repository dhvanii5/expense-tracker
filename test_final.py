"""
Pre-deployment stress test — 200+ cases.

Run:
    cd backend
    python test_deployment_ready.py

    # Run specific phase only:
    python test_deployment_ready.py --phase 3
"""

import requests, json, time, sys, argparse
from typing import Optional

BASE = "http://localhost:8000"
results = {"pass": 0, "fail": 0, "warn": 0}
all_failures = []
saved_ids = []

# ── Helpers ───────────────────────────────────────────────────────────────────


def debug(msg):
    r = requests.post(f"{BASE}/debug", json={"message": msg}, timeout=30)
    return r.json()


def chat(msg, session_data=None, followup_field=None):
    p = {"message": msg}
    if session_data:
        p["session_data"] = session_data
    if followup_field:
        p["followup_field"] = followup_field
    return requests.post(f"{BASE}/chat", json=p, timeout=30).json()


def save(entry):
    r = requests.post(f"{BASE}/save", json={"entry": entry}, timeout=10)
    tx_id = r.json().get("id", "")
    if tx_id:
        saved_ids.append(tx_id)
    return tx_id


def delete_tx(tx_id):
    requests.delete(f"{BASE}/transactions/{tx_id}", timeout=5)


def get_norm(msg):
    return debug(msg).get("normalized_output") or {}


def get_item(d):
    return (d.get("items") or [{}])[0]


def get_rag(msg):
    return debug(msg).get("rag_assumptions") or {}


def make_entry(
    intent,
    cat,
    item,
    merchant=None,
    payment=None,
    amount=500,
    dt="2026-04-01T00:00:00",
    source=None,
    payer=None,
):
    return {
        "intent": intent,
        "items": [
            {
                "amount": amount,
                "category": cat,
                "currency": "INR",
                "item": item,
                "merchant": merchant,
                "payment_method": payment,
                "datetime": dt,
                "bill_no": None,
                "remarks": None,
                "source": source,
                "payer": payer,
            }
        ],
    }


def section(t):
    print(f"\n{'='*65}\n  {t}\n{'='*65}")


def sub(t):
    print(f"\n  -- {t} --")


def check(tid, desc, passed, detail="", warn=False):
    if passed:
        results["pass"] += 1
        print(f"  OK  {tid}: {desc}")
    elif warn:
        results["warn"] += 1
        print(f"  ??  {tid}: {desc} {detail}")
    else:
        results["fail"] += 1
        all_failures.append(f"{tid}: {desc} {detail}")
        print(f"  XX  {tid}: {desc} {detail}")


def cleanup():
    if saved_ids:
        print(f"\n  Cleaning up {len(saved_ids)} entries...")
        for tx_id in list(saved_ids):
            delete_tx(tx_id)
        saved_ids.clear()


# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--phase", type=int, default=0)
args = parser.parse_args()
P = args.phase

# ── Health check ──────────────────────────────────────────────────────────────
print("\nChecking backend...")
try:
    r = requests.get(f"{BASE}/health", timeout=5)
    model = r.json().get("model", "?")
    print(f"  OK | Model: {model}")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)


# ==============================================================================
# PHASE 1 — EXTRACTION STRESS TEST (60 cases)
# ==============================================================================
if P in (0, 1):
    section("PHASE 1 — EXTRACTION STRESS TEST (60 cases)")

    sub("1A: Expense intent — 20 cases")
    for tid, inp in [
        ("E01", "spent 300 on lunch today"),
        ("E02", "paid 500 for dinner today"),
        ("E03", "bought medicine 300 today"),
        ("E04", "filled petrol 1500 today"),
        ("E05", "bought groceries 600 today"),
        ("E06", "Netflix subscription 649 today"),
        ("E07", "gym membership 1000 today"),
        ("E08", "Uber 200 today"),
        ("E09", "ordered food from swiggy 350 today"),
        ("E10", "bought shoes 2000 today"),
        ("E11", "electricity bill 1200 today"),
        ("E12", "mobile recharge 239 today"),
        ("E13", "coffee at CCD 180 today"),
        ("E14", "auto rickshaw 60 today"),
        ("E15", "doctor consultation 500 today"),
        ("E16", "Jio recharge 666 today"),
        ("E17", "bought vitamins 400 today"),
        ("E18", "Zomato 350 today"),
        ("E19", "paid building maintenance 1500 today"),
        ("E20", "booked hotel for 3500 today"),
    ]:
        e = get_norm(inp)
        check(tid, inp, e.get("intent") == "expense", f"-> got '{e.get('intent')}'")

    sub("1B: Income intent — 15 cases")
    for tid, inp in [
        ("I01", "got salary 45000 today"),
        ("I02", "received salary 50000 today"),
        ("I03", "salary credited 72000 today"),
        ("I04", "got bonus 10000 today"),
        ("I05", "received freelance payment 8000 today"),
        ("I06", "Rahul paid me back 2000 today"),
        ("I07", "Amazon refunded 499 today"),
        ("I08", "got cashback 150 today"),
        ("I09", "interest credited 500 today"),
        ("I10", "rent received 12000 today"),
        ("I11", "got 5000 from Priya today"),
        ("I12", "TCS credited 65000 salary today"),
        ("I13", "HCL paid me 48000 today"),
        ("I14", "received 9000 for freelance work today"),
        ("I15", "Ankit paid me 11000 for the project today"),
    ]:
        e = get_norm(inp)
        check(tid, inp, e.get("intent") == "income", f"-> got '{e.get('intent')}'")

    sub("1C: Category accuracy — 25 cases")
    for tid, inp, cat in [
        ("C01", "bought tablets 200 today", "Health"),
        ("C02", "bought medicine 300 today", "Health"),
        ("C03", "paid gym fees 1000 today", "Health"),
        ("C04", "bought supplements 400 today", "Health"),
        ("C05", "doctor visit 500 today", "Health"),
        ("C06", "lab test 800 today", "Health"),
        ("C07", "bought vitamins 300 today", "Health"),
        ("C08", "filled petrol 1500 today", "Transport"),
        ("C09", "filled diesel 2000 today", "Transport"),
        ("C10", "Uber cab 250 today", "Transport"),
        ("C11", "auto rickshaw 80 today", "Transport"),
        ("C12", "metro ticket 60 today", "Transport"),
        ("C13", "train ticket 450 today", "Transport"),
        ("C14", "bought vegetables 400 today", "Groceries"),
        ("C15", "bought rice 600 today", "Groceries"),
        ("C16", "bought groceries 500 today", "Groceries"),
        ("C17", "bought fruits 200 today", "Groceries"),
        ("C18", "Netflix subscription 649 today", "Entertainment"),
        ("C19", "bought subscription 799 today", "Entertainment"),
        ("C20", "Spotify 119 today", "Entertainment"),
        ("C21", "movie tickets at PVR 500 today", "Entertainment"),
        ("C22", "ordered lunch from swiggy 300 today", "Food"),
        ("C23", "coffee at starbucks 150 today", "Food"),
        ("C24", "electricity bill 1200 today", "Bills"),
        ("C25", "Jio recharge 239 today", "Bills"),
    ]:
        e = get_norm(inp)
        got = str(get_item(e).get("category") or "").strip()
        check(tid, f"{inp} -> {cat}", got.lower() == cat.lower(), f"got '{got}'")


# ==============================================================================
# PHASE 2 — PAYMENT METHOD DISCIPLINE (30 cases)
# ==============================================================================
if P in (0, 2):
    section("PHASE 2 — PAYMENT METHOD DISCIPLINE (30 cases)")

    sub("2A: Must be NULL — 15 cases")
    for tid, inp in [
        ("PM01", "bought medicine 300 today"),
        ("PM02", "filled petrol 1500 today"),
        ("PM03", "bought groceries 500 today"),
        ("PM04", "got salary 45000 today"),
        ("PM05", "Netflix subscription 649 today"),
        ("PM06", "gym membership 1000 today"),
        ("PM07", "doctor consultation 500 today"),
        ("PM08", "ordered food from swiggy 350 today"),
        ("PM09", "bought tablets 200 today"),
        ("PM10", "bought supplements 400 today"),
        ("PM11", "filled diesel 2000 today"),
        ("PM12", "bought vegetables 400 today"),
        ("PM13", "received salary 45000 today"),
        ("PM14", "got bonus 10000 today"),
        ("PM15", "electricity bill 1200 today"),
    ]:
        item = get_item(get_norm(inp))
        pm = item.get("payment_method")
        check(tid, f"'{inp}' -> payment=null", pm is None, f"got '{pm}'")

    sub("2B: Must be CORRECT — 15 cases")
    for tid, inp, expected in [
        ("PM16", "bought medicine 300 today via UPI", "UPI"),
        ("PM17", "filled petrol 1500 today cash", "Cash"),
        ("PM18", "Netflix 649 today card", "Card"),
        ("PM19", "received salary 45000 via bank transfer today", "Bank Transfer"),
        ("PM20", "paid 500 via gpay today", "UPI"),
        ("PM21", "bought groceries 600 today via phonepe", "UPI"),
        ("PM22", "spent 300 on food today via paytm", "UPI"),
        ("PM23", "bought shoes 2000 today credit card", "Card"),
        ("PM24", "paid electricity 1200 today via netbanking", "Bank Transfer"),
        ("PM25", "bought vitamins 400 today via UPI", "UPI"),
        ("PM26", "filled diesel 2000 cash today", "Cash"),
        ("PM27", "gym membership 1000 today by cash", "Cash"),
        ("PM28", "Zomato 350 today via UPI", "UPI"),
        ("PM29", "received 8000 freelance via bank transfer today", "Bank Transfer"),
        ("PM30", "bought tablet 200 today by card", "Card"),
    ]:
        got = str(get_item(get_norm(inp)).get("payment_method") or "").strip()
        check(
            tid,
            f"'{inp}' -> {expected}",
            got.lower() == expected.lower(),
            f"got '{got}'",
        )


# ==============================================================================
# PHASE 3 — RAG STRESS TEST (40 cases)
# ==============================================================================
if P in (0, 3):
    section("PHASE 3 — RAG STRESS TEST (40 cases)")
    print("  Seeding 15 transactions...")
    seeds = [
        make_entry("expense", "Food", "Lunch", "Swiggy", "UPI", 300),
        make_entry(
            "expense", "Food", "Dinner", "Zomato", "UPI", 500, "2026-04-02T00:00:00"
        ),
        make_entry(
            "expense", "Food", "Coffee", "Starbucks", "Card", 150, "2026-04-03T00:00:00"
        ),
        make_entry(
            "expense", "Food", "Pizza", "Dominos", "UPI", 650, "2026-04-04T00:00:00"
        ),
        make_entry("expense", "Groceries", "Vegetables", "BigBazaar", "Cash", 400),
        make_entry(
            "expense",
            "Groceries",
            "Groceries",
            "DMart",
            "UPI",
            600,
            "2026-04-02T00:00:00",
        ),
        make_entry(
            "expense",
            "Groceries",
            "Fruits",
            "Reliance Fresh",
            "Cash",
            200,
            "2026-04-03T00:00:00",
        ),
        make_entry("expense", "Health", "Medicines", "Apollo Pharmacy", "Card", 300),
        make_entry(
            "expense",
            "Health",
            "Gym fees",
            "Gold Gym",
            "Cash",
            1000,
            "2026-04-02T00:00:00",
        ),
        make_entry(
            "expense",
            "Health",
            "Lab test",
            "Lal Path Lab",
            "Cash",
            800,
            "2026-04-03T00:00:00",
        ),
        make_entry("expense", "Transport", "Petrol", "HP Pump", "Cash", 1500),
        make_entry(
            "expense",
            "Transport",
            "Cab ride",
            "Uber",
            "UPI",
            250,
            "2026-04-02T00:00:00",
        ),
        make_entry("expense", "Entertainment", "Subscription", "Netflix", "Card", 649),
        make_entry(
            "expense",
            "Entertainment",
            "Subscription",
            "Spotify",
            "UPI",
            119,
            "2026-04-02T00:00:00",
        ),
        make_entry(
            "income",
            "Salary",
            "Salary",
            None,
            "Bank Transfer",
            45000,
            source="Salary",
            payer="TCS",
        ),
    ]
    for s in seeds:
        save(s)
    time.sleep(0.5)

    sub("3A: Food — item-specific merchant (8 cases)")
    for tid, inp, must_have, must_not in [
        (
            "RF01",
            "spent 200 on lunch today",
            "swiggy",
            ["zomato", "starbucks", "dominos"],
        ),
        ("RF02", "spent 400 on dinner today", "zomato", ["swiggy", "starbucks"]),
        ("RF03", "spent 100 on coffee today", "starbucks", ["swiggy", "zomato"]),
        ("RF04", "ordered pizza 600 today", "dominos", ["swiggy", "starbucks"]),
        ("RF05", "ordered food 350 today", None, ["starbucks", "hp pump", "apollo"]),
        ("RF06", "bought breakfast today 200", None, ["hp pump", "apollo", "gold gym"]),
        ("RF07", "lunch 300 today", None, ["hp pump", "apollo", "dmart"]),
        (
            "RF08",
            "dinner at restaurant 500 today",
            None,
            ["hp pump", "apollo", "dmart"],
        ),
    ]:
        m = str(get_rag(inp).get("merchant") or "").lower()
        if must_have:
            check(
                tid,
                f"'{inp}' -> chip={must_have}",
                must_have in m and not any(w in m for w in must_not),
                f"got '{m}'",
            )
        else:
            check(
                tid,
                f"'{inp}' -> not {must_not}",
                (not m) or not any(w in m for w in must_not),
                f"got '{m}'",
            )

    sub("3B: Groceries — item-specific merchant (7 cases)")
    for tid, inp, must_have, must_not in [
        (
            "RG01",
            "spent 300 on vegetables today",
            "bigbazaar",
            ["dmart", "reliance", "apollo"],
        ),
        ("RG02", "bought rice 500 today", "dmart", ["bigbazaar", "reliance"]),
        ("RG03", "spent 150 on fruits today", "reliance", ["dmart", "bigbazaar"]),
        ("RG04", "bought groceries 800 today", None, ["hp pump", "swiggy", "apollo"]),
        ("RG05", "bought milk 100 today", None, ["hp pump", "swiggy", "apollo"]),
        ("RG06", "vegetable shopping 600 today", "bigbazaar", ["apollo", "gold gym"]),
        ("RG07", "weekly groceries 1200 today", None, ["hp pump", "apollo", "swiggy"]),
    ]:
        m = str(get_rag(inp).get("merchant") or "").lower()
        if must_have:
            check(
                tid,
                f"'{inp}' -> chip={must_have}",
                must_have in m and not any(w in m for w in must_not),
                f"got '{m}'",
            )
        else:
            check(
                tid,
                f"'{inp}' -> not {must_not}",
                (not m) or not any(w in m for w in must_not),
                f"got '{m}'",
            )

    sub("3C: Health — item-specific merchant (7 cases)")
    for tid, inp, must_have, must_not in [
        (
            "RH01",
            "bought tablets 200 today",
            "apollo",
            ["gold gym", "hp pump", "dmart"],
        ),
        ("RH02", "paid gym membership 1200 today", "gold gym", ["apollo", "hp pump"]),
        ("RH03", "bought medicine 300 today", "apollo", ["gold gym", "hp pump"]),
        ("RH04", "lab test 900 today", "lal path", ["gold gym", "hp pump"]),
        ("RH05", "bought supplements 400 today", None, ["hp pump", "swiggy", "dmart"]),
        (
            "RH06",
            "doctor consultation 500 today",
            None,
            ["hp pump", "swiggy", "zomato"],
        ),
        ("RH07", "bought vitamins 300 today", "apollo", ["gold gym", "hp pump"]),
    ]:
        m = str(get_rag(inp).get("merchant") or "").lower()
        if must_have:
            check(
                tid,
                f"'{inp}' -> chip={must_have}",
                must_have in m and not any(w in m for w in must_not),
                f"got '{m}'",
            )
        else:
            check(
                tid,
                f"'{inp}' -> not {must_not}",
                (not m) or not any(w in m for w in must_not),
                f"got '{m}'",
            )

    sub("3D: Transport (5 cases)")
    for tid, inp, must_have, must_not in [
        ("RT01", "filled diesel 2000 today", "hp pump", ["apollo", "dmart", "swiggy"]),
        ("RT02", "filled petrol 1000 today", "hp pump", ["apollo", "dmart"]),
        ("RT03", "cab ride 300 today", "uber", ["hp pump", "apollo"]),
        ("RT04", "paid for cab 300 today", None, ["apollo", "dmart", "swiggy"]),
        ("RT05", "flight ticket 4500 today", None, ["apollo", "dmart", "swiggy"]),
    ]:
        m = str(get_rag(inp).get("merchant") or "").lower()
        if must_have:
            check(
                tid,
                f"'{inp}' -> chip={must_have}",
                must_have in m and not any(w in m for w in must_not),
                f"got '{m}'",
            )
        else:
            check(
                tid,
                f"'{inp}' -> no wrong chip",
                (not m) or not any(w in m for w in must_not),
                f"got '{m}'",
            )

    sub("3E: Entertainment (4 cases)")
    for tid, inp, must_have, must_not in [
        ("RE01", "Netflix 649 today", "netflix", ["apollo", "hp pump", "dmart"]),
        ("RE02", "Spotify 119 today", "spotify", ["hp pump", "dmart"]),
        ("RE03", "bought subscription 799 today", None, ["hp pump", "apollo", "dmart"]),
        ("RE04", "OTT subscription 299 today", None, ["hp pump", "apollo", "dmart"]),
    ]:
        m = str(get_rag(inp).get("merchant") or "").lower()
        if must_have:
            check(
                tid,
                f"'{inp}' -> chip={must_have}",
                must_have in m and not any(w in m for w in must_not),
                f"got '{m}'",
            )
        else:
            check(
                tid,
                f"'{inp}' -> not {must_not}",
                (not m) or not any(w in m for w in must_not),
                f"got '{m}'",
            )

    sub("3F: Cross-category isolation — 9 critical tests")
    for tid, inp, must_not in [
        (
            "XI01",
            "bought medicine 300 today",
            ["hp pump", "uber", "dmart", "bigbazaar", "swiggy", "zomato", "netflix"],
        ),
        (
            "XI02",
            "filled petrol 1500 today",
            ["apollo", "gold gym", "dmart", "swiggy", "zomato", "netflix"],
        ),
        (
            "XI03",
            "bought groceries 500 today",
            ["hp pump", "uber", "apollo", "gold gym", "swiggy", "zomato", "netflix"],
        ),
        (
            "XI04",
            "ordered food 200 today",
            ["hp pump", "uber", "apollo", "gold gym", "dmart", "netflix"],
        ),
        (
            "XI05",
            "Netflix subscription 649 today",
            ["hp pump", "uber", "apollo", "gold gym", "dmart", "swiggy"],
        ),
        (
            "XI06",
            "gym fees 1000 today",
            ["hp pump", "uber", "dmart", "bigbazaar", "swiggy", "zomato"],
        ),
        (
            "XI07",
            "bought vegetables 400 today",
            ["hp pump", "uber", "apollo", "gold gym", "swiggy", "netflix"],
        ),
        (
            "XI08",
            "bought tablets 200 today",
            ["hp pump", "uber", "dmart", "bigbazaar", "swiggy", "netflix"],
        ),
        (
            "XI09",
            "coffee at starbucks 150 today",
            ["hp pump", "uber", "apollo", "gold gym", "dmart", "netflix"],
        ),
    ]:
        m = str(get_rag(inp).get("merchant") or "").lower()
        no_leak = (not m) or not any(w in m for w in must_not)
        check(tid, f"'{inp}' -> no cross-category leak", no_leak, f"got chip='{m}'")

    sub("3G: Income RAG — payment suggestion (2 cases)")
    for tid, inp, expected_pm in [
        ("RI01", "received salary 50000 today", "Bank Transfer"),
        ("RI02", "got salary 60000 today", "Bank Transfer"),
    ]:
        pm = str(get_rag(inp).get("payment_method") or "").lower()
        check(
            tid,
            f"'{inp}' -> income payment={expected_pm}",
            expected_pm.lower() in pm,
            f"got '{pm}'",
        )

    cleanup()


# ==============================================================================
# PHASE 4 — AMOUNT AND DATETIME ROBUSTNESS (20 cases)
# ==============================================================================
if P in (0, 4):
    section("PHASE 4 — AMOUNT AND DATETIME ROBUSTNESS (20 cases)")

    sub("4A: Amount formats (10 cases)")
    for tid, inp, expected in [
        ("AM01", "spent 1.5k on shopping today", 1500.0),
        ("AM02", "spent 1 lakh on car today", 100000.0),
        ("AM03", "paid Rs. 500 for food today", 500.0),
        ("AM04", "bought 300 medicine today", 300.0),
        ("AM05", "spent 1,500 on clothes today", 1500.0),
        ("AM06", "paid 2.5k for gym today", 2500.0),
        ("AM07", "bought groceries for 750 rupees today", 750.0),
        ("AM08", "spent 10 dollars today", 10.0),
        ("AM09", "paid 999 rs for subscription today", 999.0),
        ("AM10", "bought shoes 3000 today", 3000.0),
    ]:
        item = get_item(get_norm(inp))
        got = item.get("amount")
        check(tid, f"'{inp}' -> amount={expected}", got == expected, f"got {got}")

    sub("4B: Datetime null when not in input (10 cases)")
    for tid, inp, expect_dt in [
        ("DT01", "bought medicine 300 today", True),
        ("DT02", "bought medicine 300 yesterday", True),
        ("DT03", "bought medicine 300", False),
        ("DT04", "got salary 45000 today", True),
        ("DT05", "got salary 45000", False),
        ("DT06", "filled petrol 1500 yesterday", True),
        ("DT07", "filled petrol 1500", False),
        ("DT08", "bought groceries 500 today", True),
        ("DT09", "bought groceries 500", False),
        ("DT10", "Netflix subscription 649 today", True),
    ]:
        item = get_item(get_norm(inp))
        has_dt = item.get("datetime") is not None
        check(
            tid,
            f"'{inp}' -> datetime={'present' if expect_dt else 'null'}",
            has_dt == expect_dt,
            f"got datetime='{item.get('datetime')}'",
        )


# ==============================================================================
# PHASE 5 — REAL CONVERSATION FLOWS (25 cases)
# ==============================================================================
if P in (0, 5):
    section("PHASE 5 — REAL CONVERSATION FLOWS (25 cases)")

    sub("5A: Complete inputs — no mandatory followup (5 cases)")
    for tid, inp in [
        ("FC01", "spent 450 on dinner at zomato today via UPI"),
        ("FC02", "received salary 55000 today via bank transfer from TCS"),
        ("FC03", "bought medicine 300 at Apollo today via card"),
        ("FC04", "filled petrol at HP pump 1500 today via cash"),
        ("FC05", "Netflix subscription 649 today via card"),
    ]:
        resp = chat(inp)
        session = resp.get("extracted", {})
        session["original_query"] = inp
        for _ in range(6):
            if resp.get("status") == "complete":
                break
            if not resp.get("is_optional_followup"):
                break
            field = resp.get("followup_field")
            session = resp.get("extracted", session)
            resp = chat("skip", session_data=session, followup_field=field)
            session = resp.get("extracted", session)
        check(
            tid,
            f"'{inp}' -> no mandatory followup",
            resp.get("status") == "complete",
            f"got status='{resp.get('status')}' asking '{resp.get('followup_field')}'",
        )

    sub("5B: Incomplete inputs ask correct mandatory field (5 cases)")
    for tid, inp, expected_field in [
        ("FM01", "bought medicine at apollo", "amount"),
        ("FM02", "bought medicine 300", "datetime"),
        ("FM03", "spent on groceries today", "amount"),
        ("FM04", "got salary today", "amount"),
        ("FM05", "received freelance payment", "amount"),
    ]:
        resp = chat(inp)
        field = resp.get("followup_field")
        is_opt = resp.get("is_optional_followup", True)
        check(
            tid,
            f"'{inp}' -> asks '{expected_field}'",
            field == expected_field and not is_opt,
            f"got field='{field}', is_optional={is_opt}",
        )

    sub("5C: Amount followup parses correctly (5 cases)")

    def test_amount_fu(tid, initial, answer, expected):
        resp = chat(initial)
        session = resp.get("extracted", {})
        session["original_query"] = initial
        for _ in range(3):
            if resp.get("followup_field") == "amount":
                session = resp.get("extracted", session)
                resp = chat(answer, session_data=session, followup_field="amount")
                got = get_item(resp.get("extracted", {})).get("amount")
                check(
                    tid,
                    f"Amount fu '{answer}' -> {expected}",
                    got == expected,
                    f"got {got}",
                )
                return
            f = resp.get("followup_field")
            session = resp.get("extracted", session)
            resp = chat("skip", session_data=session, followup_field=f)
            session = resp.get("extracted", session)
        check(tid, f"Amount fu test", False, "(no amount followup)")

    test_amount_fu("FA01", "bought medicine today", "300", 300.0)
    test_amount_fu("FA02", "bought medicine today", "rs 500", 500.0)
    test_amount_fu("FA03", "got salary today", "45000", 45000.0)
    test_amount_fu("FA04", "bought medicine today", "1.5k", 1500.0)
    test_amount_fu("FA05", "got salary today", "72000", 72000.0)

    sub("5D: Datetime followup parses correctly (5 cases)")

    def test_dt_fu(tid, initial, answer):
        resp = chat(initial)
        session = resp.get("extracted", {})
        session["original_query"] = initial
        for _ in range(3):
            if resp.get("followup_field") == "datetime":
                session = resp.get("extracted", session)
                resp = chat(answer, session_data=session, followup_field="datetime")
                got = get_item(resp.get("extracted", {})).get("datetime")
                check(
                    tid, f"DT fu '{answer}' -> not null", got is not None, f"got null"
                )
                return
            f = resp.get("followup_field")
            session = resp.get("extracted", session)
            resp = chat("skip", session_data=session, followup_field=f)
            session = resp.get("extracted", session)
        check(tid, f"DT fu test", False, "(no datetime followup)")

    test_dt_fu("FD01", "bought medicine 300", "today")
    test_dt_fu("FD02", "bought medicine 300", "yesterday")
    test_dt_fu("FD03", "bought medicine 300", "15 april")
    test_dt_fu("FD04", "got salary 45000", "today")
    test_dt_fu("FD05", "got salary 45000", "1st april")

    sub("5E: Payment followup normalizes correctly (5 cases)")

    def test_pm_fu(tid, initial, answer, expected):
        resp = chat(initial)
        session = resp.get("extracted", {})
        session["original_query"] = initial
        for _ in range(6):
            if resp.get("followup_field") == "payment_method":
                session = resp.get("extracted", session)
                resp = chat(
                    answer, session_data=session, followup_field="payment_method"
                )
                got = str(
                    get_item(resp.get("extracted", {})).get("payment_method") or ""
                ).lower()
                check(
                    tid,
                    f"PM fu '{answer}' -> {expected}",
                    expected.lower() in got,
                    f"got '{got}'",
                )
                return
            f = resp.get("followup_field")
            session = resp.get("extracted", session)
            resp = chat("skip", session_data=session, followup_field=f)
            session = resp.get("extracted", session)
        check(tid, f"PM fu test", False, "(no payment followup)")

    test_pm_fu("FP01", "bought medicine 300 today", "gpay", "UPI")
    test_pm_fu("FP02", "bought medicine 300 today", "cash", "Cash")
    test_pm_fu("FP03", "bought medicine 300 today", "card", "Card")
    test_pm_fu("FP04", "bought medicine 300 today", "upi", "UPI")
    test_pm_fu("FP05", "bought medicine 300 today", "phonepe", "UPI")


# ==============================================================================
# PHASE 6 — BOUNDARY CASES (20 cases)
# ==============================================================================
if P in (0, 6):
    section("PHASE 6 — BOUNDARY CASES (20 cases)")

    sub("6A: Unsupported inputs (8 cases)")
    for tid, inp in [
        ("UN01", "what is 2+2"),
        ("UN02", "hello"),
        ("UN03", "help me plan my budget"),
        ("UN04", "who is elon musk"),
        ("UN05", "remind me to buy milk"),
        ("UN06", "what is inflation"),
        ("UN07", "how is the weather today"),
        ("UN08", "tell me a joke"),
    ]:
        intent = get_norm(inp).get("intent")
        check(
            tid,
            f"Unsupported: '{inp}' -> not expense/income",
            intent not in {"expense", "income"},
            f"got '{intent}'",
            warn=True,
        )

    sub("6B: Amount NOT hallucinated (7 cases)")
    for tid, inp in [
        ("NA01", "bought medicine today"),
        ("NA02", "got salary today"),
        ("NA03", "filled petrol today"),
        ("NA04", "bought groceries today"),
        ("NA05", "received freelance payment today"),
        ("NA06", "Netflix subscription today"),
        ("NA07", "paid gym today"),
    ]:
        amt = get_item(get_norm(inp)).get("amount")
        check(tid, f"'{inp}' -> amount=null", amt is None, f"got {amt}")

    sub("6C: Typos handled correctly (5 cases)")
    for tid, inp, expected_intent in [
        ("TY01", "recieved salary 45000 today", "income"),
        ("TY02", "payed 300 at zomato today", "expense"),
        ("TY03", "bout shoes 2000 today", "expense"),
        ("TY04", "spent 300 on grocieries today", "expense"),
        ("TY05", "fillled petrol 1500 today", "expense"),
    ]:
        got = get_norm(inp).get("intent")
        check(
            tid,
            f"Typo: '{inp}' -> {expected_intent}",
            got == expected_intent,
            f"got '{got}'",
            warn=True,
        )


# ==============================================================================
# FINAL REPORT
# ==============================================================================
cleanup()
total = results["pass"] + results["fail"] + results["warn"]
denom = results["pass"] + results["fail"]
rate = results["pass"] / denom * 100 if denom > 0 else 0

section("FINAL REPORT")
print(
    f"""
  Total : {total}  |  Pass: {results['pass']}  |  Fail: {results['fail']}  |  Warn: {results['warn']}
  Score : {results['pass']}/{denom} = {rate:.1f}%
"""
)

if results["fail"]:
    print("  ── Failures ──")
    for f in all_failures:
        print(f"    XX {f}")

print()
if rate >= 95:
    print("  DEPLOYMENT READY - excellent accuracy")
elif rate >= 88:
    print("  MOSTLY READY - fix failures before deploying")
else:
    print("  NOT READY - too many failures")

print(
    """
  Phase guide:
  Phase 1 -> model extraction    (fix fine-tuning)
  Phase 2 -> payment discipline  (fix normalize_model_output)
  Phase 3 -> RAG suggestions     (fix embed text / thresholds)
  Phase 4 -> amount/datetime     (fix coerce_amount / coerce_datetime)
  Phase 5 -> chat flow           (fix /chat route)
  Phase 6 -> edge cases          (fix is_query_like_message)
"""
)
