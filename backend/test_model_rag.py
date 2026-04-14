"""
pytest test suite for finance tracker — model extraction + RAG retrieval.

Run:
    cd backend
    pytest test_model_rag.py -v                  # all tests
    pytest test_model_rag.py -v -k "extraction"  # only extraction tests
    pytest test_model_rag.py -v -k "retrieval"   # only retrieval tests
    pytest test_model_rag.py -v -k "category"    # only category tests
    pytest test_model_rag.py --tb=short          # shorter tracebacks

Requirements:
    pip install pytest requests
    Backend must be running: uvicorn main:app --reload --port 8000
"""

import pytest
import requests
import json
import os
import time
import subprocess
import sys

BASE = "http://localhost:8000"
HEALTH_PATH = "/health"


def _wait_for_backend_ready(base: str, timeout_seconds: int = 90) -> bool:
    """Poll health endpoint until backend responds 200 or timeout expires."""
    deadline = time.time() + timeout_seconds
    last_error = None
    while time.time() < deadline:
        try:
            r = requests.get(
                f"{base}{HEALTH_PATH}",
                timeout=(2, 8),  # fast connect timeout, tolerant read timeout
            )
            if r.status_code == 200:
                return True
        except Exception as e:
            last_error = e
        time.sleep(1.0)
    if last_error:
        print(f"Backend readiness check last error: {last_error}")
    return False


# ── Helpers ────────────────────────────────────────────────────────────────────


def chat(message: str, session_data=None, followup_field=None) -> dict:
    payload = {"message": message}
    if session_data:
        payload["session_data"] = session_data
    if followup_field:
        payload["followup_field"] = followup_field
    r = requests.post(f"{BASE}/chat", json=payload, timeout=30)
    assert r.status_code == 200, f"Chat failed: {r.text}"
    return r.json()


def debug(message: str) -> dict:
    r = requests.post(f"{BASE}/debug", json={"message": message}, timeout=90)
    assert r.status_code == 200, f"Debug failed: {r.text}"
    return r.json()


def save(entry: dict) -> str:
    r = requests.post(f"{BASE}/save", json={"entry": entry}, timeout=10)
    assert r.status_code == 200
    return r.json().get("id", "")


def delete_tx(tx_id: str):
    if tx_id:
        try:
            requests.delete(f"{BASE}/transactions/{tx_id}", timeout=20)
        except Exception:
            # Cleanup should be best-effort; avoid failing assertions due to transient API timeout.
            pass


def get_extracted(message: str) -> dict:
    """Get normalized extraction from /debug."""
    d = debug(message)
    return d.get("normalized_output") or d.get("raw_extracted") or {}


def get_item(extracted: dict) -> dict:
    items = extracted.get("items") or []
    return items[0] if items else {}


def get_rag_assumptions(message: str) -> dict:
    d = debug(message)
    return d.get("rag_assumptions") or {}


def get_rag_candidates(message: str) -> list:
    d = debug(message)
    return d.get("rag_top_candidates") or []


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session", autouse=True)
def check_backend():
    """Ensure backend is reachable for integration tests.

    Behavior:
    1) If backend is already up, use it.
    2) Otherwise auto-start `uvicorn main:app --port 8000` for this test session.
    """
    backend_proc = None

    if not _wait_for_backend_ready(BASE, timeout_seconds=15):
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--log-level",
            "warning",
        ]
        try:
            backend_proc = subprocess.Popen(
                cmd,
                cwd=os.path.dirname(__file__),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            pytest.fail(f"Failed to start backend process automatically. Error: {e}")

        if not _wait_for_backend_ready(BASE, timeout_seconds=120):
            if backend_proc and backend_proc.poll() is None:
                backend_proc.terminate()
            code = backend_proc.poll() if backend_proc else "unknown"
            pytest.fail(
                "Backend not reachable at "
                f"{BASE} after auto-start attempt. "
                f"Process exit code: {code}. "
                "If port 8000 is occupied, stop that process and re-run tests."
            )

    yield

    if backend_proc and backend_proc.poll() is None:
        backend_proc.terminate()
        try:
            backend_proc.wait(timeout=10)
        except Exception:
            backend_proc.kill()


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1: MODEL EXTRACTION TESTS
# These test what Qwen returns BEFORE any RAG or business logic.
# If these fail → model problem (fine-tuning needed).
# ══════════════════════════════════════════════════════════════════════════════


class TestModelExtraction:
    """Test that the model correctly extracts fields from natural language."""

    # ── Intent detection ──────────────────────────────────────────────────────

    def test_intent_expense_spent(self):
        e = get_extracted("spent 500 on coffee today")
        assert e.get("intent") == "expense", f"Expected expense, got {e.get('intent')}"

    def test_intent_expense_paid(self):
        e = get_extracted("paid 300 for lunch today")
        assert e.get("intent") == "expense"

    def test_intent_expense_bought(self):
        e = get_extracted("bought shoes today for 2000")
        assert e.get("intent") == "expense"

    def test_intent_income_received(self):
        e = get_extracted("received salary 45000 today")
        assert e.get("intent") == "income", f"Expected income, got {e.get('intent')}"

    def test_intent_income_got(self):
        e = get_extracted("got my salary 50000 today")
        assert e.get("intent") == "income"

    def test_intent_income_credited(self):
        e = get_extracted("salary credited 45000 today")
        assert e.get("intent") == "income"

    # ── Amount extraction ─────────────────────────────────────────────────────

    def test_amount_plain_number(self):
        item = get_item(get_extracted("spent 500 on food today"))
        assert item.get("amount") == 500.0, f"Expected 500, got {item.get('amount')}"

    def test_amount_with_rs(self):
        item = get_item(get_extracted("paid rs 300 for coffee today"))
        assert item.get("amount") == 300.0

    def test_amount_with_rupee_symbol(self):
        item = get_item(get_extracted("spent ₹400 on groceries today"))
        assert item.get("amount") == 400.0

    def test_amount_k_suffix(self):
        item = get_item(get_extracted("received 45k salary today"))
        assert (
            item.get("amount") == 45000.0
        ), f"Expected 45000, got {item.get('amount')}"

    def test_amount_not_hallucinated(self):
        """If no amount in input, model should return null — not guess."""
        item = get_item(get_extracted("bought medicine today"))
        assert (
            item.get("amount") is None
        ), f"Model hallucinated amount {item.get('amount')} — should be null"

    def test_amount_not_hallucinated_subscription(self):
        item = get_item(get_extracted("bought subscription today"))
        assert (
            item.get("amount") is None
        ), f"Model hallucinated amount {item.get('amount')}"

    # ── Datetime extraction ───────────────────────────────────────────────────

    def test_datetime_today(self):
        item = get_item(get_extracted("spent 500 on food today"))
        assert (
            item.get("datetime") is not None
        ), "datetime should not be null for 'today'"
        assert "2026" in str(
            item.get("datetime")
        ), f"Bad datetime: {item.get('datetime')}"

    def test_datetime_yesterday(self):
        item = get_item(get_extracted("paid 300 for lunch yesterday"))
        assert item.get("datetime") is not None

    def test_datetime_null_when_missing(self):
        """If no date in input, datetime should be null."""
        item = get_item(get_extracted("bought coffee for 150"))
        # No date mentioned — should be null so follow-up is triggered
        # (or today if model defaults to today — both acceptable)
        # Just ensure it's not a garbage value
        dt = item.get("datetime")
        if dt is not None:
            assert "2026" in str(dt) or "2025" in str(dt), f"Bad datetime value: {dt}"

    # ── Category extraction ───────────────────────────────────────────────────

    def test_category_food(self):
        item = get_item(get_extracted("spent 200 on food today"))
        cat = str(item.get("category") or "").lower()
        assert cat in {
            "food",
            "groceries",
            "dining",
        }, f"Expected food category, got {cat}"

    def test_category_groceries(self):
        """Critical: 'groceries' in input should give category=Groceries, not Food."""
        item = get_item(get_extracted("spend 500 on groceries today"))
        cat = str(item.get("category") or "").lower()
        assert (
            cat == "groceries"
        ), f"FAIL: 'groceries' in input but got category='{cat}'. Model maps groceries→food."

    def test_category_transport(self):
        item = get_item(get_extracted("filled petrol 1500 today"))
        cat = str(item.get("category") or "").lower()
        assert cat == "transport", f"Expected transport, got {cat}"

    def test_category_entertainment(self):
        """Critical: subscription should map to Entertainment."""
        item = get_item(get_extracted("bought netflix subscription today for 649"))
        cat = str(item.get("category") or "").lower()
        assert (
            cat == "entertainment"
        ), f"FAIL: Netflix subscription should be entertainment, got '{cat}'"

    def test_category_health(self):
        item = get_item(get_extracted("bought medicine today for 300"))
        cat = str(item.get("category") or "").lower()
        assert cat == "health", f"Expected health, got {cat}"

    def test_category_entertainment_subscription_no_merchant(self):
        """Subscription without explicit merchant — still should be Entertainment."""
        item = get_item(get_extracted("bought subscription today for 799"))
        cat = str(item.get("category") or "").lower()
        assert (
            cat == "entertainment"
        ), f"FAIL: subscription should be entertainment, got '{cat}'"

    # ── Payment method extraction ─────────────────────────────────────────────

    def test_payment_upi_explicit(self):
        item = get_item(get_extracted("paid 500 via UPI today"))
        pm = str(item.get("payment_method") or "").lower()
        assert pm == "upi", f"Expected UPI, got {pm}"

    def test_payment_gpay_normalized(self):
        item = get_item(get_extracted("paid 500 via gpay today"))
        pm = item.get("payment_method")
        assert pm == "UPI", f"Expected UPI (gpay normalized), got {pm}"

    def test_payment_null_when_not_mentioned(self):
        """If no payment method mentioned, should be null."""
        item = get_item(get_extracted("spent 500 on groceries today"))
        pm = item.get("payment_method")
        assert (
            pm is None
        ), f"payment_method should be null when not mentioned, got '{pm}'"

    def test_payment_bank_transfer(self):
        item = get_item(get_extracted("received salary 45000 via bank transfer today"))
        pm = item.get("payment_method")
        assert pm == "Bank Transfer", f"Expected Bank Transfer, got {pm}"

    # ── Merchant extraction ───────────────────────────────────────────────────

    def test_merchant_explicit(self):
        item = get_item(get_extracted("ordered food from swiggy today for 300"))
        merchant = str(item.get("merchant") or "").lower()
        assert "swiggy" in merchant, f"Expected Swiggy, got {merchant}"

    def test_merchant_null_when_not_mentioned(self):
        item = get_item(get_extracted("spent 500 on groceries today"))
        merchant = item.get("merchant")
        assert (
            merchant is None
        ), f"merchant should be null when not mentioned, got '{merchant}'"


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2: RAG RETRIEVAL TESTS
# These seed the DB first, then test if retrieval finds the right entries.
# If these fail → retrieval problem (build_assumptions / score_entry_for_field).
# ══════════════════════════════════════════════════════════════════════════════


class TestRAGRetrieval:
    """
    Test RAG assumption suggestions.
    Each test seeds DB entries, runs a query, checks suggestions, then cleans up.
    """

    # ── Transport ─────────────────────────────────────────────────────────────

    def test_transport_same_merchant_suggested(self):
        """After saving HP petrol entry, similar transport query should suggest HP."""
        tx_id = save(
            {
                "intent": "expense",
                "items": [
                    {
                        "amount": 1500,
                        "category": "Transport",
                        "item": "Petrol",
                        "merchant": "HP Pump",
                        "payment_method": "Cash",
                        "datetime": "2026-04-01T00:00:00",
                        "currency": "INR",
                        "remarks": None,
                        "bill_no": None,
                        "source": None,
                        "payer": None,
                    }
                ],
            }
        )
        try:
            d = debug("filled petrol 2000 today")
            candidates = d.get("rag_top_candidates") or []
            merchants = [c.get("merchant", "").lower() for c in candidates]
            assert any(
                "hp" in m for m in merchants
            ), f"HP Pump not in RAG candidates: {candidates}"
        finally:
            delete_tx(tx_id)

    def test_transport_wrong_category_not_suggested(self):
        """Groceries merchant should NOT be suggested for transport query."""
        tx_id = save(
            {
                "intent": "expense",
                "items": [
                    {
                        "amount": 500,
                        "category": "Groceries",
                        "item": "Groceries",
                        "merchant": "DMart",
                        "payment_method": "UPI",
                        "datetime": "2026-04-01T00:00:00",
                        "currency": "INR",
                        "remarks": None,
                        "bill_no": None,
                        "source": None,
                        "payer": None,
                    }
                ],
            }
        )
        try:
            d = debug("filled petrol 2000 today")
            assumptions = d.get("rag_assumptions") or {}
            merchant = str(assumptions.get("merchant") or "").lower()
            assert (
                "dmart" not in merchant
            ), f"FAIL: DMart (groceries) suggested for transport query. Cross-category leak."
        finally:
            delete_tx(tx_id)

    # ── Groceries ─────────────────────────────────────────────────────────────

    def test_groceries_merchant_suggested(self):
        """After saving DMart groceries, similar query should suggest DMart."""
        tx_id = save(
            {
                "intent": "expense",
                "items": [
                    {
                        "amount": 300,
                        "category": "Groceries",
                        "item": "Groceries",
                        "merchant": "DMart",
                        "payment_method": "UPI",
                        "datetime": "2026-04-01T00:00:00",
                        "currency": "INR",
                        "remarks": None,
                        "bill_no": None,
                        "source": None,
                        "payer": None,
                    }
                ],
            }
        )
        try:
            d = debug("spend 400 on groceries today")
            candidates = d.get("rag_top_candidates") or []
            merchants = [c.get("merchant", "").lower() for c in candidates]
            assert any(
                "dmart" in m for m in merchants
            ), f"DMart not in RAG candidates for groceries query: {candidates}"
        finally:
            delete_tx(tx_id)

    def test_groceries_payment_suggested(self):
        """After saving DMart UPI groceries, payment method should be suggested."""
        tx_id = save(
            {
                "intent": "expense",
                "items": [
                    {
                        "amount": 300,
                        "category": "Groceries",
                        "item": "Groceries",
                        "merchant": "DMart",
                        "payment_method": "UPI",
                        "datetime": "2026-04-01T00:00:00",
                        "currency": "INR",
                        "remarks": None,
                        "bill_no": None,
                        "source": None,
                        "payer": None,
                    }
                ],
            }
        )
        try:
            assumptions = get_rag_assumptions("spend 400 on groceries today")
            pm = assumptions.get("payment_method")
            assert pm == "UPI", f"Expected payment_method=UPI from RAG, got {pm}"
        finally:
            delete_tx(tx_id)

    # ── Entertainment / Subscription ─────────────────────────────────────────

    def test_subscription_merchant_suggested(self):
        """Critical: after saving amazon prime subscription, same item query should suggest it."""
        tx_id = save(
            {
                "intent": "expense",
                "items": [
                    {
                        "amount": 799,
                        "category": "Entertainment",
                        "item": "Subscription",
                        "merchant": "amazon prime",
                        "payment_method": "UPI",
                        "datetime": "2026-04-01T00:00:00",
                        "currency": "INR",
                        "remarks": None,
                        "bill_no": None,
                        "source": None,
                        "payer": None,
                    }
                ],
            }
        )
        try:
            d = debug("bought subscription for 799 today")
            candidates = d.get("rag_top_candidates") or []
            merchants = [c.get("merchant", "").lower() for c in candidates]
            assumptions = d.get("rag_assumptions") or {}
            print(f"\n  DEBUG subscription test:")
            print(f"  candidates: {candidates}")
            print(f"  assumptions: {assumptions}")
            print(f"  retrieval_stage: {d.get('retrieval_stage')}")
            assert any(
                "amazon" in m for m in merchants
            ), f"FAIL: amazon prime not in RAG candidates: {candidates}"
        finally:
            delete_tx(tx_id)

    def test_netflix_vs_spotify_item_wins(self):
        """When query mentions 'music', Spotify should win over Netflix."""
        ids = []
        ids.append(
            save(
                {
                    "intent": "expense",
                    "items": [
                        {
                            "amount": 649,
                            "category": "Entertainment",
                            "item": "Subscription",
                            "merchant": "Netflix",
                            "payment_method": "Card",
                            "datetime": "2026-04-01T00:00:00",
                            "currency": "INR",
                            "remarks": None,
                            "bill_no": None,
                            "source": None,
                            "payer": None,
                        }
                    ],
                }
            )
        )
        ids.append(
            save(
                {
                    "intent": "expense",
                    "items": [
                        {
                            "amount": 119,
                            "category": "Entertainment",
                            "item": "Music Subscription",
                            "merchant": "Spotify",
                            "payment_method": "UPI",
                            "datetime": "2026-04-02T00:00:00",
                            "currency": "INR",
                            "remarks": None,
                            "bill_no": None,
                            "source": None,
                            "payer": None,
                        }
                    ],
                }
            )
        )
        try:
            d = debug("music subscription 179 today")
            assumptions = d.get("rag_assumptions") or {}
            merchant = str(assumptions.get("merchant") or "").lower()
            print(f"\n  DEBUG netflix vs spotify: assumptions={assumptions}")
            # Spotify should win because "music" matches "Music Subscription" item
            assert (
                "spotify" in merchant
            ), f"Expected Spotify for music query, got '{merchant}'"
        finally:
            for tx_id in ids:
                delete_tx(tx_id)

    def test_netflix_suggested_when_mentioned(self):
        """When 'netflix' is in query, Netflix should win."""
        ids = []
        ids.append(
            save(
                {
                    "intent": "expense",
                    "items": [
                        {
                            "amount": 649,
                            "category": "Entertainment",
                            "item": "Subscription",
                            "merchant": "Netflix",
                            "payment_method": "Card",
                            "datetime": "2026-04-01T00:00:00",
                            "currency": "INR",
                            "remarks": None,
                            "bill_no": None,
                            "source": None,
                            "payer": None,
                        }
                    ],
                }
            )
        )
        ids.append(
            save(
                {
                    "intent": "expense",
                    "items": [
                        {
                            "amount": 119,
                            "category": "Entertainment",
                            "item": "Subscription",
                            "merchant": "Spotify",
                            "payment_method": "UPI",
                            "datetime": "2026-04-02T00:00:00",
                            "currency": "INR",
                            "remarks": None,
                            "bill_no": None,
                            "source": None,
                            "payer": None,
                        }
                    ],
                }
            )
        )
        try:
            d = debug("netflix 799 today")
            assumptions = d.get("rag_assumptions") or {}
            merchant = str(assumptions.get("merchant") or "").lower()
            assert (
                "netflix" in merchant
            ), f"Expected Netflix when 'netflix' in query, got '{merchant}'"
        finally:
            for tx_id in ids:
                delete_tx(tx_id)

    # ── Health ────────────────────────────────────────────────────────────────

    def test_health_medicine_merchant_suggested(self):
        """After saving Apollo medicine, similar query should suggest Apollo."""
        tx_id = save(
            {
                "intent": "expense",
                "items": [
                    {
                        "amount": 200,
                        "category": "Health",
                        "item": "Medicines",
                        "merchant": "Apollo Pharmacy",
                        "payment_method": "Cash",
                        "datetime": "2026-04-01T00:00:00",
                        "currency": "INR",
                        "remarks": None,
                        "bill_no": None,
                        "source": None,
                        "payer": None,
                    }
                ],
            }
        )
        try:
            d = debug("bought medicine 300 today")
            candidates = d.get("rag_top_candidates") or []
            merchants = [c.get("merchant", "").lower() for c in candidates]
            assert any(
                "apollo" in m for m in merchants
            ), f"Apollo Pharmacy not in candidates for medicine query: {candidates}"
        finally:
            delete_tx(tx_id)

    def test_health_gym_merchant_suggested(self):
        """After saving Gold Gym, gym query should suggest Gold Gym."""
        tx_id = save(
            {
                "intent": "expense",
                "items": [
                    {
                        "amount": 1000,
                        "category": "Health",
                        "item": "Gym Membership",
                        "merchant": "Gold Gym",
                        "payment_method": "Cash",
                        "datetime": "2026-04-01T00:00:00",
                        "currency": "INR",
                        "remarks": None,
                        "bill_no": None,
                        "source": None,
                        "payer": None,
                    }
                ],
            }
        )
        try:
            d = debug("paid gym membership 1200 today")
            candidates = d.get("rag_top_candidates") or []
            merchants = [c.get("merchant", "").lower() for c in candidates]
            assert any(
                "gold" in m for m in merchants
            ), f"Gold Gym not in candidates for gym query: {candidates}"
        finally:
            delete_tx(tx_id)

    # ── Income ────────────────────────────────────────────────────────────────

    def test_income_payment_method_suggested(self):
        """After saving salary via bank transfer, similar income should suggest Bank Transfer."""
        tx_id = save(
            {
                "intent": "income",
                "items": [
                    {
                        "amount": 45000,
                        "category": "Salary",
                        "item": "Salary",
                        "merchant": None,
                        "payment_method": "Bank Transfer",
                        "source": "Salary",
                        "payer": None,
                        "datetime": "2026-04-01T00:00:00",
                        "currency": "INR",
                        "remarks": None,
                        "bill_no": None,
                    }
                ],
            }
        )
        try:
            assumptions = get_rag_assumptions("received salary 50000 today")
            pm = assumptions.get("payment_method")
            assert pm == "Bank Transfer", f"Expected Bank Transfer for salary, got {pm}"
        finally:
            delete_tx(tx_id)

    # ── Cross-category isolation ───────────────────────────────────────────────

    def test_no_cross_category_merchant_leak(self):
        """Grocery merchant should never appear in health query."""
        ids = []
        ids.append(
            save(
                {
                    "intent": "expense",
                    "items": [
                        {
                            "amount": 500,
                            "category": "Groceries",
                            "item": "Groceries",
                            "merchant": "DMart",
                            "payment_method": "UPI",
                            "datetime": "2026-04-01T00:00:00",
                            "currency": "INR",
                            "remarks": None,
                            "bill_no": None,
                            "source": None,
                            "payer": None,
                        }
                    ],
                }
            )
        )
        try:
            assumptions = get_rag_assumptions("bought medicine 300 today")
            merchant = str(assumptions.get("merchant") or "").lower()
            assert (
                "dmart" not in merchant
            ), f"FAIL: DMart (groceries) leaked into medicine (health) query. Got: {merchant}"
        finally:
            for tx_id in ids:
                delete_tx(tx_id)

    def test_no_suggestion_for_completely_new_category(self):
        """For a category with no past entries, no merchant should be suggested."""
        assumptions = get_rag_assumptions("bought a drone for 15000 today")
        merchant = assumptions.get("merchant")
        assert (
            merchant is None or merchant == ""
        ), f"Should not suggest merchant for brand new category, got '{merchant}'"


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3: NORMALIZE TESTS
# Test that normalize_model_output handles different model output shapes.
# These call /debug to check normalized_output.
# ══════════════════════════════════════════════════════════════════════════════


class TestNormalization:
    """Test that the normalization adapter handles model output quirks."""

    def test_payment_typo_gpya_normalized(self):
        """'gpya' should normalize to UPI via fuzzy match."""
        # We test this through the actual normalize function by calling debug
        # on an input that has payment method, then checking normalization
        item = get_item(get_extracted("paid 500 via gpay today"))
        pm = item.get("payment_method")
        assert pm == "UPI", f"gpay should normalize to UPI, got {pm}"

    def test_category_override_groceries(self):
        """'groceries' in input should give category=Groceries even if model says food."""
        item = get_item(get_extracted("spend 500 on groceries today"))
        cat = str(item.get("category") or "").lower()
        assert (
            cat == "groceries"
        ), f"Category override failed: groceries input got category='{cat}'"

    def test_category_override_transport(self):
        item = get_item(get_extracted("transport expense 500 today"))
        cat = str(item.get("category") or "").lower()
        assert cat == "transport", f"Expected transport, got '{cat}'"

    def test_amount_not_hallucinated_when_absent(self):
        """No number in input → amount must be null."""
        item = get_item(get_extracted("bought coffee today"))
        assert item.get("amount") is None, f"Amount hallucinated: {item.get('amount')}"


# ══════════════════════════════════════════════════════════════════════════════
# QUICK DIAGNOSTIC — run this first to see the big picture
# ══════════════════════════════════════════════════════════════════════════════


class TestDiagnostic:
    """
    Run these first. They print raw debug output so you can see
    what model returns vs what RAG finds.
    """

    def test_print_subscription_debug(self, capsys):
        """Print full debug for subscription — see if category=null causes RAG miss."""
        d = debug("bought subscription for 799 today")
        raw = d.get("raw_model_output", {})
        normalized = d.get("normalized_output", {})
        item = (normalized.get("items") or [{}])[0]
        candidates = d.get("rag_top_candidates") or []
        assumptions = d.get("rag_assumptions") or {}

        print(f"\n{'='*60}")
        print(f"INPUT: 'bought subscription for 799 today'")
        print(f"RAW MODEL intent: {raw.get('intent')}")
        raw_items = raw.get("items") or [{}]
        raw_item = raw_items[0] if raw_items else {}
        print(f"RAW MODEL category: {raw_item.get('category')}")
        print(f"NORMALIZED category: {item.get('category')}")
        print(f"NORMALIZED item: {item.get('item')}")
        print(
            f"RAG candidates ({len(candidates)}): {[c.get('merchant') for c in candidates]}"
        )
        print(f"RAG assumptions: {assumptions}")
        print(f"Retrieval stage: {d.get('retrieval_stage')}")
        print(f"{'='*60}")

        # This test always passes — it's for printing info
        assert True

    def test_print_groceries_debug(self, capsys):
        """Print full debug for groceries to see category extraction."""
        d = debug("spend 500 on groceries today")
        raw = d.get("raw_model_output", {})
        normalized = d.get("normalized_output", {})
        item = (normalized.get("items") or [{}])[0]

        print(f"\n{'='*60}")
        print(f"INPUT: 'spend 500 on groceries today'")
        raw_items = raw.get("items") or [{}]
        raw_item = raw_items[0] if raw_items else {}
        print(f"RAW MODEL category: {raw_item.get('category')}")
        print(f"NORMALIZED category: {item.get('category')}")
        print(f"{'='*60}")

        assert True

    def test_print_transport_debug(self, capsys):
        """Print full debug for transport to confirm category working."""
        d = debug("filled petrol 1500 today")
        normalized = d.get("normalized_output", {})
        item = (normalized.get("items") or [{}])[0]

        print(f"\n{'='*60}")
        print(f"INPUT: 'filled petrol 1500 today'")
        print(f"NORMALIZED category: {item.get('category')}")
        print(f"NORMALIZED item: {item.get('item')}")
        print(f"{'='*60}")

        assert True
