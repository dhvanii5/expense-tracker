import shutil
import sys
from pathlib import Path

import chromadb
import pytest

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import main as rag_main
from main import _majority_vote, build_embed_text, retrieve_assumptions


class _Vector(list):
    def tolist(self):
        return list(self)


class SimpleTestEmbedder:
    """Deterministic lightweight embedder for stable unit tests."""

    TOKENS = [
        "expense",
        "income",
        "zomato",
        "dmart",
        "myntra",
        "starbucks",
        "techcorp",
        "clientabc",
        "food",
        "groceries",
        "shopping",
        "employment",
        "freelance",
        "salary",
        "pizza",
        "burger",
        "groceries_item",
        "shoes",
        "coffee",
        "dinner",
        "freelance_item",
        "upi",
        "cash",
        "creditcard",
    ]

    def encode(self, text: str):
        t = str(text or "").lower()
        feats = [0.0] * len(self.TOKENS)

        def on(token: str) -> float:
            return 1.0 if token in t else 0.0

        feats[0] = on("expense")
        feats[1] = on("income")
        feats[2] = on("zomato")
        feats[3] = on("dmart")
        feats[4] = on("myntra")
        feats[5] = on("starbucks")
        feats[6] = on("techcorp")
        feats[7] = on("clientabc")
        feats[8] = on("category:food") + on(" food")
        feats[9] = on("category:groceries")
        feats[10] = on("category:shopping")
        feats[11] = on("category:employment")
        feats[12] = on("category:freelance")
        feats[13] = on("source:salary")
        feats[14] = on("pizza")
        feats[15] = on("burger")
        feats[16] = on("for groceries")
        feats[17] = on("shoes")
        feats[18] = on("coffee")
        feats[19] = on("dinner")
        feats[20] = on("freelance")
        feats[21] = on(" via upi")
        feats[22] = on(" via cash")
        feats[23] = on("creditcard") + on("credit card")

        return _Vector(feats)


def _mk_entry(
    intent,
    amount,
    item,
    category=None,
    merchant=None,
    payment_method=None,
    payer=None,
    source=None,
    date=None,
):
    return {
        "intent": intent,
        "items": [
            {
                "amount": amount,
                "item": item,
                "category": category,
                "merchant": merchant,
                "payment_method": payment_method,
                "payer": payer,
                "source": source,
                "currency": "INR",
                "datetime": date,
                "remarks": None,
                "bill_no": None,
            }
        ],
    }


def _flat_meta(entry: dict) -> dict:
    item = entry["items"][0]
    return {
        "intent": str(entry.get("intent") or ""),
        "amount": str(item.get("amount") or ""),
        "item": str(item.get("item") or ""),
        "category": str(item.get("category") or ""),
        "merchant": str(item.get("merchant") or ""),
        "payment_method": str(item.get("payment_method") or ""),
        "payer": str(item.get("payer") or ""),
        "source": str(item.get("source") or ""),
        "currency": str(item.get("currency") or "INR"),
        "datetime": str(item.get("datetime") or ""),
        "date": str(item.get("datetime") or ""),
        "remarks": str(item.get("remarks") or ""),
        "bill_no": str(item.get("bill_no") or ""),
    }


def _seed_transactions():
    return [
        _mk_entry(
            "expense", 200, "pizza", "Food", "Zomato", "UPI", None, None, "2024-01-01"
        ),
        _mk_entry(
            "expense", 150, "burger", "Food", "Zomato", "UPI", None, None, "2024-01-05"
        ),
        _mk_entry(
            "expense",
            500,
            "groceries",
            "Groceries",
            "DMart",
            "Cash",
            None,
            None,
            "2024-01-07",
        ),
        _mk_entry(
            "expense",
            300,
            "groceries",
            "Groceries",
            "DMart",
            "Cash",
            None,
            None,
            "2024-01-10",
        ),
        _mk_entry(
            "expense",
            999,
            "shoes",
            "Shopping",
            "Myntra",
            "CreditCard",
            None,
            None,
            "2024-01-12",
        ),
        _mk_entry(
            "expense",
            80,
            "coffee",
            "Food",
            "Starbucks",
            "UPI",
            None,
            None,
            "2024-01-14",
        ),
        _mk_entry(
            "expense",
            120,
            "coffee",
            "Food",
            "Starbucks",
            "UPI",
            None,
            None,
            "2024-01-18",
        ),
        _mk_entry(
            "income",
            50000,
            "salary",
            "Employment",
            None,
            None,
            "TechCorp",
            "Salary",
            "2024-01-31",
        ),
        _mk_entry(
            "income",
            8000,
            "freelance",
            "Freelance",
            None,
            None,
            "ClientABC",
            "Freelance",
            "2024-01-20",
        ),
        _mk_entry(
            "expense", 450, "dinner", "Food", "Zomato", "UPI", None, None, "2024-01-22"
        ),
    ]


def _retrieve_with_compat(
    extracted: dict, collection, intent: str | None = None
) -> dict:
    """Compatibility shim for both old/new retrieve_assumptions signatures."""
    try:
        result = retrieve_assumptions(extracted, collection, intent)
    except TypeError:
        result = retrieve_assumptions(extracted)

    if isinstance(result, dict) and "assumptions" in result:
        assumptions = dict(result.get("assumptions") or {})
        assumptions["_stage"] = result.get("retrieval_stage")
        return assumptions
    return dict(result or {})


@pytest.fixture(scope="session")
def rag_test_collection():
    tmp_path = Path("./test_finance_memory")
    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)

    client = chromadb.PersistentClient(path=str(tmp_path))
    try:
        client.delete_collection("test_transactions")
    except Exception:
        pass

    collection = client.create_collection(
        name="test_transactions",
        metadata={"hnsw:space": "cosine"},
    )

    old_embed_model = rag_main.embed_model
    old_collection = rag_main.collection
    old_client = rag_main.chroma_client

    rag_main.embed_model = SimpleTestEmbedder()
    rag_main.chroma_client = client
    rag_main.collection = collection

    entries = _seed_transactions()
    for idx, entry in enumerate(entries, start=1):
        text = build_embed_text(entry)
        emb = rag_main.embed_model.encode(text).tolist()
        meta = _flat_meta(entry)
        collection.add(
            ids=[f"seed-{idx}"],
            documents=[text],
            embeddings=[emb],
            metadatas=[meta],
        )

    yield collection

    rag_main.embed_model = old_embed_model
    rag_main.collection = old_collection
    rag_main.chroma_client = old_client

    try:
        client.delete_collection("test_transactions")
    except Exception:
        pass

    shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.fixture(scope="session")
def empty_test_collection():
    tmp_path = Path("./test_finance_memory_empty")
    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)

    client = chromadb.PersistentClient(path=str(tmp_path))
    try:
        client.delete_collection("test_transactions_empty")
    except Exception:
        pass

    collection = client.create_collection(
        name="test_transactions_empty",
        metadata={"hnsw:space": "cosine"},
    )

    yield collection

    try:
        client.delete_collection("test_transactions_empty")
    except Exception:
        pass
    shutil.rmtree(tmp_path, ignore_errors=True)


class TestMerchantAnchored:
    # Validates Stage 1 infers payment method from known merchant history.
    def test_1_1_known_merchant_infer_payment_method(self, rag_test_collection):
        extracted = {
            "intent": "expense",
            "items": [{"amount": 350, "item": "biryani", "merchant": "Zomato"}],
        }
        assumptions = _retrieve_with_compat(extracted, rag_test_collection, "expense")
        assert (
            assumptions.get("payment_method") == "UPI"
        ), f"Got assumptions: {assumptions}"
        assert (
            assumptions.get("payment_method_confidence", 0.0) >= 0.8
        ), f"Got assumptions: {assumptions}"
        assert (
            assumptions.get("_stage") == "merchant_anchored"
        ), f"Got assumptions: {assumptions}"

    # Validates Stage 1 infers category from known merchant history.
    def test_1_2_known_merchant_infer_category(self, rag_test_collection):
        extracted = {
            "intent": "expense",
            "items": [{"amount": 200, "item": "vegetables", "merchant": "DMart"}],
        }
        assumptions = _retrieve_with_compat(extracted, rag_test_collection, "expense")
        assert (
            assumptions.get("category") == "Groceries"
        ), f"Got assumptions: {assumptions}"
        assert (
            assumptions.get("category_confidence", 0.0) >= 0.8
        ), f"Got assumptions: {assumptions}"

    # Validates Stage 1 yields full confidence when merchant behavior is consistent.
    def test_1_3_known_merchant_consistent_payment(self, rag_test_collection):
        extracted = {
            "intent": "expense",
            "items": [{"amount": 90, "item": "latte", "merchant": "Starbucks"}],
        }
        assumptions = _retrieve_with_compat(extracted, rag_test_collection, "expense")
        assert (
            assumptions.get("payment_method") == "UPI"
        ), f"Got assumptions: {assumptions}"
        assert (
            assumptions.get("payment_method_confidence") == 1.0
        ), f"Got assumptions: {assumptions}"


class TestSemanticFallback:
    # Validates Stage 2 fallback works when merchant is unknown.
    def test_2_1_unknown_merchant_category_inferred_from_item(
        self, rag_test_collection
    ):
        extracted = {
            "intent": "expense",
            "items": [{"amount": 180, "item": "pasta", "category": "Food"}],
        }
        assumptions = _retrieve_with_compat(extracted, rag_test_collection, "expense")
        assert "payment_method" in assumptions, f"Got assumptions: {assumptions}"
        assert assumptions.get("_stage") in (
            "semantic_fallback",
            "category_anchored",
        ), f"Got assumptions: {assumptions}"

    # Validates expense retrieval never bleeds income-only fields.
    def test_2_2_intent_filter_blocks_income_fields(self, rag_test_collection):
        extracted = {"intent": "expense", "items": [{"amount": 100, "item": "coffee"}]}
        assumptions = _retrieve_with_compat(extracted, rag_test_collection, "expense")
        assert "source" not in assumptions, f"Got assumptions: {assumptions}"
        assert "payer" not in assumptions, f"Got assumptions: {assumptions}"

    # Validates low-signal unknown category query avoids unsafe assumptions.
    def test_2_3_unknown_merchant_unknown_category_low_confidence(
        self, rag_test_collection
    ):
        extracted = {
            "intent": "expense",
            "items": [{"amount": 5000, "item": "laptop", "category": "Electronics"}],
        }
        assumptions = _retrieve_with_compat(extracted, rag_test_collection, "expense")
        if assumptions:
            confidences = [
                v for k, v in assumptions.items() if k.endswith("_confidence")
            ]
            assert all(c < 0.4 for c in confidences), f"Got assumptions: {assumptions}"
        else:
            assert assumptions == {}, f"Got assumptions: {assumptions}"

    # Validates non-informative category labels do not over-filter retrieval.
    def test_2_4_other_category_does_not_block_semantic_matches(
        self, rag_test_collection
    ):
        extracted = {
            "intent": "expense",
            "items": [{"amount": 95, "item": "coffee", "category": "Other"}],
        }
        assumptions = _retrieve_with_compat(extracted, rag_test_collection, "expense")
        assert (
            assumptions.get("payment_method") == "UPI"
        ), f"Got assumptions: {assumptions}"


class TestIncome:
    # Validates income retrieval can infer source for a known payer.
    def test_3_1_known_payer_infer_source(self, rag_test_collection):
        extracted = {
            "intent": "income",
            "items": [{"amount": 52000, "payer": "TechCorp"}],
        }
        assumptions = _retrieve_with_compat(extracted, rag_test_collection, "income")
        assert assumptions.get("source") == "Salary", f"Got assumptions: {assumptions}"
        assert (
            assumptions.get("source_confidence", 0.0) >= 0.8
        ), f"Got assumptions: {assumptions}"

    # Validates income retrieval does not infer expense-only payment_method.
    def test_3_2_income_filter_blocks_expense_fields(self, rag_test_collection):
        extracted = {
            "intent": "income",
            "items": [{"amount": 10000, "payer": "NewClient"}],
        }
        assumptions = _retrieve_with_compat(extracted, rag_test_collection, "income")
        assert "payment_method" not in assumptions, f"Got assumptions: {assumptions}"


class TestSafetyRules:
    # Validates amount is never assumed by RAG.
    def test_4_1_amount_never_assumed(self, rag_test_collection):
        extracted = {"intent": "expense", "items": [{"merchant": "Zomato"}]}
        assumptions = _retrieve_with_compat(extracted, rag_test_collection, "expense")
        assert "amount" not in assumptions, f"Got assumptions: {assumptions}"

    # Validates remarks is never assumed by RAG.
    def test_4_2_remarks_never_assumed(self, rag_test_collection):
        extracted = {
            "intent": "expense",
            "items": [{"merchant": "Zomato", "item": "pizza"}],
        }
        assumptions = _retrieve_with_compat(extracted, rag_test_collection, "expense")
        assert "remarks" not in assumptions, f"Got assumptions: {assumptions}"

    # Validates empty collection returns no assumptions.
    def test_4_3_empty_db_returns_empty_assumptions(
        self, empty_test_collection, rag_test_collection
    ):
        old_collection = rag_main.collection
        rag_main.collection = empty_test_collection
        try:
            extracted = {"intent": "expense", "items": [{"merchant": "Zomato"}]}
            assumptions = _retrieve_with_compat(
                extracted, empty_test_collection, "expense"
            )
            assumptions.pop("_stage", None)
            assert assumptions == {}, f"Got assumptions: {assumptions}"
        finally:
            rag_main.collection = old_collection


class TestEmbedQuality:
    # Validates embed text generation strips null-like placeholders.
    def test_5_1_build_embed_text_strips_none_values(self):
        tx = {
            "intent": "expense",
            "items": [
                {
                    "amount": 200,
                    "item": "pizza",
                    "merchant": "Zomato",
                    "payment_method": None,
                    "payer": None,
                    "source": None,
                    "category": None,
                }
            ],
        }
        text = build_embed_text(tx)
        assert "None" not in text, f"Got assumptions: {{'embed_text': text}}"
        assert "null" not in text.lower(), f"Got assumptions: {{'embed_text': text}}"
        assert "expense" in text.lower(), f"Got assumptions: {{'embed_text': text}}"
        assert "pizza" in text.lower(), f"Got assumptions: {{'embed_text': text}}"

    # Validates embed text includes every present optional field.
    def test_5_2_build_embed_text_includes_all_present_fields(self):
        tx = {
            "intent": "expense",
            "items": [
                {
                    "amount": 500,
                    "item": "groceries",
                    "category": "Groceries",
                    "merchant": "DMart",
                    "payment_method": "Cash",
                }
            ],
        }
        text = build_embed_text(tx)
        assert "groceries" in text.lower(), f"Got assumptions: {{'embed_text': text}}"
        assert "DMart" in text, f"Got assumptions: {{'embed_text': text}}"
        assert "Cash" in text, f"Got assumptions: {{'embed_text': text}}"
        assert "Groceries" in text, f"Got assumptions: {{'embed_text': text}}"


class TestConfidenceThresholds:
    # Validates high-confidence fields remain in final assumptions.
    def test_6_1_high_confidence_fields_included(self, rag_test_collection):
        extracted = {
            "intent": "expense",
            "items": [{"merchant": "Zomato", "item": "meal"}],
        }
        assumptions = _retrieve_with_compat(extracted, rag_test_collection, "expense")
        assert "payment_method" in assumptions, f"Got assumptions: {assumptions}"
        assert (
            assumptions.get("payment_method_confidence", 0.0) >= 0.8
        ), f"Got assumptions: {assumptions}"

    # Validates majority vote includes >=0.4 and excludes below-threshold ties.
    def test_6_2_low_confidence_fields_excluded_in_majority_vote(self):
        metas = [
            {"payment_method": "UPI"},
            {"payment_method": "Cash"},
            {"payment_method": "CreditCard"},
            {"payment_method": "UPI"},
        ]
        result = _majority_vote(metas, ["payment_method"])
        assert "payment_method" in result, f"Got assumptions: {result}"
        assert (
            result.get("payment_method_confidence") == 0.5
        ), f"Got assumptions: {result}"

        metas_tied = [
            {"payment_method": "UPI"},
            {"payment_method": "Cash"},
            {"payment_method": "CreditCard"},
            {"payment_method": "NetBanking"},
        ]
        result2 = _majority_vote(metas_tied, ["payment_method"])
        assert "payment_method" not in result2, f"Got assumptions: {result2}"


class TestOptionalMerchantRanking:
    # Validates item-specific merchant selection beats a generic subscription merchant.
    def test_7_1_movie_tickets_prefers_pvr_over_netflix(self):
        live_past = [
            {
                "merchant": "Netflix",
                "item": "Subscription",
                "category": "Entertainment",
                "remarks": "Paid for Subscription at Netflix.",
                "datetime": "2024-01-01",
                "_similarity": 0.62,
            },
            {
                "merchant": "PVR",
                "item": "Movie tickets",
                "category": "Entertainment",
                "remarks": "Paid today at movie theatre",
                "datetime": "2024-01-02",
                "_similarity": 0.58,
            },
        ]

        value, score = rag_main._pick_optional_candidate(
            live_past,
            "merchant",
            "movie tickets 500 today",
            "movie tickets",
            "Entertainment",
        )

        assert value == "PVR", f"Got candidate: {(value, score)}"
        assert score >= 0.45, f"Got candidate: {(value, score)}"

    # Validates an explicit merchant mention still prefers the matching merchant.
    def test_7_2_merchant_name_in_query_prefers_netflix(self):
        live_past = [
            {
                "merchant": "Netflix",
                "item": "Subscription",
                "category": "Entertainment",
                "remarks": "Paid for Subscription at Netflix.",
                "datetime": "2024-01-03",
                "_similarity": 0.57,
            },
            {
                "merchant": "PVR",
                "item": "Movie tickets",
                "category": "Entertainment",
                "remarks": "Paid today at movie theatre",
                "datetime": "2024-01-04",
                "_similarity": 0.55,
            },
        ]

        value, score = rag_main._pick_optional_candidate(
            live_past,
            "merchant",
            "netflix 799 today",
            "Subscription",
            "Entertainment",
        )

        assert value == "Netflix", f"Got candidate: {(value, score)}"
        assert score >= 0.45, f"Got candidate: {(value, score)}"
