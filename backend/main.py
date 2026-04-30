from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
import logging
import json
import re
import uuid
from datetime import datetime, timedelta
import math
import time
import os
import warnings
import traceback
from pathlib import Path
import chromadb

_SentenceTransformer = None

try:
    from llama_cpp import Llama  # type: ignore[import-not-found]
except ModuleNotFoundError:
    Llama = None

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# ── Global variables for models and db ─────────────────────────────────────────
embed_model = None
chroma_client = None
collection = None
collection_path = None
llm = None
_embed_model_load_attempted = False
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"


def ensure_collection():
    global chroma_client, collection, collection_path
    base_dir = Path(__file__).resolve().parent
    target_path = str(base_dir / "finance_memory")

    if collection is not None and collection_path == target_path:
        return collection

    chroma_client = chromadb.PersistentClient(path=target_path)
    collection = chroma_client.get_or_create_collection(
        "transactions", metadata={"hf:space": "cosine"}
    )
    collection_path = target_path
    return collection


def ensure_embed_model():
    global embed_model, _embed_model_load_attempted, _SentenceTransformer
    if embed_model is not None:
        return embed_model
    if _embed_model_load_attempted:
        return None

    _embed_model_load_attempted = True
    if _SentenceTransformer is None:
        try:
            from sentence_transformers import (
                SentenceTransformer as sentence_transformer_cls,
            )

            _SentenceTransformer = sentence_transformer_cls
            try:
                from transformers.utils import logging as transformers_logging

                transformers_logging.set_verbosity_error()
                logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            except Exception:
                pass
        except Exception as e:
            print(f"⚠️ sentence_transformers unavailable: {e}")
            return None

    try:
        embed_model = _SentenceTransformer(EMBED_MODEL_NAME, local_files_only=True)
    except Exception:
        try:
            embed_model = _SentenceTransformer(EMBED_MODEL_NAME)
        except Exception as e:
            print(f"⚠️ Failed to initialize embedding model: {e}")
            embed_model = None
    return embed_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embed_model, chroma_client, collection, llm
    warnings.filterwarnings(
        "ignore",
        message=r"You are sending unauthenticated requests to the HF Hub.*",
    )
    # Load the local GGUF model if available; fall back to heuristics otherwise.
    if Llama is not None:
        try:
            llm = Llama(
                model_path=str(MODEL_PATH),
                n_ctx=1024,
                verbose=False,
            )
            print(f"Loaded local LLM: {MODEL_PATH}")
        except Exception as e:
            llm = None
            print(f"⚠️ Failed to load local LLM: {e}")
    else:
        llm = None
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = MODEL_DIR / "qwen3b-dhvani-q8_0.gguf"
MODEL_NAME = MODEL_PATH.name
CHATML_TEMPLATE = "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"

# Ephemeral last follow-up storage for clients that omit session_data.
# This is a best-effort fallback for single-user/dev scenarios.
last_followup_session = None
last_followup_time = None
last_followup_field = None

# Canonical schema consumed by business logic regardless of model output quirks.
CANONICAL_SCHEMA = {
    "intent": None,
    "items": [
        {
            "amount": None,
            "category": None,
            "currency": "INR",
            "item": None,
            "merchant": None,
            "payment_method": None,
            "remarks": None,
            "datetime": None,
            "bill_no": None,
            "source": None,
            "payer": None,
        }
    ],
}

FIELD_ALIASES = {
    "type": "intent",
    "transaction_type": "intent",
    "kind": "intent",
    "price": "amount",
    "cost": "amount",
    "value": "amount",
    "total": "amount",
    "date": "datetime",
    "time": "datetime",
    "timestamp": "datetime",
    "transaction_date": "datetime",
    "date_time": "datetime",
    "type_category": "category",
    "expense_category": "category",
    "spending_category": "category",
    "payment": "payment_method",
    "mode_of_payment": "payment_method",
    "payment_mode": "payment_method",
    "pay_method": "payment_method",
    "paid_via": "payment_method",
    "store": "merchant",
    "vendor": "merchant",
    "shop": "merchant",
    "place": "merchant",
    "location": "merchant",
    "description": "item",
    "product": "item",
    "service": "item",
    "what": "item",
    "income_source": "source",
    "from": "source",
    "salary_source": "source",
}

INTENT_ALIASES = {
    "expense": "expense",
    "expenses": "expense",
    "debit": "expense",
    "spending": "expense",
    "spent": "expense",
    "purchase": "expense",
    "income": "income",
    "credit": "income",
    "earning": "income",
    "earnings": "income",
    "salary": "income",
    "receipt": "income",
    "query": "query",
    "question": "query",
    "analytics": "query",
}

REQUIRED_FIELDS = ["intent", "amount", "datetime"]
OPTIONAL_FIELDS = ["category", "currency", "item", "merchant", "remarks", "bill_no"]

OPTIONAL_FOLLOWUP_FIELDS = {
    "expense": ["merchant", "payment_method", "bill_no"],
    "income": ["payment_method", "payer"],
}

SKIP_ANSWER_TOKENS = {"skip", "none", "na", "n/a", "no", "not now", "later"}
YES_ANSWER_TOKENS = {"yes", "y", "yeah", "yep", "correct", "right", "ok", "okay"}
NO_ANSWER_TOKENS = {"no", "n", "nope", "wrong", "incorrect"}

# Income/source phrases that should not be treated as payer names.
INCOME_SOURCE_LIKE_TERMS = {
    "income",
    "salary",
    "bonus",
    "refund",
    "cashback",
    "interest",
    "investment",
    "business",
    "rent",
    "gift",
    "freelance",
    "freelancing",
    "freelancer",
    "project",
    "projects",
    "gig",
    "payment",
    "payments",
    "client",
    "clients",
}

# Minimum semantic similarity for keyword-boosted retrieval.
ASSUMPTION_SIMILARITY_THRESHOLD = 0.55
SEMANTIC_ONLY_THRESHOLD = 0.35

# ── FIX 2: Canonical null-like sentinel set (used everywhere)
FILLER_VALUES = {
    "unknown",
    "not specified",
    "not mentioned",
    "n/a",
    "none",
    "null",
    "",
    "unspecified",
}


class ChatRequest(BaseModel):
    message: str
    session_data: Optional[dict] = None
    followup_field: Optional[str] = None
    assumption_action: Optional[str] = None
    edited_assumptions: Optional[dict] = None
    rejected_assumptions: Optional[dict] = None
    selected_assumption_options: Optional[dict] = None


class AnalyticsFilters(BaseModel):
    time_range: str
    category: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class AnalyticsRequest(BaseModel):
    query_type: str
    filters: AnalyticsFilters


class SaveRequest(BaseModel):
    entry: dict


MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

# ── Single source of truth for category → keyword mapping ──────────────────────
# Used by: infer_expense_category(), prepare_chat_outcome(), extract_category()
CATEGORY_KEYWORDS = {
    "Groceries": [
        "grocery",
        "groceries",
        "vegetable",
        "vegetables",
        "veggies",
        "kirana",
        "rice",
        "dal",
        "fruits",
        "atta",
        "flour",
        "oil",
        "sugar",
        "milk",
        "bread",
        "eggs",
    ],
    "Food": [
        "food",
        "lunch",
        "dinner",
        "breakfast",
        "coffee",
        "tea",
        "snack",
        "pizza",
        "burger",
        "biryani",
        "meal",
        "cafe",
        "restaurant",
        "zomato",
        "swiggy",
        "starbucks",
    ],
    "Transport": [
        "petrol",
        "diesel",
        "fuel",
        "uber",
        "ola",
        "cab",
        "taxi",
        "auto",
        "bus",
        "train",
        "metro",
        "flight",
        "ticket",
        "rapido",
        "ride",
        "travel",
        "commute",
    ],
    "Health": [
        "medicine",
        "medicines",
        "pharmacy",
        "doctor",
        "hospital",
        "gym",
        "workout",
        "fitness",
        "lab",
        "test",
        "clinic",
        "tablet",
        "tablets",
        "pills",
        "vitamins",
        "supplements",
        "health",
    ],
    "Entertainment": [
        "subscription",
        "netflix",
        "spotify",
        "hotstar",
        "prime",
        "movie",
        "cinema",
        "ott",
        "gaming",
        "stream",
        "tickets",
    ],
    "Shopping": [
        "shopping",
        "clothes",
        "clothing",
        "shoe",
        "shoes",
        "sneakers",
        "shirt",
        "tshirt",
        "dress",
        "watch",
        "bag",
        "jeans",
        "kurta",
        "jacket",
        "cap",
        "accessories",
        "amazon",
        "flipkart",
        "myntra",
        "nykaa",
        "ajio",
        "croma",
    ],
    "Bills": [
        "bill",
        "electricity",
        "water",
        "recharge",
        "rent",
        "internet",
        "mobile recharge",
        "utility",
        "wifi",
        "gas",
        "broadband",
    ],
    "Education": ["education", "tuition", "course", "school", "college", "books"],
    "Travel": ["travel", "hotel", "booking", "airbnb", "holiday", "trip"],
    "Rent": ["rent", "lease", "housing"],
}

# Flat list derived from the canonical dict for quick membership checks.
KNOWN_CATEGORIES = [c.lower() for c in CATEGORY_KEYWORDS.keys()]

# Normalized payment method lookup (handles typos + aliases)
PAYMENT_ALIASES = {
    "upi": "UPI",
    "gpay": "UPI",
    "google pay": "UPI",
    "phonepe": "UPI",
    "paytm": "UPI",
    "bhim": "UPI",
    "cash": "Cash",
    "card": "Card",
    "credit card": "Card",
    "debit card": "Card",
    "net banking": "Bank Transfer",
    "netbanking": "Bank Transfer",
    "bank transfer": "Bank Transfer",
    "neft": "Bank Transfer",
    "imps": "Bank Transfer",
    "rtgs": "Bank Transfer",
    "wallet": "Wallet",
    "cheque": "Cheque",
    "check": "Cheque",
    "emi": "EMI",
    "online": "Online",
}


def normalize_payment_method(raw: str) -> Optional[str]:
    """Canonicalize a freeform payment string to a known label."""
    if not raw:
        return None
    lower = raw.strip().lower()
    if lower in FILLER_VALUES:
        return None
    # Exact match first
    if lower in PAYMENT_ALIASES:
        return PAYMENT_ALIASES[lower]
    # Substring match
    for key, val in PAYMENT_ALIASES.items():
        if key in lower:
            return val
    # Return title-cased original rather than raw garbage
    return raw.strip().title()


def _normalize_answer_text(answer: str) -> str:
    cleaned = re.sub(r"[^a-z\s]", " ", str(answer or "").lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def is_affirmative_answer(answer: str) -> bool:
    cleaned = _normalize_answer_text(answer)
    if not cleaned:
        return False
    first = cleaned.split()[0]
    if first in YES_ANSWER_TOKENS:
        return True
    return bool(re.fullmatch(r"ye+s+", first))


def is_negative_answer(answer: str) -> bool:
    cleaned = _normalize_answer_text(answer)
    if not cleaned:
        return False
    first = cleaned.split()[0]
    if first in NO_ANSWER_TOKENS:
        return True
    return bool(re.fullmatch(r"no+|nah+|nop+e?", first))


def is_skip_answer(answer: str) -> bool:
    cleaned = _normalize_answer_text(answer)
    if not cleaned:
        return False
    first = cleaned.split()[0]
    return first in SKIP_ANSWER_TOKENS


def empty_item() -> dict:
    return {
        "amount": None,
        "category": None,
        "currency": "INR",
        "item": None,
        "merchant": None,
        "payment_method": None,
        "remarks": None,
        "datetime": None,
        "bill_no": None,
        "source": None,
        "payer": None,
    }


def remap_keys(obj: dict, aliases: dict) -> dict:
    """Rename any aliased keys to canonical names."""
    if not isinstance(obj, dict):
        return {}
    result = {}
    for k, v in obj.items():
        key = str(k).lower().strip()
        canonical_key = aliases.get(key, k)
        result[canonical_key] = v
    return result


def coerce_amount(val) -> Optional[float]:
    """Parse amount from numeric or text formats returned by models."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        value = float(val)
        return value if value > 0 else None
    if isinstance(val, str):
        cleaned = val.strip().lower()
        if cleaned in FILLER_VALUES:
            return None
        cleaned = re.sub(r"[₹$€£]", "", cleaned)
        cleaned = re.sub(r"\b(rs\.?|inr|rupees?|usd|eur)\b", "", cleaned)
        cleaned = cleaned.replace(",", "").strip()

        k_match = re.search(r"([\d.]+)\s*k\b", cleaned)
        lakh_match = re.search(r"([\d.]+)\s*(lakh|lac)\b", cleaned)
        if k_match:
            try:
                return float(k_match.group(1)) * 1000
            except ValueError:
                return None
        if lakh_match:
            try:
                return float(lakh_match.group(1)) * 100000
            except ValueError:
                return None

        num_match = re.search(r"[\d.]+", cleaned)
        if num_match:
            try:
                value = float(num_match.group(0))
                return value if value > 0 else None
            except ValueError:
                return None
    return None


def coerce_datetime(val, user_input: str) -> Optional[str]:
    """Normalize datetime to ISO-like output using value or user-input hints."""

    def parse_relative_datetime_hint(text: str) -> Optional[str]:
        lower = (text or "").lower()
        now = datetime.now()

        if "today" in lower or "now" in lower or "just now" in lower:
            return now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        if "yesterday" in lower:
            return (
                (now - timedelta(days=1))
                .replace(hour=0, minute=0, second=0, microsecond=0)
                .isoformat()
            )
        if "tomorrow" in lower:
            return (
                (now + timedelta(days=1))
                .replace(hour=0, minute=0, second=0, microsecond=0)
                .isoformat()
            )
        if "this week" in lower:
            return (
                (now - timedelta(days=now.weekday()))
                .replace(hour=0, minute=0, second=0, microsecond=0)
                .isoformat()
            )

        # Supports numeric and worded forms like "2 months back" and "two years ago".
        word_to_num = {
            "a": 1,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
        }
        ago_match = re.search(
            r"\b(\d+|a|one|two|three|four|five|six|seven|eight|nine|ten)\s*"
            r"(day(?:s|ss)?|week(?:s|ss)?|month(?:s|ss)?|year(?:s|ss)?)\s+(ago|back)\b",
            lower,
        )
        if ago_match:
            raw_num = ago_match.group(1)
            value = int(raw_num) if raw_num.isdigit() else word_to_num.get(raw_num, 1)
            unit = ago_match.group(2)
            if unit.startswith("day"):
                target = now - timedelta(days=value)
            elif unit.startswith("week"):
                target = now - timedelta(weeks=value)
            elif unit.startswith("month"):
                target = now - timedelta(days=30 * value)
            else:
                target = now - timedelta(days=365 * value)
            return target.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

        weekdays = [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ]
        for idx, day_name in enumerate(weekdays):
            if f"last {day_name}" in lower:
                days_behind = (now.weekday() - idx) % 7 or 7
                return (
                    (now - timedelta(days=days_behind))
                    .replace(hour=0, minute=0, second=0, microsecond=0)
                    .isoformat()
                )

        return None

    for raw_text in [val, user_input]:
        if not isinstance(raw_text, str):
            continue
        text = raw_text.strip()
        if not text:
            continue
        if text.lower() in FILLER_VALUES:
            continue

        if re.match(r"\d{4}-\d{2}-\d{2}", text):
            return text

        lower = text.lower()
        parsed_relative = parse_relative_datetime_hint(lower)
        if parsed_relative:
            return parsed_relative

        parsed = parse_explicit_date(lower)
        if parsed:
            return parsed[0]

    return None


def coerce_intent(val, user_input: str) -> Optional[str]:
    """Normalize intent into expense/income/query with user-input fallback."""
    if val is None:
        return infer_transaction_intent(user_input)
    text = str(val).strip().lower()
    if text in FILLER_VALUES:
        return infer_transaction_intent(user_input)
    return INTENT_ALIASES.get(text) or infer_transaction_intent(user_input)


def normalize_model_output(raw: dict, user_input: str) -> dict:
    """Convert model-specific output shapes into canonical extraction schema."""
    if not isinstance(raw, dict):
        fallback = {
            "intent": infer_transaction_intent(user_input),
            "items": [empty_item()],
        }
        return apply_transaction_hints(user_input, fallback)

    data = remap_keys(raw, FIELD_ALIASES)
    data = normalize_null_like(data)

    item_field_names = {
        "amount",
        "category",
        "currency",
        "item",
        "merchant",
        "payment_method",
        "remarks",
        "datetime",
        "bill_no",
        "source",
        "payer",
    }

    # Handle flat outputs by wrapping item fields into items[0].
    if "items" not in data:
        item_data = {}
        non_item_data = {}
        for k, v in data.items():
            canonical_key = FIELD_ALIASES.get(str(k).lower().strip(), k)
            if canonical_key in item_field_names:
                item_data[canonical_key] = v
            else:
                non_item_data[canonical_key] = v
        if item_data:
            data = {**non_item_data, "items": [item_data]}

    if isinstance(data.get("items"), dict):
        data["items"] = [data["items"]]

    items_list = data.get("items")
    if not isinstance(items_list, list) or not items_list:
        data["items"] = [empty_item()]
    else:
        normalized_items = []
        for item in items_list:
            if not isinstance(item, dict):
                normalized_items.append(empty_item())
                continue
            remapped_item = remap_keys(item, FIELD_ALIASES)
            merged_item = {**empty_item(), **normalize_null_like(remapped_item)}
            normalized_items.append(merged_item)
        data["items"] = normalized_items or [empty_item()]

    item = {**empty_item(), **data["items"][0]}
    item["amount"] = coerce_amount(item.get("amount"))
    if not amount_in_input(user_input, item.get("amount")):
        item["amount"] = None

    item["datetime"] = coerce_datetime(
        item.get("datetime") or item.get("date"), user_input
    )
    item["currency"] = str(item.get("currency") or "INR").strip().upper() or "INR"

    for field in [
        "category",
        "item",
        "merchant",
        "remarks",
        "bill_no",
        "source",
        "payer",
    ]:
        value = item.get(field)
        if is_empty(value):
            item[field] = None
        else:
            item[field] = str(value).strip()

    # ── FIX: Automatically map original user text to remarks if model left it blank
    if not item.get("remarks") and user_input and len(user_input.strip()) >= 2:
        item["remarks"] = user_input.strip().capitalize()

    raw_payment_method = item.get("payment_method")
    # ALWAYS clear payment_method if user didn't explicitly mention it.
    # This stops the model's hallucinated "cash" from leaking through.
    if not has_explicit_payment_method(user_input):
        item["payment_method"] = None
    elif not is_empty(raw_payment_method):
        # User mentioned a payment method — extract it directly from their input
        payment_match = re.search(
            r"\b(cash|upi|gpay|phonepe|paytm|card|credit card|debit card|"
            r"bank transfer|net ?banking|netbanking|neft|imps|wallet|online|cheque|emi)\b",
            user_input.lower(),
        )
        if payment_match:
            item["payment_method"] = normalize_payment_method(payment_match.group(0))
        else:
            item["payment_method"] = normalize_payment_method(str(raw_payment_method))
    else:
        item["payment_method"] = None

    data["intent"] = coerce_intent(data.get("intent"), user_input)
    if data.get("intent") == "query":
        inferred = infer_transaction_intent(user_input)
        data["intent"] = inferred if inferred else "expense"

    if data.get("intent") == "income":
        item = data["items"][0]
        if is_empty(item.get("source")):
            # ── FIX: Automatically 'assume' or identify source for income (no follow-up asked)
            text = user_input.lower()
            if any(w in text for w in ["salary", "paycheck", "wages"]):
                item["source"] = "Salary"
            elif any(w in text for w in ["freelance", "project", "gig"]):
                item["source"] = "Freelance"
            elif any(w in text for w in ["bonus", "incentive"]):
                item["source"] = "Bonus"
            elif any(w in text for w in ["gift", "present"]):
                item["source"] = "Gift"
            elif any(w in text for w in ["someone", "friend", "returned"]):
                item["source"] = "Other Income"
            elif any(w in text for w in ["dividend"]):
                item["source"] = "Dividend"
            elif any(w in text for w in ["interest"]):
                item["source"] = "Interest"
            elif any(w in text for w in ["investment", "stock", "return", "gain"]):
                item["source"] = "Investment"
            elif any(w in text for w in ["business", "profit", "sales"]):
                item["source"] = "Business"
            elif any(w in text for w in ["refund", "cashback", "cash back"]):
                item["source"] = "Refund"
            elif "rent" in text:
                item["source"] = "Rent"
            elif "scholarship" in text:
                item["source"] = "Scholarship"
            elif not is_empty(item.get("category")):
                item["source"] = item["category"]
            else:
                item["source"] = "Income"

        normalized_source = normalize_income_source_value(
            item.get("source") or item.get("category"), user_input
        )
        if normalized_source:
            item["source"] = normalized_source
            if is_empty(item.get("category")):
                item["category"] = normalized_source
            if is_empty(item.get("item")):
                item["item"] = normalized_source
        elif str(item.get("source") or "").strip().lower() in {
            "income",
            "other income",
            "general income",
        }:
            item["source"] = None
            if str(item.get("category") or "").strip().lower() == "income":
                item["category"] = None
            if str(item.get("item") or "").strip().lower() == "income":
                item["item"] = None

    data["items"] = [item]
    return apply_transaction_hints(user_input, data)


def normalize_null_like(value):
    if isinstance(value, dict):
        return {k: normalize_null_like(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_null_like(v) for v in value]
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in FILLER_VALUES:
        return None
    return value


def extract_json_object(raw: str) -> str:
    start = raw.find("{")
    if start == -1:
        return raw.strip()
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : idx + 1]
    return raw[start:].strip()


def repair_json_string(raw: str) -> str:
    repaired = extract_json_object(raw)
    repaired = repaired.replace(": None", ": null")
    repaired = re.sub(r":\s*([a-zA-Z_]\w*)\s*([,}])", r': "\1"\2', repaired)
    open_braces = repaired.count("{")
    close_braces = repaired.count("}")
    if open_braces > close_braces:
        repaired += "}" * (open_braces - close_braces)
    open_brackets = repaired.count("[")
    close_brackets = repaired.count("]")
    if open_brackets > close_brackets:
        repaired += "]" * (open_brackets - close_brackets)
    return repaired.strip()


def extract_category(text: str) -> Optional[str]:
    for candidate in KNOWN_CATEGORIES:
        if re.search(rf"\b{re.escape(candidate)}\b", text):
            return candidate.title()
    return None


def parse_explicit_date(text: str) -> Optional[tuple[str, str]]:
    normalized = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", text.lower())
    patterns = [
        r"\b(?P<day>\d{1,2})\s+(?P<month>[a-z]+)(?:\s+(?P<year>\d{4}))?\b",
        r"\b(?P<month>[a-z]+)\s+(?P<day>\d{1,2})(?:\s+(?P<year>\d{4}))?\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue
        month_name = match.group("month")
        month = MONTHS.get(month_name)
        if not month:
            continue
        day = int(match.group("day"))
        year = (
            int(match.group("year"))
            if match.groupdict().get("year")
            else datetime.now().year
        )
        try:
            parsed = datetime(year, month, day)
        except ValueError:
            continue
        start = parsed.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        end = parsed.replace(
            hour=23, minute=59, second=59, microsecond=999999
        ).isoformat()
        return start, end
    return None


def has_amount(text: str) -> bool:
    return bool(
        re.search(r"\b\d+(?:[.,]\d+)?\s*(k|lakh|lakhs|rs|inr)?\b", text.lower())
    )


def starts_with_phrase(text: str, phrases: list[str]) -> bool:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    return any(normalized.startswith(phrase) for phrase in phrases)


def looks_like_transaction_statement(text: str) -> bool:
    transaction_starters = [
        "spent",
        "spend",
        "paid",
        "pay",
        "received",
        "recieved",
        "got",
        "earned",
        "income",
        "salary",
        "bought",
        "i spent",
        "i spend",
        "i paid",
        "i pay",
        "i received",
        "i recieved",
        "i got",
        "i earned",
        "my salary",
        "ordered",
        "order",
        "i ordered",
        "recharge",
        "recharged",
        "subscribed",
        "booked",
        "purchased",
        "purchase",
        "bout",
    ]
    return starts_with_phrase(text, transaction_starters)


def infer_transaction_intent(text: str) -> Optional[str]:
    normalized = (text or "").lower()

    # Pattern-first handling for person-to-person transfers and repayments.
    if has_amount(normalized):
        if re.search(r"\b(receive|received|recieve|recieved)\b", normalized):
            return "income"
        if re.search(r"\b(gave|sent|paid)\s+me\b", normalized):
            return "income"
        if re.search(
            r"\b(paid me back|sent me back|gave me back|paid me|sent me|credited to me|receive from|received from)\b",
            normalized,
        ):
            return "income"
        if re.search(r"\b(refunded|refund received|cashback received)\b", normalized):
            return "income"
        if (
            re.search(r"\b(gave|sent|paid)\s+[a-z][a-z0-9&'.\- ]{1,30}\b", normalized)
            and " me " not in f" {normalized} "
        ):
            return "expense"

    income_starters = [
        "receive",
        "received",
        "recieved",
        "i receive",
        "got",
        "earned",
        "income",
        "salary",
        "i received",
        "i recieved",
        "i got",
        "i earned",
        "my salary",
        "got salary",
        "got my salary",
        "got paid",
        "got bonus",
        "credited",
        "salary credited",
        "amount credited",
        "freelance",
        "payment received",
        "got payment",
        "refunded",
        "cashback",
    ]
    expense_starters = [
        "spent",
        "spend",
        "paid",
        "pay",
        "bought",
        "filled",
        "fill",
        "ordered",
        "order",
        "purchased",
        "purchase",
        "recharge",
        "recharged",
        "subscribed",
        "booked",
        "i spent",
        "i spend",
        "i paid",
        "i pay",
        "i bought",
        "gave",
        "i gave",
        "sent",
        "i sent",
        "gym membership",
        "bout",
    ]
    if starts_with_phrase(text, income_starters):
        return "income"
    if starts_with_phrase(text, expense_starters):
        return "expense"

    if has_amount(normalized):
        income_markers = [
            "receive",
            "salary",
            "received",
            "credited",
            "income",
            "bonus",
            "refund",
            "cashback",
            "freelance",
            "payment",
            "gave me",
            "paid me back",
            "sent me",
        ]
        expense_markers = [
            "petrol",
            "fuel",
            "subscription",
            "medicine",
            "groceries",
            "food",
            "uber",
            "ola",
            "zomato",
            "swiggy",
            "recharge",
            "bill",
            "netflix",
            "spotify",
            "coffee",
            "starbucks",
            "gym",
            "membership",
            "gave",
            "sent",
        ]
        if any(m in normalized for m in income_markers):
            return "income"
        all_expense_keywords = {
            k for words in CATEGORY_KEYWORDS.values() for k in words
        }
        if any(k in normalized for k in all_expense_keywords):
            return "expense"
        if any(m in normalized for m in expense_markers):
            return "expense"

    return None


def infer_income_source(text: str) -> Optional[str]:
    normalized = text.lower()
    source_keywords = {
        "salary": "Salary",
        "bonus": "Bonus",
        "refund": "Refund",
        "interest": "Interest",
        "freelance": "Freelance",
        "freelancing": "Freelance",
        "freelancer": "Freelance",
        "project": "Freelance",
        "gig": "Freelance",
        "cashback": "Cashback",
        "gift": "Gift",
        "rent": "Rent",
    }
    for keyword, label in source_keywords.items():
        if keyword in normalized:
            return label
    return None


def normalize_income_source_value(
    raw_source: Optional[str], user_text: str
) -> Optional[str]:
    """Normalize source labels and prefer user-text inference over model mashups."""
    inferred = infer_income_source(user_text)
    if inferred:
        return inferred

    normalized = str(raw_source or "").strip().lower()
    if not normalized or normalized in FILLER_VALUES:
        return None
    if normalized in {"income", "other income", "other", "general income"}:
        return None

    if "cashback" in normalized and "refund" not in normalized:
        return "Cashback"
    if "refund" in normalized and "cashback" not in normalized:
        return "Refund"
    if "refund" in normalized and "cashback" in normalized:
        return "Refund"

    aliases = {
        "salary": "Salary",
        "bonus": "Bonus",
        "refund": "Refund",
        "interest": "Interest",
        "freelance": "Freelance",
        "freelancing": "Freelance",
        "freelancer": "Freelance",
        "project": "Freelance",
        "gig": "Freelance",
        "cashback": "Cashback",
        "gift": "Gift",
        "rent": "Rent",
    }
    for key, label in aliases.items():
        if key in normalized:
            return label

    return str(raw_source).strip().title()


def infer_expense_category(text: str, item_name: Optional[str] = None) -> Optional[str]:
    """Infer an expense category from user text or item label."""
    merged = f"{text or ''} {item_name or ''}".lower()

    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(re.search(rf"\b{re.escape(k)}\b", merged) for k in keywords):
            return category

    explicit = extract_category(merged)
    return explicit if explicit else None


def infer_income_payer(text: str) -> Optional[str]:
    """Infer payer name from patterns like 'X paid me' or 'X sent me'."""
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    if not normalized:
        return None

    # Pattern: [Name] paid me / [Name] sent me
    match_payer = re.search(
        r"^([a-zA-Z0-9][a-zA-Z0-9&'.\- ]{1,30}?)\s+(?:paid|sent|gave)\s+me\b",
        normalized,
        re.IGNORECASE,
    )
    if match_payer:
        payer = match_payer.group(1).strip(" .,-")
        if payer and payer.lower() not in {"someone", "friend", "i", "he", "she"}:
            return payer.title()

    # Pattern: received/got/from [Name]
    match_from = re.search(
        r"\b(?:from)\s+([a-zA-Z][a-zA-Z0-9&'.\- ]{1,30}?)(?=\s+(?:today|yesterday|for|via|using|with|by|on|in|rs\.?|inr|rupees?|\d)|$)",
        normalized,
        re.IGNORECASE,
    )
    if match_from:
        payer = match_from.group(1).strip(" .,-")
        payer_lower = payer.lower()
        if payer and payer_lower not in {"someone", "friend", "i", "he", "she"}:
            if payer_lower not in INCOME_SOURCE_LIKE_TERMS:
                return payer.title()

    return infer_explicit_merchant(text)


def infer_explicit_merchant(text: str) -> Optional[str]:
    """Infer merchant name from common patterns like 'from X' / 'at X'."""
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    if not normalized:
        return None

    # Person-to-person expense pattern: "paid ... (back) to Rahul"
    # This avoids irrelevant merchant suggestions (e.g. Starbucks) for repayments.
    pay_to_pattern = (
        r"\b(?:paid|pay|sent|send|gave|give|transferred|transfer|returned|return)\b"
        r"(?:\s+\d+(?:[.,]\d+)?(?:\s*(?:rs\.?|inr|rupees?))?)?"
        r"(?:\s+back)?\s+to\s+([a-zA-Z][a-zA-Z0-9&'.\- ]{1,40}?)"
        r"(?=\s+(?:today|yesterday|for|via|using|with|by|on|in|at|rs\.?|inr|rupees?|\d)|$)"
    )
    pay_to_match = re.search(pay_to_pattern, normalized, re.IGNORECASE)
    if pay_to_match:
        merchant = pay_to_match.group(1).strip(" .,-")
        if merchant and merchant.lower() not in {
            "me",
            "myself",
            "him",
            "her",
            "them",
            "someone",
        }:
            return merchant.title()

    # Prefer structured capture after 'from'/'at'
    pattern = (
        r"\b(?:from|at)\s+([a-zA-Z0-9][a-zA-Z0-9&'.\- ]{1,40}?)"
        r"(?=\s+(?:today|yesterday|for|via|using|with|by|on|in|paid|spent|bought|ordered|"
        r"received|got|rs\.?|inr|rupees?|\d)|$)"
    )
    match = re.search(pattern, normalized, re.IGNORECASE)
    if match:
        merchant = match.group(1).strip(" .,-")
        if merchant and merchant.lower() not in {
            "my friend",
            "friend",
            "dad",
            "mom",
            "brother",
            "sister",
            "him",
            "her",
            "them",
        }:
            return merchant

    # Brand fallback for short messages without prepositions.
    known_brands = [
        "swiggy",
        "zomato",
        "netflix",
        "spotify",
        "amazon",
        "flipkart",
        "dmart",
        "apollo",
        "uber",
        "ola",
        "rapido",
        "hotstar",
    ]
    lower = normalized.lower()
    for brand in known_brands:
        if re.search(rf"\b{re.escape(brand)}\b", lower):
            return brand.title()
    return None


def normalize_expense_category_value(
    raw_category: Optional[str], user_text: str, item_name: Optional[str] = None
) -> Optional[str]:
    """Normalize model category values to canonical labels used by tests/business logic."""
    text = str(raw_category or "").strip().lower()
    aliases = {
        "grocery": "Groceries",
        "groceries": "Groceries",
        "food": "Food",
        "dining": "Food",
        "transportation": "Transport",
        "transport": "Transport",
        "healthcare": "Health",
        "medical": "Health",
        "medicine": "Health",
        "medicines": "Health",
        "entertainment": "Entertainment",
        "subscription": "Entertainment",
        "ott": "Entertainment",
        "shopping": "Shopping",
        "bills": "Bills",
        "utility": "Bills",
        "utilities": "Bills",
    }
    if text in aliases:
        return aliases[text]

    inferred = infer_expense_category(user_text, item_name)
    if inferred:
        return inferred

    return raw_category if raw_category else None


def has_explicit_payment_method(text: str) -> bool:
    payment_pattern = r"\b(cash|upi|gpay|phonepe|paytm|card|credit card|debit card|bank transfer|net ?banking|netbanking|neft|imps|rtgs|wallet|online|cheque|emi)\b"
    return bool(re.search(payment_pattern, (text or "").lower()))


def has_explicit_remarks(text: str) -> bool:
    normalized = (text or "").lower()
    markers = ["note", "remark", "comment", "because", "reason"]
    return any(marker in normalized for marker in markers)


def has_explicit_bill_reference(text: str) -> bool:
    normalized = (text or "").lower()
    return bool(
        re.search(
            r"\b(?:bill|invoice|receipt|reciept|recipt|receit|txn|transaction|order)\b",
            normalized,
        )
        or re.search(r"#\s*[a-z0-9\-/]{2,}", normalized)
    )


def infer_expense_item(text: str) -> Optional[str]:
    """Infer a likely explicit expense item from natural language input."""
    normalized = re.sub(r"\s+", " ", (text or "").lower()).strip()
    if not normalized:
        return None

    stopwords = {
        "today",
        "yesterday",
        "tomorrow",
        "at",
        "from",
        "via",
        "using",
        "with",
        "by",
        "in",
        "on",
        "for",
        "and",
        "to",
        "the",
        "a",
        "an",
        "my",
        "our",
        "his",
        "her",
        "their",
        "took",
        "got",
        "had",
        "went",
        "was",
    }
    verbs = (
        r"(?:spent|spend|paid|pay|bought|buy|ordered|order|purchased|purchase|took|got)"
    )

    patterns = [
        # Pattern 1: "spent 300 on/for ITEM today" - most specific
        rf"\b{verbs}\b(?:\s+\d+(?:[.,]\d+)?(?:\s*(?:k|lakh|lac|rs\.?|inr|rupees?))?)?\s+(?:on|for)\s+([a-z][a-z\s]{{1,40}}?)(?:\s+(?:today|yesterday|at|from|via|using|with|by|morning|evening|night|afternoon)\b|$)",
        # Pattern 2: Catch "ITEM for [amount]" - e.g., "pizza for 500" (avoid leading verbs)
        r"(?:^|[\.!\s])([a-z]{2}[a-z\s]{1,35}?)\s+for\s+(?:\d+(?:[.,]\d+)?(?:\s*(?:k|lakh|lac|rs\.?|inr|rupees?))?)(?:\s+(?:today|yesterday|morning|evening|night)\b|$)",
        # Pattern 3: "spent ITEM today/yesterday"
        rf"\b{verbs}\b\s+([a-z][a-z\s]{{1,40}}?)(?:\s+(?:today|yesterday|at|from|via|using|with|by|morning|evening|night|afternoon)\b|$)",
        # Pattern 4: "for ITEM" at start or after keywords (standalone)
        r"^(?:for|on)\s+([a-z]{2}[a-z\s]{1,35}?)(?:\s+(?:today|yesterday|\d+|morning|evening|night)\b|$)",
        r"(?:\s|^)(?:for|on)\s+([a-z]{2}[a-z\s]{1,35}?)(?:\s+(?:today|yesterday|morning|evening|night)\b|$)",
        # Pattern 5: Common transaction nouns (ride, movie, food, etc.)
        r"\b(uber|ola|cab|taxi|auto|bus|train|movie|cinema|food|lunch|dinner|breakfast|coffee|tea|petrol|gas|grocery|groceries|gym|workout|book|ticket|subscription|music|medicine|medicines)\b",
        # Pattern 6: "a/an ITEM" pattern catch
        r"(?:a|an)\s+([a-z][a-z\s]{{1,35}}?)(?:\s+(?:today|yesterday|morning|evening)\b|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue
        phrase = re.sub(r"\s+", " ", match.group(1)).strip()
        tokens = [t for t in phrase.split() if t and t not in stopwords]
        if not tokens:
            continue
        # Remove leading verbs if present
        verbs_list = [
            "spent",
            "spend",
            "paid",
            "pay",
            "bought",
            "buy",
            "ordered",
            "order",
            "purchased",
            "purchase",
            "took",
            "got",
            "ordered",
        ]
        while tokens and tokens[0].lower() in verbs_list:
            tokens.pop(0)

        if not tokens:
            continue

        # Keep concise labels for UI cards and retrieval matching.
        result = " ".join(tokens[:3]).title()
        # Ensure we don't return time-related tokens
        if not any(t in result.lower() for t in ["am", "pm", "oclock", "o'clock"]):
            return result

    return None


def infer_explicit_amount(text: str) -> Optional[float]:
    """Extract an explicit transaction amount from free-form user text."""
    if not text:
        return None
    normalized = text.lower()
    match = re.search(
        r"(?:₹|rs\.?|inr)?\s*(\d+(?:[.,]\d+)?)\s*(k|lakh|lac|rupees?)?\b",
        normalized,
    )
    if not match:
        return None

    raw = match.group(1).replace(",", "")
    unit = (match.group(2) or "").strip().lower()
    try:
        value = float(raw)
    except ValueError:
        return None

    if unit == "k":
        value *= 1000
    elif unit in {"lakh", "lac"}:
        value *= 100000

    return value if value > 0 else None


def resolve_best_item(user_input: str, model_item: Optional[str]) -> Optional[str]:
    """Pick the best item between model output and user-text inference.

    Priority:
    1. Explicit user-mentioned items (e.g., "on coffee", "for pizza")
    2. Model extraction if high-quality
    3. None if both weak or absent
    """
    # Priority 1: Extract from explicit user patterns
    inferred = infer_expense_item(user_input)
    if inferred:
        return inferred

    # Priority 2: Validate model output
    model_cleaned = str(model_item or "").strip() if model_item else None
    if model_cleaned and model_cleaned.lower() not in FILLER_VALUES:
        # Model gave us something; assess quality
        words = model_cleaned.lower().split()
        # Reject single generic words (too vague)
        if len(words) == 1 and words[0] in {
            "item",
            "thing",
            "transaction",
            "payment",
            "expense",
        }:
            return None
        # Keep multi-word items as they're usually specific
        if len(words) >= 2:
            return model_cleaned
        # Single word: keep if it's a known item category
        known_items = {
            "coffee",
            "tea",
            "food",
            "lunch",
            "dinner",
            "breakfast",
            "snack",
            "groceries",
            "petrol",
            "gas",
            "fuel",
            "uber",
            "ola",
            "cab",
            "taxi",
            "movie",
            "cinema",
            "book",
            "app",
            "subscription",
            "gym",
            "workout",
            "water",
            "juice",
            "beer",
            "wine",
            "alcohol",
            "clothes",
            "shoes",
        }
        if words[0] in known_items:
            return model_cleaned
        # Default: reject single vague word
        return None

    return None


def auto_generate_remarks(user_input: str, data: dict, item: dict) -> Optional[str]:
    """Generate a concise, user-facing remark when none is explicitly provided."""
    intent = str(data.get("intent") or "").strip().lower()
    item_name = str(item.get("item") or "").strip()
    merchant = str(item.get("merchant") or "").strip()
    category = str(item.get("category") or "").strip()

    if intent == "income":
        source = str(item.get("source") or item.get("category") or "").strip()
        payer = str(item.get("payer") or "").strip()
        if source and payer:
            return f"Income from {source} via {payer}."
        if source:
            return f"Income from {source}."
        if payer:
            return f"Income received from {payer}."
        return "Income recorded."

    if item_name and merchant:
        return f"Paid for {item_name} at {merchant}."
    if item_name:
        return f"Paid for {item_name}."
    if category and merchant:
        return f"Paid for {category.lower()} at {merchant}."
    if category:
        return f"Paid for {category.lower()}."
    if user_input and user_input.strip():
        return "Transaction recorded from user input."
    return "Expense recorded."


def apply_transaction_hints(user_input: str, data: dict) -> dict:
    if not isinstance(data, dict):
        return data

    inferred_intent = infer_transaction_intent(user_input)
    if inferred_intent and str(data.get("intent")).lower() in {"query", "none", "null"}:
        data["intent"] = inferred_intent
    elif inferred_intent and data.get("intent") is None:
        data["intent"] = inferred_intent

    items_list = data.get("items")
    if not isinstance(items_list, list) or not items_list:
        data["items"] = [empty_item()]
        items_list = data["items"]

    item = items_list[0]
    if not isinstance(item, dict):
        item = empty_item()
        data["items"] = [item]

    # Preserve explicit amount from user text when model misses it.
    if is_empty(item.get("amount")):
        inferred_amount = infer_explicit_amount(user_input)
        if inferred_amount is not None:
            item["amount"] = inferred_amount

    # Keep payment_method empty unless explicitly stated by the user.
    if not has_explicit_payment_method(user_input):
        item["payment_method"] = None
    else:
        # ── FIX 3: Normalize payment_method extracted from user input right away.
        # This ensures "gpay", "net banking" etc. are canonicalized before
        # RAG comparison and before being shown to the user.
        match = re.search(
            r"\b(cash|upi|gpay|phonepe|paytm|card|credit card|debit card|bank transfer|net ?banking|netbanking|neft|imps|wallet|online|cheque|emi)\b",
            user_input.lower(),
        )
        if match:
            item["payment_method"] = normalize_payment_method(match.group(0))
        else:
            raw_pm = item.get("payment_method")
            if raw_pm:
                item["payment_method"] = normalize_payment_method(str(raw_pm))

    if data.get("intent") == "income":
        normalized_source = normalize_income_source_value(
            item.get("source") or item.get("category"), user_input
        )
        if normalized_source:
            item["source"] = normalized_source
            if (
                not item.get("source")
                or str(item.get("source")).strip().lower() in FILLER_VALUES
            ):
                item["source"] = normalized_source
            if (
                not item.get("category")
                or str(item.get("category")).strip().lower() in FILLER_VALUES
            ):
                item["category"] = normalized_source
            if (
                not item.get("item")
                or str(item.get("item")).strip().lower() in FILLER_VALUES
            ):
                item["item"] = normalized_source

        # If model placed a source-like token into payer (e.g. "freelancing"),
        # promote it to source and clear payer.
        current_payer = str(item.get("payer") or "").strip()
        if current_payer:
            source_from_payer = infer_income_source(current_payer)
            if source_from_payer:
                if not normalized_source:
                    item["source"] = source_from_payer
                    if is_empty(item.get("category")):
                        item["category"] = source_from_payer
                    if is_empty(item.get("item")):
                        item["item"] = source_from_payer
                item["payer"] = None

        # ── FIX: Better Payer Extraction for Income
        inferred_payer = infer_income_payer(user_input)
        if inferred_payer:
            source_from_payer_text = infer_income_source(inferred_payer)
            if source_from_payer_text:
                if is_empty(item.get("source")):
                    item["source"] = source_from_payer_text
                if is_empty(item.get("category")):
                    item["category"] = source_from_payer_text
                if is_empty(item.get("item")):
                    item["item"] = source_from_payer_text
                item["payer"] = None
            elif is_empty(item.get("payer")):
                item["payer"] = inferred_payer

    elif data.get("intent") == "expense":
        # Use the intelligent item resolver: prioritize user-text inference, then model output
        best_item = resolve_best_item(user_input, item.get("item"))
        item["item"] = best_item

        item["category"] = normalize_expense_category_value(
            item.get("category"),
            user_input,
            item.get("item"),
        )

        # Backfill category from explicit text/item when model leaves it null.
        if is_empty(item.get("category")):
            inferred_category = infer_expense_category(user_input, item.get("item"))
            if inferred_category:
                item["category"] = inferred_category

        # Backfill merchant from explicit user phrase when model omits it.
        if is_empty(item.get("merchant")):
            inferred_merchant = infer_explicit_merchant(user_input)
            if inferred_merchant:
                item["merchant"] = inferred_merchant

    # FIX 5: Override model category if user explicitly mentioned a known category word
    # Fixes: 'spend on groceries' → model says Food → override to Groceries
    if data.get("intent") == "expense":
        user_lower_cat = (user_input or "").lower()
        for known_cat in KNOWN_CATEGORIES:
            if re.search(rf"\b{re.escape(known_cat)}\b", user_lower_cat):
                item["category"] = known_cat.title()
                break

    # FIX 6: Item keyword → category fallback when model returns null/Other
    # Fixes: 'bought subscription 799' → category=None → RAG gets no category signal
    ITEM_TO_CATEGORY = {
        # Entertainment
        "subscription": "Entertainment",
        "netflix": "Entertainment",
        "spotify": "Entertainment",
        "hotstar": "Entertainment",
        "prime": "Entertainment",
        "disney": "Entertainment",
        "youtube premium": "Entertainment",
        "movie ticket": "Entertainment",
        "movie tickets": "Entertainment",
        "cinema": "Entertainment",
        "pvr": "Entertainment",
        "inox": "Entertainment",
        "gaming": "Entertainment",
        "game": "Entertainment",
        # Transport
        "petrol": "Transport",
        "diesel": "Transport",
        "fuel": "Transport",
        "cab": "Transport",
        "uber": "Transport",
        "ola": "Transport",
        "rapido": "Transport",
        "auto": "Transport",
        "metro": "Transport",
        "bus fare": "Transport",
        "bus pass": "Transport",
        "train ticket": "Transport",
        "flight ticket": "Transport",
        "flight": "Transport",
        "parking": "Transport",
        "toll": "Transport",
        "car service": "Transport",
        "bike repair": "Transport",
        "car wash": "Transport",
        # Health
        "medicine": "Health",
        "medicines": "Health",
        "tablet": "Health",
        "tablets": "Health",
        "doctor": "Health",
        "hospital": "Health",
        "pharmacy": "Health",
        "gym": "Health",
        "supplement": "Health",
        "supplements": "Health",
        "protein": "Health",
        "vitamin": "Health",
        "vitamins": "Health",
        "lab test": "Health",
        "checkup": "Health",
        "dental": "Health",
        "physiotherapy": "Health",
        # Groceries
        "groceries": "Groceries",
        "grocery": "Groceries",
        "vegetables": "Groceries",
        "fruits": "Groceries",
        "sabzi": "Groceries",
        "doodh": "Groceries",
        # Bills
        "electricity": "Bills",
        "electric bill": "Bills",
        "water bill": "Bills",
        "recharge": "Bills",
        "wifi": "Bills",
        "internet bill": "Bills",
        "mobile bill": "Bills",
        "gas bill": "Bills",
    }
    if is_empty(item.get("category")) or str(item.get("category", "")).lower() in {
        "other",
        "none",
        "null",
    }:
        user_lower_item = (user_input or "").lower()
        item_val = str(item.get("item") or "").lower()
        combined = user_lower_item + " " + item_val
        for keyword, category in ITEM_TO_CATEGORY.items():
            if keyword in combined:
                item["category"] = category
                break

    # Auto-generate remarks from transaction context when not explicitly provided.
    raw_remarks = item.get("remarks")
    unsafe_markers = ["not supported", "unsupported", "internal", "error"]
    if has_explicit_remarks(user_input):
        # User explicitly mentioned remarks/notes
        if raw_remarks is not None:
            remarks_text = str(raw_remarks).strip()
            if (
                not remarks_text
                or remarks_text.lower() in FILLER_VALUES
                or any(marker in remarks_text.lower() for marker in unsafe_markers)
            ):
                item["remarks"] = None
            else:
                item["remarks"] = remarks_text
        else:
            item["remarks"] = None
    else:
        # No explicit remarks in user input — auto-generate from context
        item["remarks"] = auto_generate_remarks(user_input, data, item)

    # Extract bill number directly from user input and override hallucinated values.
    try:
        import sys, os

        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
        from time_parser import parse_bill_no

        extracted_bill = parse_bill_no(user_input)
        if extracted_bill:
            item["bill_no"] = extracted_bill
        elif not has_explicit_bill_reference(user_input):
            # No bill marker in raw text: drop model hallucinations like "BER".
            item["bill_no"] = None
    except Exception:
        pass

    # Resolve relative datetime phrases during first extraction (before missing-field checks).
    dt_value = item.get("datetime")
    has_datetime = (
        dt_value is not None and str(dt_value).strip().lower() not in FILLER_VALUES
    )
    if not has_datetime:
        try:
            import sys, os

            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
            from time_parser import parse_datetime, contains_relative_datetime_term

            if contains_relative_datetime_term(user_input):
                item["datetime"] = parse_datetime(user_input)
        except Exception:
            pass

    data["items"] = [item]
    return data


def parse_relative_time(text: str) -> tuple[str, Optional[str], Optional[str]]:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    if "today" in normalized:
        return "today", None, None
    if "yesterday" in normalized:
        start = (datetime.now() - timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        end = start.replace(hour=23, minute=59, second=59, microsecond=999999)
        return "custom", start.isoformat(), end.isoformat()
    if re.search(r"\bthis\s+week\b", normalized):
        return "this_week", None, None
    if re.search(r"\bthis\s+month\b", normalized):
        return "this_month", None, None
    if re.search(r"\blast\s+we+k\b", normalized):
        now = datetime.now()
        start_of_this_week = (now - timedelta(days=now.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        start = start_of_this_week - timedelta(days=7)
        end = start_of_this_week - timedelta(microseconds=1)
        return "custom", start.isoformat(), end.isoformat()
    if "last month" in normalized:
        return "last_month", None, None
    if parsed_date := parse_explicit_date(normalized):
        start_date, end_date = parsed_date
        return "custom", start_date, end_date
    return "all_time", None, None


def parse_relative_datetime_point(text: str) -> Optional[str]:
    """
    Parse relative datetime phrases for transaction timestamp follow-ups.
    Returns a single ISO datetime point (start-of-day) when possible.
    """
    normalized = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    if not normalized:
        return None

    now = datetime.now()

    if "today" in normalized:
        return now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    if "yesterday" in normalized:
        return (
            (now - timedelta(days=1))
            .replace(hour=0, minute=0, second=0, microsecond=0)
            .isoformat()
        )

    if re.search(r"\blast\s+we+k\b", normalized):
        start_of_this_week = (now - timedelta(days=now.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return (start_of_this_week - timedelta(days=7)).isoformat()
    if "last month" in normalized:
        first_day_this_month = now.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        last_month_end = first_day_this_month - timedelta(days=1)
        return last_month_end.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        ).isoformat()

    ago_match = re.search(
        r"\b(\d+|a|one|two|three|four|five|six|seven|eight|nine|ten)\s+(day|week|month|year)s?\s+(ago|back)\b",
        normalized,
    )
    if ago_match:
        val_str = ago_match.group(1)
        word_to_num = {
            "a": 1,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
        }
        value = int(val_str) if val_str.isdigit() else word_to_num[val_str]
        unit = ago_match.group(2)
        if unit == "day":
            target = now - timedelta(days=value)
        elif unit == "week":
            target = now - timedelta(days=7 * value)
        elif unit == "month":
            target = now - timedelta(days=30 * value)
        else:
            target = now - timedelta(days=365 * value)
        return target.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

    parsed = parse_explicit_date(normalized)
    if parsed:
        return parsed[0]
    return None


def is_query_like_message(user_input: str, is_followup: bool = False) -> bool:
    text = user_input.lower().strip()
    always_tx_words = [
        "ordered",
        "order",
        "recharge",
        "bought",
        "paid",
        "spent",
        "received",
        "got",
        "earned",
        "medicine",
        "groceries",
        "food",
        "salary",
        "subscription",
        "purchased",
        "purchase",
        "booked",
    ]
    pattern = r"\b(" + "|".join(re.escape(w) for w in always_tx_words) + r")\b"

    query_keywords = [
        "total",
        "balance",
        "breakdown",
        "how much",
        "average",
        "summary",
        "list",
        "show",
        "give me",
        "what did",
        "what is",
        "what's",
    ]

    has_query_keyword = any(kw in text for kw in query_keywords) or "?" in text
    has_time = parse_relative_time(text)[0] != "all_time" or bool(
        parse_explicit_date(text)
    )
    has_category = bool(extract_category(text))

    # Strong transaction marker: contains an amount and a transaction word
    strong_tx = has_amount(text) and (
        re.search(pattern, text) or looks_like_transaction_statement(text)
    )

    if strong_tx and "?" not in text:
        return False

    if has_query_keyword:
        return True

    # Guard: if input has an amount it's a transaction not a query.
    # Prevents messages like "travel expense 800 today" from being classified as analytics.
    if has_amount(text):
        return False

    if re.fullmatch(r"(on|in|for)\s+.+", text):
        return True

    # If it's NOT a followup, be more generous about what constitutes a query
    if not is_followup:
        # A single word that happens to be a category or a time, with no amount
        if (
            (has_category or has_time)
            and not has_amount(text)
            and len(text.split()) <= 3
        ):
            return True

        # If it doesn't look like a clear transaction and has no amount, but implies category/time
        if not looks_like_transaction_statement(text) and not has_amount(text):
            if has_category or has_time:
                return True

    return False


def should_start_fresh_transaction(user_input: str) -> bool:
    """
    Detect whether a message is a new transaction statement even if a follow-up
    session is still active.

    This prevents a new sentence like "spend 300 on groceries today" from being
    merged into the prior merchant/payment follow-up.
    """
    text = (user_input or "").strip()
    if not text:
        return False

    if looks_like_transaction_statement(text):
        return True

    intent = infer_transaction_intent(text)
    return bool(intent and has_amount(text))


def parse_query_heuristically(user_input: str) -> dict:
    text = user_input.lower()
    category = extract_category(text)
    time_range, start_date, end_date = parse_relative_time(text)
    if "balance" in text:
        query_type = "balance"
    elif "category" in text or "by category" in text or "breakdown" in text:
        query_type = "category_breakdown"
    elif "income" in text:
        query_type = "total_income"
    else:
        query_type = "total_expense"
    return {
        "intent": "query",
        "query_type": query_type,
        "filters": {
            "time_range": time_range,
            "category": category,
            "start_date": start_date,
            "end_date": end_date,
        },
    }


def amount_in_input(user_input: str, amount) -> bool:
    if amount is None:
        return True
    try:
        amount_str = (
            str(int(amount)) if float(amount) == int(float(amount)) else str(amount)
        )
        return bool(re.search(rf"\b{re.escape(amount_str)}\b", user_input))
    except:
        return True


def build_extraction_prompt(user_input: str, context: list = []) -> str:
    context_str = ""
    if context:
        context_str = "\nRelevant past transactions:\n" + "\n".join(
            [f"- {json.dumps(c)}" for c in context[:2]]
        )

    return f"""Extract financial transaction data from the user input.

IMPORTANT RULES:
- Only extract what is EXPLICITLY mentioned. Do NOT guess or assume.
- If something is not clearly stated, use null.
- payment_method must be null unless user explicitly says cash/UPI/card/online etc.
- datetime: if user says "today" use today's date, "yesterday" use yesterday, otherwise null if not mentioned.
- category: guess from context (food, shopping, transport etc.) - this is ok to infer.
- remarks: generate a concise remark based on transaction context when user does not provide one explicitly.
- Income rules:
    - payer = company/person sending money (e.g., "TechCorp" in "received from TechCorp")
    - source = one of Salary/Freelance/Business/Investment/Other when identifiable
    - for income, item should stay null unless user explicitly gives an income item label
    - NEVER set item to "Cash" for income transactions
- CRITICAL: amount must ALWAYS be null unless the user explicitly states a number in their input. Never use amounts from the context/past transactions shown below.

Return ONLY valid JSON:
{{
  "intent": "expense" or "income",
  "items": [{{
    "amount": number or null,
    "category": string or null,
    "currency": "INR",
    "item": short description or null (for income keep null unless explicitly provided; never "Cash"),
    "merchant": string or null,
    "payment_method": null if not mentioned,
        "remarks": concise string summary (auto-generate if not explicitly provided),
    "datetime": ISO datetime or null (today = {datetime.now().strftime('%Y-%m-%d %H:%M')}),
        "bill_no": null if not mentioned
  }}]
}}
{context_str}
Input: "{user_input}"
JSON:"""


def should_use_fast_extraction(user_input: str) -> bool:
    text = (user_input or "").strip()
    if not text:
        return False
    if is_query_like_message(text):
        return False

    strong_signals = [
        has_amount(text),
        bool(infer_transaction_intent(text)),
        looks_like_transaction_statement(text),
        bool(infer_explicit_merchant(text)),
        bool(infer_expense_item(text)),
        bool(infer_income_source(text)),
        bool(infer_income_payer(text)),
    ]
    return any(strong_signals)


def extract_fields(user_input: str, context: list = []) -> dict:
    is_query = is_query_like_message(user_input)
    if is_query:
        return parse_query_heuristically(user_input)

    # Fast path: simple transaction statements already have enough signal for
    # heuristic extraction, so avoid a full GGUF round-trip when possible.
    if should_use_fast_extraction(user_input):
        heuristic = {
            "intent": infer_transaction_intent(user_input),
            "items": [empty_item()],
        }
        return normalize_model_output(heuristic, user_input)

    raw_prompt = build_extraction_prompt(user_input, context)
    chat_prompt = CHATML_TEMPLATE.format(user=raw_prompt.strip())

    try:
        t1 = time.time()
        if llm is None:
            raise RuntimeError("Local LLM not initialized")

        response = llm(
            chat_prompt,
            max_tokens=128,
            temperature=0.0,
            stop=["<|im_end|>"],
            echo=False,
        )
        t2 = time.time()
        print(f"Local LLM response time: {t2 - t1:.2f} seconds")
        raw = response["choices"][0]["text"].strip()
    except Exception as e:
        print(f"Local LLM failed: {e}")
        return normalize_model_output(
            {"intent": infer_transaction_intent(user_input), "items": [empty_item()]},
            user_input,
        )

    repaired = repair_json_string(raw)
    try:
        parsed = json.loads(repaired)
    except Exception as e:
        print(f"Error parsing JSON: {repaired}\nReason: {e}")
        parsed = {}

    return normalize_model_output(parsed, user_input)


def generate_followup(data: dict, missing_field: str) -> str:
    item = data.get("items", [{}])[0] if data.get("items") else {}
    merchant = item.get("merchant")
    source = item.get("source") or item.get("category") or item.get("item")
    intent = str(data.get("intent") or "").strip().lower()

    context_str = (
        f" at {merchant}"
        if merchant and str(merchant).strip().lower() not in FILLER_VALUES
        else ""
    )
    income_context = (
        f" from {source}"
        if source and str(source).strip().lower() not in FILLER_VALUES
        else ""
    )

    if missing_field == "amount":
        if intent == "income":
            return f"How much did you receive{income_context}?"
        return f"How much did you spend{context_str}?"
    if missing_field == "payment_method":
        if intent == "income":
            return f"How did you receive it{income_context}? (e.g., cash, bank transfer, UPI)"
        return f"How did you pay{context_str}? (e.g., cash, UPI, card)"
    if missing_field == "datetime":
        return f"When did this transaction occur{context_str}?"
    if missing_field == "intent":
        return "Is this an expense or an income?"
    if missing_field == "category":
        return "What category should I use? (optional - say 'skip' to continue)"
    if missing_field == "item":
        return "What was the item or short description? (optional - say 'skip' to continue)"
    if missing_field == "merchant":
        return "Which merchant/store was this at? (optional - say 'skip' to continue)"
    if missing_field == "source":
        return "What is the income source? (optional - say 'skip' to continue)"
    if missing_field == "payer":
        return "Who paid you? (optional - say 'skip' to continue)"
    if missing_field == "bill_no":
        return "Do you want to add a bill number? (optional - say 'skip' to continue)"
    if missing_field == "remarks":
        return "Any remarks to save with this transaction? (optional - say 'skip' to continue)"
    return f"Could you please provide the {missing_field}?"


def generate_optional_assumption_followup(field: str, value: str) -> str:
    clean_value = str(value or "").strip()
    if field == "merchant":
        return f"I found a similar transaction. Is the merchant {clean_value}? (yes/no, or type the correct merchant)"
    if field == "payment_method":
        return f"I found a similar transaction. Was payment method {clean_value}? (yes/no, or type the correct method)"
    if field == "category":
        return f"I found a similar transaction. Should I keep category as {clean_value}? (yes/no, or type the correct category)"
    if field == "item":
        return f"I found a similar transaction. Is item {clean_value}? (yes/no, or type the correct item)"
    return f"I found a similar transaction. Is {field} {clean_value}? (yes/no, or type the correct value)"


def build_optional_assumption_suggestions(
    extracted: dict, optional_queue: list[str], user_input: str
) -> dict:
    if not optional_queue:
        return {}
    item = extracted.get("items", [{}])[0] if extracted.get("items") else {}
    if not isinstance(item, dict):
        return {}

    explicit_fields = detect_explicit_fields(user_input, extracted)
    item_text = str(item.get("item") or "")
    category_text = str(item.get("category") or "")
    intent = str(extracted.get("intent") or "").strip().lower()
    allowed_fields = (
        {"merchant", "payment_method"}
        if intent == "expense"
        else {"payment_method", "payer"}
    )

    # Use already-computed RAG results from /chat — don't call retrieve_assumptions again
    rag_assumptions = extracted.get("rag_assumptions") or {}
    matched_entries = extracted.get("retrieved_context") or []

    suggestions = {}

    # Primary: use rag_assumptions directly (already computed with correct query)
    for field in optional_queue:
        if field not in allowed_fields:
            continue
        if field in explicit_fields:
            continue
        if not is_empty(item.get(field)):
            continue
        value = rag_assumptions.get(field)
        if is_empty(value):
            continue
        suggestions[field] = str(value).strip()

    # Fallback: score matched_entries directly
    for field in optional_queue:
        if field not in allowed_fields:
            continue
        if field in suggestions:
            continue
        if field in explicit_fields:
            continue
        if not is_empty(item.get(field)):
            continue

        best_value, best_score = _pick_optional_candidate(
            matched_entries[:10],
            field,
            user_input,
            item_text,
            category_text,
        )

        threshold = 0.44 if field == "merchant" else 0.45
        if best_value and best_score >= threshold:
            suggestions[field] = best_value

    return suggestions


def merge_followup_answer(existing: dict, answer: str, field: str) -> dict:
    """
    Merge a follow-up answer back into the session dict.

    Key fixes vs the original:
    - payment_method: use normalize_payment_method (no Ollama round-trip for a single word).
    - datetime: use parse_relative_time for simple words; only call time_parser if available.
    - amount: handle "rs 300", "₹300", commas, "300 rupees" variants.
    - All other fields: write the stripped answer directly (no extract_fields call).
    """
    if field == "intent":
        ans = answer.strip().lower()
        if "expense" in ans:
            existing["intent"] = "expense"
        elif "income" in ans:
            existing["intent"] = "income"
        else:
            # ── FIX: Allow the follow-up to understand sentences like "got freelance payment"
            inferred = infer_transaction_intent(ans)
            if inferred:
                existing["intent"] = inferred
            else:
                existing["intent"] = "unsupported"
        return existing

    if not existing.get("items"):
        existing["items"] = [empty_item()]
    item = existing["items"][0]
    normalized_answer = str(answer or "").strip().lower()

    optional_fields = OPTIONAL_FOLLOWUP_FIELDS.get(
        str(existing.get("intent") or "").strip().lower(), []
    )
    skipped = existing.get("skipped_optional_fields")
    if not isinstance(skipped, list):
        skipped = []

    optional_suggestions = existing.get("optional_assumption_suggestions")
    if not isinstance(optional_suggestions, dict):
        optional_suggestions = {}

    rejected_suggestion_fields = existing.get("rejected_optional_suggestion_fields")
    if not isinstance(rejected_suggestion_fields, list):
        rejected_suggestion_fields = []

    if field in optional_fields:
        assumed_value = optional_suggestions.get(field)
        if assumed_value and is_affirmative_answer(answer):
            if field == "payment_method":
                item[field] = normalize_payment_method(str(assumed_value))
            else:
                item[field] = str(assumed_value).strip()
            existing["items"] = [item]
            if field in rejected_suggestion_fields:
                rejected_suggestion_fields = [
                    f for f in rejected_suggestion_fields if f != field
                ]
                existing["rejected_optional_suggestion_fields"] = (
                    rejected_suggestion_fields
                )
            return existing

        if assumed_value and is_negative_answer(answer):
            if field not in rejected_suggestion_fields:
                rejected_suggestion_fields.append(field)
            existing["rejected_optional_suggestion_fields"] = rejected_suggestion_fields
            existing["items"] = [item]
            return existing

        if is_skip_answer(answer):
            item[field] = None
            existing["items"] = [item]
            if field not in skipped:
                skipped.append(field)
            existing["skipped_optional_fields"] = skipped
            return existing

        if field in skipped:
            skipped = [f for f in skipped if f != field]
            existing["skipped_optional_fields"] = skipped
        if field in rejected_suggestion_fields:
            rejected_suggestion_fields = [
                f for f in rejected_suggestion_fields if f != field
            ]
            existing["rejected_optional_suggestion_fields"] = rejected_suggestion_fields

    if field == "payment_method":
        # ── FIX 4: Normalize inline instead of calling Ollama for a single word.
        item["payment_method"] = normalize_payment_method(answer.strip())

    elif field == "amount":
        # ── FIX 5: Robust amount parsing — handles "rs 300", "₹ 300", "1,500", "1.5k"
        cleaned = answer.strip().lower()
        cleaned = re.sub(r"[₹$€£]", "", cleaned)
        cleaned = re.sub(r"\b(rs\.?|inr|rupees?)\b", "", cleaned).strip()
        cleaned = cleaned.replace(",", "")
        k_match = re.search(r"([\d.]+)\s*k\b", cleaned)
        if k_match:
            item["amount"] = float(k_match.group(1)) * 1000
        else:
            num_match = re.search(r"[\d.]+", cleaned)
            if num_match:
                item["amount"] = float(num_match.group(0))

    elif field == "datetime":
        # Handle relative follow-up values (last week, 3 months back, etc.)
        parsed_point = parse_relative_datetime_point(answer)
        if parsed_point:
            item["datetime"] = parsed_point
        else:
            try:
                import sys, os

                parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if parent_dir not in sys.path:
                    sys.path.append(parent_dir)
                from time_parser import parse_datetime

                item["datetime"] = parse_datetime(answer.strip())
            except (ImportError, Exception):
                # Last resort: store the raw string — better than None
                item["datetime"] = answer.strip()

    elif field == "source":
        # Keep income source discoverable under both source and category keys.
        item["source"] = answer.strip()
        if is_empty(item.get("category")):
            item["category"] = answer.strip()

    elif field == "payer":
        # Keep income payer discoverable under both payer and merchant keys.
        item["payer"] = answer.strip()
        if is_empty(item.get("merchant")):
            item["merchant"] = answer.strip()

    else:
        item[field] = answer.strip()

    existing["items"] = [item]
    return existing


def is_empty(val) -> bool:
    """Returns True if value is None, null-like, or a model guess placeholder."""
    if val is None:
        return True
    if str(val).strip().lower() in FILLER_VALUES:
        return True
    return False


def get_missing_required(data: dict) -> list:
    intent = str(data.get("intent")).strip().lower()
    if intent in ["unsupported", "query"]:
        return []
    item = data.get("items", [{}])[0] if data.get("items") else {}
    missing = []
    if is_empty(data.get("intent")):
        missing.append("intent")
    if is_empty(item.get("amount")):
        missing.append("amount")
    if is_empty(item.get("datetime")):
        missing.append("datetime")
    return missing


def get_missing_optional(data: dict) -> list:
    item = data.get("items", [{}])[0] if data.get("items") else {}
    intent = str(data.get("intent") or "").strip().lower()
    if intent not in OPTIONAL_FOLLOWUP_FIELDS:
        return []
    skipped = data.get("skipped_optional_fields")
    if not isinstance(skipped, list):
        skipped = []
    missing = []
    for field in OPTIONAL_FOLLOWUP_FIELDS[intent]:
        if field in skipped:
            continue
        if is_empty(item.get(field)):
            missing.append(field)
    return missing


def parse_stored_datetime(value: Optional[str]) -> datetime:
    """
    ── FIX 7: Robust datetime parse for sorting.
    Handles ISO strings, date-only strings, and garbage gracefully.
    """
    if not value:
        return datetime.min
    v = str(value).strip()
    # Try standard ISO parse
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(v[: len(fmt) + 2], fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(v.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return datetime.min


def to_similarity(distance: Optional[float]) -> float:
    """Convert ChromaDB cosine distance to similarity score.

    ChromaDB cosine distance is in [0, 2] where 0 = identical.
    Similarity = 1 - (distance / 2), clamped to [0, 1].
    """
    try:
        d = float(distance)
        return max(0.0, min(1.0, 1.0 - d / 2.0))
    except Exception:
        return 0.0


def _annotate_rag_entry(
    meta: dict,
    similarity: float,
    base_similarity: Optional[float] = None,
    keyword_boosted: bool = False,
) -> dict:
    cleaned = clean_assumed_prefix(meta)
    cleaned["_similarity"] = round(float(similarity), 3)
    cleaned["_base_similarity"] = round(
        float(base_similarity if base_similarity is not None else similarity), 3
    )
    cleaned["_keyword_boosted"] = bool(keyword_boosted)
    return cleaned


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def clean_assumed_prefix(meta: dict) -> dict:
    """
    ── FIX 8: Strip accidental 'assumed' prefix baked into stored values.
    Also strip 'Assumed' (title-case) variant.
    """
    cleaned = {}
    for k, v in meta.items():
        if isinstance(v, str):
            stripped = v.strip()
            lower = stripped.lower()
            if lower.startswith("assumed") and len(stripped) > 7:
                cleaned[k] = stripped[7:].strip()
            else:
                cleaned[k] = stripped
        else:
            cleaned[k] = v
    return cleaned


def _is_real_value(value) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    return text.lower() not in FILLER_VALUES


def _tokenize_match_text(text: Optional[str]) -> set[str]:
    if not text:
        return set()
    cleaned = re.sub(r"[^\w\s]", " ", str(text).lower())
    return {token for token in cleaned.split() if token}


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _score_optional_candidate(
    entry: dict,
    field: str,
    query_text: str,
    item_text: str = "",
    category_text: str = "",
) -> float:
    base = float(entry.get("_similarity") or 0.0)
    query_tokens = _tokenize_match_text(query_text)
    item_tokens = _tokenize_match_text(item_text)
    category_tokens = _tokenize_match_text(category_text)
    focus_tokens = item_tokens | category_tokens
    entry_item_tokens = _tokenize_match_text(str(entry.get("item") or ""))
    entry_category_tokens = _tokenize_match_text(str(entry.get("category") or ""))
    entry_context_tokens = _tokenize_match_text(
        " ".join(
            filter(
                None,
                [
                    str(entry.get("item") or ""),
                    str(entry.get("remarks") or ""),
                    str(entry.get("category") or ""),
                ],
            )
        )
    )

    if field == "merchant":
        candidate_tokens = _tokenize_match_text(
            str(entry.get("merchant") or entry.get("payer") or "")
        )
        score = base
        item_overlap = _jaccard_similarity(item_tokens, entry_item_tokens)
        category_overlap = _jaccard_similarity(category_tokens, entry_category_tokens)
        context_overlap = _jaccard_similarity(focus_tokens, entry_context_tokens)

        score += 0.85 * item_overlap
        score += 0.15 * context_overlap
        score += 0.20 * _jaccard_similarity(query_tokens, candidate_tokens)
        if query_tokens & candidate_tokens:
            score += 0.08
        if item_overlap >= 0.5:
            score += 0.18

        # ── GUARDRAIL: apply cap FIRST before any category boost
        if item_tokens and item_overlap < 0.20:
            score = min(score, 0.46)
        if (
            category_tokens
            and category_overlap > 0
            and item_tokens
            and item_overlap == 0
        ):
            score = min(score, 0.48)

        # ── CATEGORY BOOST: added AFTER cap so it isn't wiped out
        # When same category (e.g. both "health"), this is strong signal
        entry_cat_tokens = _tokenize_match_text(str(entry.get("category") or ""))
        current_cat_tokens = _tokenize_match_text(category_text)
        category_match = _jaccard_similarity(current_cat_tokens, entry_cat_tokens)
        if category_match >= 0.8:
            score += 0.20

        return score

    # payment_method, category, source, payer — unchanged
    if field == "payment_method":
        candidate_tokens = _tokenize_match_text(str(entry.get("payment_method") or ""))
        score = base
        score += 0.25 * _jaccard_similarity(focus_tokens, entry_context_tokens)
        score += 0.45 * _jaccard_similarity(query_tokens, candidate_tokens)
        return score

    if field in {"category", "source", "payer"}:
        candidate_tokens = _tokenize_match_text(str(entry.get(field) or ""))
        score = base
        score += 0.25 * _jaccard_similarity(focus_tokens, entry_context_tokens)
        score += 0.20 * _jaccard_similarity(query_tokens, candidate_tokens)
        return score

    candidate_tokens = _tokenize_match_text(str(entry.get(field) or ""))
    score = base
    score += 0.20 * _jaccard_similarity(focus_tokens, entry_context_tokens)
    score += 0.15 * _jaccard_similarity(query_tokens, candidate_tokens)
    return score


def _pick_optional_candidate(
    entries: list[dict],
    field: str,
    query_text: str,
    item_text: str = "",
    category_text: str = "",
) -> tuple[Optional[str], float]:
    scored_candidates = []
    item_tokens = _tokenize_match_text(item_text)

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        value = entry.get(field)
        if field == "merchant" and is_empty(value):
            value = entry.get("payer")
        if field == "category" and is_empty(value):
            value = entry.get("source")
        if is_empty(value):
            continue

        score = _score_optional_candidate(
            entry, field, query_text, item_text, category_text
        )
        entry_item_tokens = _tokenize_match_text(str(entry.get("item") or ""))
        item_overlap = _jaccard_similarity(item_tokens, entry_item_tokens)
        entry_datetime = parse_stored_datetime(entry.get("datetime"))
        scored_candidates.append(
            (score, str(value).strip(), entry_datetime, item_overlap)
        )

    if not scored_candidates:
        return None, 0.0

    # If we have an explicit item in query, prefer candidates with strongest
    # item overlap (e.g., rice -> dmart over vegetables -> bigbazaar).
    if field == "merchant" and item_tokens:
        max_item_overlap = max(c[3] for c in scored_candidates)
        if max_item_overlap >= 0.20:
            scored_candidates = [
                c for c in scored_candidates if c[3] >= max_item_overlap - 1e-6
            ]

    scored_candidates.sort(key=lambda x: (x[0], x[2]), reverse=True)
    best_score, best_value, _, _ = scored_candidates[0]

    return best_value, best_score


def build_embed_text(entry: dict) -> str:
    item = entry.get("items", [{}])[0] if entry.get("items") else {}
    intent = str(entry.get("intent") or "").strip().lower() or "expense"

    # Plain natural language — no colons, no amount, no prepositions
    # all-MiniLM-L6-v2 scores plain words higher than "category:Health"
    parts = [intent]

    category = item.get("category")
    if _is_real_value(category):
        parts.append(str(category).strip())

    item_name = item.get("item")
    if _is_real_value(item_name):
        parts.append(str(item_name).strip())

    if intent == "income":
        source = item.get("source")
        if _is_real_value(source):
            parts.append(str(source).strip())
        payer = item.get("payer")
        if _is_real_value(payer):
            parts.append(str(payer).strip())
    else:
        merchant = item.get("merchant")
        if _is_real_value(merchant):
            parts.append(str(merchant).strip())

    payment = item.get("payment_method")
    if _is_real_value(payment):
        parts.append(str(payment).strip())

    return " ".join(parts)


def _majority_vote(metas: list[dict], fields: list[str]) -> dict:
    votes: dict[str, str | float] = {}
    for field in fields:
        values = []
        for meta in metas:
            if not isinstance(meta, dict):
                continue
            value = meta.get(field)
            if _is_real_value(value):
                values.append(str(value).strip())
        if not values:
            continue
        counts: dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        best_value, best_count = max(counts.items(), key=lambda kv: kv[1])
        confidence = best_count / len(values)
        if confidence >= 0.4:
            votes[field] = best_value
            votes[f"{field}_confidence"] = round(confidence, 3)
    return votes


def retrieve_assumptions(
    extracted: dict, target_collection=None, intent_override: Optional[str] = None
) -> dict:
    try:
        active_collection = target_collection or ensure_collection()
        model = ensure_embed_model()
        if active_collection is None:
            return {
                "assumptions": {},
                "retrieval_stage": "semantic_fallback",
                "matched_entries": [],
            }

        count = active_collection.count()
        if count == 0:
            return {
                "assumptions": {},
                "retrieval_stage": "semantic_fallback",
                "matched_entries": [],
            }

        item = extracted.get("items", [{}])[0] if extracted.get("items") else {}
        intent = str(intent_override or extracted.get("intent") or "").strip().lower()
        if intent not in {"expense", "income"}:
            return {
                "assumptions": {},
                "retrieval_stage": "semantic_fallback",
                "matched_entries": [],
            }

        query_text = str(extracted.get("original_query") or "")
        print(
            f"DEBUG retrieve_assumptions: query_text='{query_text}' | intent='{intent}' | item='{item.get('item')}' | category='{item.get('category')}' | merchant='{item.get('merchant')}' | collection_count={count}"
        )

        def _norm_category(value: Optional[str]) -> str:
            normalized = normalize_expense_category_value(
                value, query_text, item.get("item")
            )
            return str(normalized or "").strip().lower()

        def _is_non_informative_category(value: Optional[str]) -> bool:
            normalized = str(value or "").strip().lower()
            return normalized in {
                "",
                "other",
                "others",
                "misc",
                "miscellaneous",
                "general",
                "unknown",
                "uncategorized",
                "none",
                "null",
                "na",
                "n/a",
            }

        inferred_query_category = ""
        if intent == "expense":
            inferred_query_category = _norm_category(item.get("category"))
            if _is_non_informative_category(inferred_query_category):
                inferred_query_category = ""
            if not inferred_query_category:
                inferred = infer_expense_category(query_text, item.get("item"))
                inferred_query_category = str(inferred or "").strip().lower()
                if _is_non_informative_category(inferred_query_category):
                    inferred_query_category = ""

            if not inferred_query_category:
                for cat, kws in CATEGORY_KEYWORDS.items():
                    if any(k in query_text.lower() for k in kws):
                        inferred_query_category = cat.lower()
                        break

        if model is None:
            lexical_rows = []
            try:
                all_rows = active_collection.get(include=["metadatas"])
                all_metas = (
                    all_rows.get("metadatas", []) if isinstance(all_rows, dict) else []
                )
                item_text = str(item.get("item") or "")
                category_value = inferred_query_category or str(
                    item.get("category") or ""
                )
                for meta in all_metas:
                    if not isinstance(meta, dict):
                        continue
                    cleaned = _annotate_rag_entry(meta, 0.0, 0.0, False)
                    if str(cleaned.get("intent") or "").strip().lower() != intent:
                        continue
                    if intent == "expense" and inferred_query_category:
                        entry_category = _norm_category(cleaned.get("category"))
                        if entry_category and entry_category != inferred_query_category:
                            continue

                    lexical_score = _score_optional_candidate(
                        cleaned,
                        "merchant" if intent == "expense" else "payment_method",
                        query_text,
                        item_text,
                        category_value,
                    )
                    if intent == "expense":
                        lexical_score = max(
                            lexical_score,
                            _score_optional_candidate(
                                cleaned,
                                "payment_method",
                                query_text,
                                item_text,
                                category_value,
                            ),
                        )

                    if lexical_score >= 0.35:
                        lexical_rows.append(
                            _annotate_rag_entry(
                                meta, lexical_score, lexical_score, True
                            )
                        )

                lexical_rows.sort(
                    key=lambda entry: (
                        float(entry.get("_similarity", 0.0)),
                        parse_stored_datetime(entry.get("datetime")),
                    ),
                    reverse=True,
                )
            except Exception:
                lexical_rows = []

            if lexical_rows:
                if intent == "income":
                    assumptions = _majority_vote(
                        lexical_rows, ["source", "payment_method", "payer"]
                    )
                else:
                    assumptions = {}
                    item_text = str(item.get("item") or "")
                    category_text = inferred_query_category or str(
                        item.get("category") or ""
                    )

                    merchant_value, merchant_score = _pick_optional_candidate(
                        lexical_rows,
                        "merchant",
                        query_text,
                        item_text,
                        category_text,
                    )
                    if merchant_value and merchant_score >= 0.35:
                        assumptions["merchant"] = merchant_value
                        assumptions["merchant_confidence"] = round(
                            min(1.0, merchant_score), 3
                        )

                    payment_value, payment_score = _pick_optional_candidate(
                        lexical_rows,
                        "payment_method",
                        query_text,
                        item_text,
                        category_text,
                    )
                    if payment_value and payment_score >= 0.35:
                        assumptions["payment_method"] = payment_value
                        assumptions["payment_method_confidence"] = round(
                            min(1.0, payment_score), 3
                        )

                return {
                    "assumptions": assumptions,
                    "retrieval_stage": "lexical_fallback",
                    "matched_entries": lexical_rows[:5],
                }

        merchant = (
            item.get("merchant") if _is_real_value(item.get("merchant")) else None
        )
        payer = item.get("payer") if _is_real_value(item.get("payer")) else None

        # Stage 1: merchant/payer anchored exact metadata match with intent filter.
        # Category is intentionally excluded as an anchor because it is too broad
        # and can cause incorrect merchant suggestions.
        anchor_field = None
        anchor_value = None
        if merchant:
            anchor_field = "merchant"
            anchor_value = str(merchant).strip()
        elif payer:
            anchor_field = "payer"
            anchor_value = str(payer).strip()

        if anchor_field and anchor_value:
            # Stage 1 query uses same plain-text format as build_embed_text
            query_parts = [intent]
            if _is_real_value(item.get("category")):
                query_parts.append(str(item.get("category")).strip())
            if _is_real_value(item.get("item")):
                query_parts.append(str(item.get("item")).strip())
            query_parts.append(anchor_value)
            stage1_query = " ".join(query_parts)
            stage1_embedding = model.encode(stage1_query).tolist()
            where_clause = {"$and": [{"intent": intent}, {anchor_field: anchor_value}]}
            try:
                stage1 = active_collection.query(
                    query_embeddings=[stage1_embedding],
                    n_results=min(5, count),
                    where=where_clause,
                    include=["metadatas", "distances"],
                )
            except Exception:
                stage1 = active_collection.query(
                    query_embeddings=[stage1_embedding],
                    n_results=min(8, count),
                    where={"intent": intent},
                    include=["metadatas", "distances"],
                )
            stage1_metas = (
                stage1.get("metadatas", [[]])[0] if stage1.get("metadatas") else []
            )
            stage1_distances = (
                stage1.get("distances", [[]])[0] if stage1.get("distances") else []
            )
            cleaned_stage1 = []
            for idx, meta in enumerate(stage1_metas):
                if not isinstance(meta, dict):
                    continue
                distance = (
                    stage1_distances[idx] if idx < len(stage1_distances) else None
                )
                raw_similarity = to_similarity(distance)
                cleaned = _annotate_rag_entry(
                    meta, raw_similarity, raw_similarity, True
                )
                if str(cleaned.get("intent") or "").strip().lower() != intent:
                    continue
                candidate_anchor = str(cleaned.get(anchor_field) or "").strip().lower()
                if candidate_anchor != anchor_value.lower():
                    continue
                cleaned_stage1.append(cleaned)

            # Case-sensitive metadata filters can miss valid anchor matches.
            # Fallback: intent-only query, then case-insensitive anchor filtering in Python.
            if not cleaned_stage1:
                stage1_fallback = active_collection.query(
                    query_embeddings=[stage1_embedding],
                    n_results=min(12, count),
                    where={"intent": intent},
                    include=["metadatas", "distances"],
                )
                fallback_metas = (
                    stage1_fallback.get("metadatas", [[]])[0]
                    if stage1_fallback.get("metadatas")
                    else []
                )
                fallback_distances = (
                    stage1_fallback.get("distances", [[]])[0]
                    if stage1_fallback.get("distances")
                    else []
                )
                for idx, meta in enumerate(fallback_metas):
                    if not isinstance(meta, dict):
                        continue
                    distance = (
                        fallback_distances[idx]
                        if idx < len(fallback_distances)
                        else None
                    )
                    raw_similarity = to_similarity(distance)
                    cleaned = _annotate_rag_entry(
                        meta, raw_similarity, raw_similarity, True
                    )
                    if str(cleaned.get("intent") or "").strip().lower() != intent:
                        continue
                    candidate_anchor = (
                        str(cleaned.get(anchor_field) or "").strip().lower()
                    )
                    if candidate_anchor != anchor_value.lower():
                        continue
                    cleaned_stage1.append(cleaned)

            if cleaned_stage1:
                if intent == "income":
                    stage1_fields = ["source", "payment_method", "payer"]
                else:
                    stage1_fields = ["merchant", "payment_method"]
                assumptions = _majority_vote(cleaned_stage1, stage1_fields)
                return {
                    "assumptions": assumptions,
                    "retrieval_stage": f"{anchor_field}_anchored",
                    "matched_entries": cleaned_stage1,
                }

        # Stage 2: semantic fallback with intent-only metadata filter and pure similarity.
        parts = [intent]
        if _is_real_value(item.get("category")):
            parts.append(str(item.get("category")).strip())
        if _is_real_value(item.get("item")):
            parts.append(str(item.get("item")).strip())
        if intent == "income":
            if _is_real_value(item.get("source")):
                parts.append(str(item.get("source")).strip())
            if _is_real_value(item.get("payer")):
                parts.append(str(item.get("payer")).strip())
            # Add original query for income so "got salary" finds
            # "received salary" even when source/payer are empty
            if query_text:
                parts.append(query_text)
        else:
            if _is_real_value(item.get("merchant")):
                parts.append(str(item.get("merchant")).strip())
        if _is_real_value(item.get("payment_method")):
            parts.append(str(item.get("payment_method")).strip())

        stage2_query = " ".join(parts)
        stage2_embedding = model.encode(stage2_query).tolist()
        stage2 = active_collection.query(
            query_embeddings=[stage2_embedding],
            n_results=min(20, count),
            where={"intent": intent},
            include=["metadatas", "distances"],
        )
        stage2_metas = (
            stage2.get("metadatas", [[]])[0] if stage2.get("metadatas") else []
        )
        stage2_distances = (
            stage2.get("distances", [[]])[0] if stage2.get("distances") else []
        )
        filtered = []
        query_focus_tokens = _tokenize_match_text(
            " ".join(
                filter(
                    None,
                    [
                        str(item.get("item") or ""),
                        str(item.get("category") or ""),
                        stage2_query,
                    ],
                )
            )
        )
        for idx, meta in enumerate(stage2_metas):
            if not isinstance(meta, dict):
                continue
            distance = stage2_distances[idx] if idx < len(stage2_distances) else None
            similarity = to_similarity(distance)
            cleaned = _annotate_rag_entry(meta, similarity, similarity, False)
            if str(cleaned.get("intent") or "").strip().lower() != intent:
                continue

            if intent == "expense" and inferred_query_category:
                entry_category = _norm_category(cleaned.get("category"))
                if entry_category and entry_category != inferred_query_category:
                    continue

            entry_focus_tokens = _tokenize_match_text(
                " ".join(
                    filter(
                        None,
                        [
                            str(cleaned.get("item") or ""),
                            str(cleaned.get("category") or ""),
                            str(cleaned.get("merchant") or cleaned.get("payer") or ""),
                            str(cleaned.get("payment_method") or ""),
                        ],
                    )
                )
            )
            lexical_overlap = _jaccard_similarity(
                query_focus_tokens, entry_focus_tokens
            )

            # Primary include path: semantic similarity threshold.
            # Secondary include path: strong keyword overlap for short transactional queries.
            if similarity >= SEMANTIC_ONLY_THRESHOLD or (
                similarity >= 0.20 and lexical_overlap >= 0.20
            ):
                filtered.append(cleaned)

        # If strict filtering returns nothing, keep a few intent-matched nearest
        # neighbors so downstream field scoring can apply stronger category/item guards.
        if not filtered:
            relaxed = []
            current_category = inferred_query_category
            for idx, meta in enumerate(stage2_metas):
                if not isinstance(meta, dict):
                    continue
                distance = (
                    stage2_distances[idx] if idx < len(stage2_distances) else None
                )
                similarity = to_similarity(distance)
                if similarity < 0.15:
                    continue
                cleaned = _annotate_rag_entry(meta, similarity, similarity, False)
                if str(cleaned.get("intent") or "").strip().lower() != intent:
                    continue
                if current_category:
                    entry_category = _norm_category(cleaned.get("category"))
                    if entry_category and entry_category != current_category:
                        continue
                relaxed.append(cleaned)
                if len(relaxed) >= 3:
                    break
            filtered = relaxed

        if not filtered:
            try:
                lexical_rows = []
                all_rows = active_collection.get(include=["metadatas"])
                all_metas = (
                    all_rows.get("metadatas", []) if isinstance(all_rows, dict) else []
                )
                for meta in all_metas:
                    if not isinstance(meta, dict):
                        continue
                    cleaned = _annotate_rag_entry(meta, 0.0, 0.0, False)
                    if str(cleaned.get("intent") or "").strip().lower() != intent:
                        continue

                    if intent == "expense" and inferred_query_category:
                        entry_category = _norm_category(cleaned.get("category"))
                        if entry_category and entry_category != inferred_query_category:
                            continue

                    item_text = str(item.get("item") or "")
                    category_value = inferred_query_category or str(
                        item.get("category") or ""
                    )
                    lexical_score = _score_optional_candidate(
                        cleaned,
                        "merchant" if intent == "expense" else "payment_method",
                        query_text,
                        item_text,
                        category_value,
                    )
                    if intent == "expense":
                        lexical_score = max(
                            lexical_score,
                            _score_optional_candidate(
                                cleaned,
                                "payment_method",
                                query_text,
                                item_text,
                                category_value,
                            ),
                        )

                    if lexical_score >= 0.35:
                        lexical_rows.append(
                            _annotate_rag_entry(
                                meta, lexical_score, lexical_score, True
                            )
                        )

                lexical_rows.sort(
                    key=lambda entry: (
                        float(entry.get("_similarity", 0.0)),
                        parse_stored_datetime(entry.get("datetime")),
                    ),
                    reverse=True,
                )
                filtered = lexical_rows[:5]
            except Exception:
                pass

        # Recency boost for assumption voting: among similarity-qualified rows,
        # prefer recent transactions by sorting datetime descending.
        filtered.sort(
            key=lambda entry: parse_stored_datetime(entry.get("datetime")),
            reverse=True,
        )

        if intent == "income":
            stage2_fields = ["source", "payment_method", "payer"]
            assumptions = _majority_vote(filtered, stage2_fields)
        else:
            # Expense suggestions use item/category-aware ranking to avoid recency bias.
            assumptions = {}
            item_text = str(item.get("item") or "")
            category_text = inferred_query_category or str(item.get("category") or "")

            merchant_value, merchant_score = _pick_optional_candidate(
                filtered,
                "merchant",
                query_text,
                item_text,
                category_text,
            )

            # Dynamic threshold: if the user's item text strongly matches a past
            # entry's item (e.g. "Medicine" -> "Medicine"), trust the merchant even
            # at a lower overall score. Without this, the merchant name never appears
            # in the query so jaccard(query, merchant_tokens)=0 and score stalls below 0.52.
            merchant_threshold = 0.52
            if item_text and filtered:
                best_item_overlap = max(
                    _jaccard_similarity(
                        _tokenize_match_text(item_text),
                        _tokenize_match_text(str(e.get("item") or "")),
                    )
                    for e in filtered[:5]
                )
                if best_item_overlap >= 0.5:
                    merchant_threshold = 0.42

            if inferred_query_category and filtered:
                best_cat_match = any(
                    _norm_category(e.get("category")) == inferred_query_category
                    for e in filtered[:5]
                )
                if best_cat_match and merchant_threshold > 0.44:
                    merchant_threshold = 0.44

            print(
                f"DEBUG retrieve_assumptions stage2: merchant_value='{merchant_value}' | merchant_score={merchant_score:.3f} | threshold={merchant_threshold} | passes={merchant_value and merchant_score >= merchant_threshold}"
            )
            if merchant_value and merchant_score >= merchant_threshold:
                assumptions["merchant"] = merchant_value
                assumptions["merchant_confidence"] = round(min(1.0, merchant_score), 3)

            payment_value, payment_score = _pick_optional_candidate(
                filtered,
                "payment_method",
                query_text,
                item_text,
                category_text,
            )
            if payment_value and payment_score >= 0.45:
                assumptions["payment_method"] = payment_value
                assumptions["payment_method_confidence"] = round(
                    min(1.0, payment_score), 3
                )

            # If semantic filtering kept the right intent/category neighbors but
            # still missed the merchant, do one broader lexical rescue across all
            # intent-matched rows. This keeps merchant suggestions available when
            # the vector slice is weak but the historical item/merchant pattern is
            # still clearly present in the DB.
            if intent == "expense" and "merchant" not in assumptions:
                try:
                    rescue_rows = []
                    all_rows = active_collection.get(include=["metadatas"])
                    all_metas = (
                        all_rows.get("metadatas", [])
                        if isinstance(all_rows, dict)
                        else []
                    )
                    for meta in all_metas:
                        if not isinstance(meta, dict):
                            continue
                        cleaned = _annotate_rag_entry(meta, 0.0, 0.0, False)
                        if str(cleaned.get("intent") or "").strip().lower() != intent:
                            continue
                        if inferred_query_category:
                            entry_category = _norm_category(cleaned.get("category"))
                            if (
                                entry_category
                                and entry_category != inferred_query_category
                            ):
                                continue
                        rescue_rows.append(cleaned)

                    merchant_value, merchant_score = _pick_optional_candidate(
                        rescue_rows,
                        "merchant",
                        query_text,
                        item_text,
                        category_text,
                    )
                    if merchant_value and merchant_score >= 0.35:
                        assumptions["merchant"] = merchant_value
                        assumptions["merchant_confidence"] = round(
                            min(1.0, merchant_score), 3
                        )
                except Exception:
                    pass

            print(
                f"DEBUG retrieve_assumptions: filtered_count={len(filtered)} | assumptions={assumptions}"
            )
            for i, entry in enumerate(filtered[:3]):
                print(
                    f"  filtered[{i}]: item='{entry.get('item')}' cat='{entry.get('category')}' merchant='{entry.get('merchant')}' sim={entry.get('_similarity')}"
                )

        return {
            "assumptions": assumptions,
            "retrieval_stage": "semantic_fallback",
            "matched_entries": filtered,
        }

    except Exception:
        traceback.print_exc()
        return {
            "assumptions": {},
            "retrieval_stage": "semantic_fallback",
            "matched_entries": [],
        }


def retrieve_context(user_input: str) -> list:
    # Backward-compatible adapter used by legacy debug paths.
    extracted = extract_fields(user_input, [])
    result = retrieve_assumptions(extracted)
    return result.get("matched_entries", [])


def autofill_from_context(data: dict, past_entries: list) -> tuple:
    if not past_entries or not data.get("items"):
        return data, []
    item = data["items"][0]
    past = past_entries[0]
    filled = []
    for field in ["merchant", "payment_method", "category", "currency"]:
        if not item.get(field) and past.get(field):
            item[field] = past[field]
            filled.append(field)
    data["items"] = [item]
    return data, filled


def _value_is_present(value) -> bool:
    return not is_empty(value)


def detect_explicit_fields(user_input: str, data: dict) -> set[str]:
    """
    ── FIX 9: Detect which fields were EXPLICITLY in the user's original input.

    Critical fixes:
    - merchant check: only flag as explicit when the merchant string is non-None
      AND non-empty AND appears literally in the input text. Prevents str(None)="none"
      from matching accidentally.
    - payment_method: detect directly from input text pattern (not from item value),
      so if RAG previously filled it but user didn't say it, it won't be marked explicit.
    - category: only flag when the category keyword appears in the raw input.
    - bill_no: detect from receipt/invoice/bill patterns in user input.
    """
    explicit = set()
    text = (user_input or "").lower()
    item = data.get("items", [{}])[0] if data.get("items") else {}

    # Amount — if a number was parsed, it was explicit
    if _value_is_present(item.get("amount")):
        explicit.add("amount")

    # Datetime — if a time word or date was in the original text
    if _value_is_present(item.get("datetime")):
        if any(
            w in text
            for w in ["today", "yesterday", "this week", "last week", "this month"]
        ) or re.search(r"\b\d{1,2}[\/\-]\d{1,2}\b", text):
            explicit.add("datetime")

    # Bill number — if bill/receipt pattern is in the user input
    if re.search(
        r"\b(?:bill|invoice|receipt|reciept|recipt|receit|txn|transaction|order|#)\s*(?:no|num|number|#|id)?[:\s]*[A-Za-z0-9\-\/]+",
        text,
        re.IGNORECASE,
    ):
        explicit.add("bill_no")

    # Payment method — check the raw text, not the item value
    # (RAG might have pre-filled item["payment_method"] from a past entry)
    payment_pattern = r"\b(cash|upi|gpay|phonepe|paytm|card|credit card|debit card|bank transfer|net ?banking|netbanking|neft|imps|rtgs|wallet|online|cheque|emi)\b"
    if re.search(payment_pattern, text):
        explicit.add("payment_method")

    # Merchant — only if the actual merchant string appears in the input text
    merchant_val = item.get("merchant")
    if merchant_val and not is_empty(merchant_val):
        merchant_lower = str(merchant_val).strip().lower()
        if len(merchant_lower) >= 3 and merchant_lower in text:
            explicit.add("merchant")

    # Category — only if a known category keyword is literally in the input
    known = [c.lower() for c in KNOWN_CATEGORIES]
    current_category = str(item.get("category") or "").strip().lower()
    if not current_category:
        inferred_category = infer_expense_category(
            str(data.get("original_query") or ""),
            str(item.get("item") or ""),
        )
        if inferred_category:
            current_category = str(inferred_category).strip().lower()
    if current_category and current_category in known and current_category in text:
        explicit.add("category")

    # Source / item — present in parsed output implies it came from the input
    if _value_is_present(item.get("source")):
        source_val = str(item.get("source")).strip().lower()
        if len(source_val) >= 3 and source_val in text:
            explicit.add("source")

    if _value_is_present(item.get("item")):
        item_val = str(item.get("item") or "").strip().lower()
        if len(item_val) >= 3 and item_val in text:
            explicit.add("item")

    # Payer — if identified, check if it was literally in input
    if _value_is_present(item.get("payer")):
        payer_val = str(item.get("payer")).strip().lower()
        if len(payer_val) >= 3 and payer_val in text:
            explicit.add("payer")

    if (
        _value_is_present(item.get("currency"))
        and str(item.get("currency")).lower() in text
    ):
        explicit.add("currency")

    return explicit


def score_entry_for_field(
    field: str, query_text: str, entry: dict, item: dict
) -> float:
    """
    Score a past entry for how well it matches the current transaction
    for a specific field suggestion.
    Returns 0.0 to 1.0. Higher = better match.
    """
    base_sim = float(entry.get("_similarity") or 0.0)
    score = base_sim

    query_lower = query_text.lower()
    entry_category = str(entry.get("category") or "").strip().lower()
    current_category = str(item.get("category") or "").strip().lower()
    entry_item = str(entry.get("item") or "").strip().lower()
    entry_merchant = str(entry.get("merchant") or "").strip().lower()
    current_item = str(item.get("item") or "").strip().lower()

    merchant_mentioned_in_query = False
    has_query_item_overlap = False
    has_current_item_overlap = False
    merchant_in_current_item = False

    # BOOST 1: Same category -> strong boost
    if entry_category and current_category and entry_category == current_category:
        score = max(score, 0.60)

    # BOOST 2: Merchant name literally in query
    if entry_merchant and len(entry_merchant) >= 3 and entry_merchant in query_lower:
        score = max(score, 0.95)
        merchant_mentioned_in_query = True

    # BOOST 3: Item keyword overlap between query and past entry
    # Split entry item into words, check how many appear in query
    if entry_item:
        entry_words = [w for w in entry_item.split() if len(w) >= 4]
        matches = sum(1 for w in entry_words if w in query_lower)
        if entry_words:
            overlap = matches / len(entry_words)
            if overlap >= 0.5:
                score = max(score, 0.70 + overlap * 0.20)
                has_query_item_overlap = True

    # BOOST 4: Current item keyword overlap with past entry item
    if current_item and entry_item:
        curr_words = [w for w in current_item.split() if len(w) >= 4]
        matches = sum(1 for w in curr_words if w in entry_item)
        if curr_words:
            overlap = matches / len(curr_words)
            if overlap >= 0.5:
                score = max(score, 0.65 + overlap * 0.20)
                has_current_item_overlap = True

    # BOOST 5: Merchant name appears in current item description
    current_item_lower = str(item.get("item") or "").strip().lower()
    if (
        entry_merchant
        and len(entry_merchant) >= 4
        and entry_merchant in current_item_lower
    ):
        score = max(score, 0.92)
        merchant_in_current_item = True

    # Guardrail for merchant suggestions:
    # category-only matches are too noisy (e.g., "service centre" for unrelated transport items).
    if field == "merchant":
        has_direct_evidence = (
            merchant_mentioned_in_query
            or merchant_in_current_item
            or has_query_item_overlap
            or has_current_item_overlap
        )
        if not has_direct_evidence:
            # Cap BELOW the category boost (0.60) so category-only matches
            # never produce merchant suggestions without item evidence.
            # Previously 0.58 was overridden by the 0.60 category boost.
            score = min(score, 0.54)

    # PENALTY: Different category — only penalize when BOTH are known and different
    # If current_category is empty (model returned null), don't penalize
    if (
        entry_category
        and current_category  # ← both must be non-empty
        and len(current_category) > 1  # ← not a placeholder
        and entry_category != current_category
    ):
        score = min(score, 0.45)

    return min(score, 1.0)


def build_assumptions(data: dict, past_entries: list, explicit_fields: set) -> tuple:
    if not data.get("items"):
        return {}, {}, []

    item = data["items"][0]
    intent = str(data.get("intent") or "").strip().lower()
    if intent not in {"expense", "income"}:
        return {}, {}, []

    if not past_entries:
        return {}, {}, []

    # Get original query for item-level matching
    original_query = str(data.get("original_query") or "").lower()
    current_category = str(item.get("category") or "").strip().lower()

    # Filter by intent first
    intent_matched = [
        e
        for e in past_entries
        if isinstance(e, dict) and str(e.get("intent") or "").strip().lower() == intent
    ]

    if not intent_matched:
        return {}, {}, []

    if intent == "expense":
        assumable_fields = ["merchant", "payment_method"]
    else:
        assumable_fields = ["source", "payment_method", "payer"]

    VAGUE_ITEMS = {
        "something",
        "item",
        "stuff",
        "things",
        "purchase",
        "expense",
        "payment",
    }
    current_item_clean = str(item.get("item") or "").strip().lower()
    is_vague = current_item_clean in VAGUE_ITEMS or len(current_item_clean) <= 3
    if is_vague:
        MERCHANT_THRESHOLD_OVERRIDE = 0.80
    else:
        MERCHANT_THRESHOLD_OVERRIDE = None

    assumptions = {}
    candidates = {}
    assumed_fields = []

    for field in assumable_fields:
        # Skip if user already provided this field explicitly
        if field in explicit_fields and field not in {"merchant", "payer"}:
            continue
        # Skip if field already has a value
        if _value_is_present(item.get(field)):
            continue

        # Score every past entry for this field
        scored = []
        for entry in intent_matched:
            field_value = entry.get(field)
            entry_category = str(entry.get("category") or "").strip().lower()

            # For payer fallback
            if field == "payer" and is_empty(field_value):
                field_value = entry.get("merchant")

            if is_empty(field_value):
                continue

            # Strong category guard for expense assumptions to prevent cross-category leak.
            if intent == "expense" and current_category and entry_category:
                if entry_category.lower() != current_category.lower() and field in {
                    "merchant",
                    "payment_method",
                }:
                    continue

            score = score_entry_for_field(field, original_query, entry, item)
            scored.append((score, str(field_value).strip(), entry))

        if not scored:
            continue

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Only suggest if best score meets minimum threshold
        best_score, best_value, best_entry = scored[0]

        # Threshold depends on field:
        # merchant: needs higher confidence (wrong merchant is annoying)
        # payment_method: can be lower (easier to correct)
        if field == "merchant":
            min_threshold = MERCHANT_THRESHOLD_OVERRIDE or 0.60
        elif field == "payment_method":
            min_threshold = 0.55
        else:
            min_threshold = 0.50

        if best_score < min_threshold:
            continue

        # Build candidate list (top 3 unique values)
        seen = set()
        options = []
        for score, value, entry in scored:
            if value not in seen and score >= min_threshold * 0.85:
                seen.add(value)
                options.append(value)
            if len(options) >= 3:
                break

        if not options:
            continue

        assumptions[field] = options[0]
        candidates[field] = options
        assumed_fields.append(field)

    return assumptions, candidates, assumed_fields


def apply_assumption_action(data: dict, req: ChatRequest) -> dict:
    """
    ── FIX 13: Handle confirm / edit / reject cleanly.

    Reject: clear the pending fields from the item (set to None) but do NOT
    add them to the follow-up queue — they are optional. The transaction is
    complete with only mandatory fields.

    Edit: apply only the edited fields; keep the rest from pending.

    Confirm: apply all pending fields as-is (or swapped via selected_assumption_options).
    """
    if not data.get("items"):
        data["items"] = [empty_item()]

    item = data["items"][0]
    pending = data.get("pending_assumptions") or {}
    selected = req.selected_assumption_options or {}
    edited = req.edited_assumptions or {}

    action = (req.assumption_action or "").strip().lower()

    if action == "confirm":
        final_values = dict(pending)
        for key, value in selected.items():
            if key in final_values and not is_empty(value):
                final_values[key] = value
        for key, value in final_values.items():
            if not is_empty(value):
                # ── Normalize payment_method even when confirmed from RAG
                if key == "payment_method":
                    item[key] = normalize_payment_method(str(value))
                else:
                    item[key] = value
        data["autofilled_fields"] = list(final_values.keys())

    elif action == "edit":
        final_values = dict(pending)
        for key, value in edited.items():
            if key in final_values and not is_empty(value):
                final_values[key] = value
        for key, value in final_values.items():
            if not is_empty(value):
                if key == "payment_method":
                    item[key] = normalize_payment_method(str(value))
                else:
                    item[key] = value
        data["autofilled_fields"] = list(final_values.keys())

    elif action == "reject":
        # ── FIX 13: On reject, clear the assumed values from the item.
        # Do NOT add them to missing/required — they are optional.
        # The transaction should be saved with only mandatory fields.
        for field in pending.keys():
            item[field] = None
        data["autofilled_fields"] = []

    data["items"] = [item]
    data.pop("pending_assumptions", None)
    data.pop("assumption_candidates", None)
    return data


def _merge_extracted_with_priority(
    extracted: dict, assumptions: dict, explicit_fields: set[str]
) -> dict:
    """
    Merge precedence:
    1) Explicit user fields (never overwritten)
    2) RAG assumptions
    3) LLM extracted values (fallback)
    """
    if not extracted.get("items"):
        extracted["items"] = [empty_item()]
    item = extracted["items"][0]

    blocked = {
        "amount",
        "date",
        "datetime",
        "remarks",
        "category",
        "item",
        "source",
        "currency",
    }
    for key, value in assumptions.items():
        if key.endswith("_confidence"):
            continue
        if key in blocked:
            continue
        if key in explicit_fields:
            continue
        if is_empty(value):
            continue
        if key == "payment_method":
            item[key] = normalize_payment_method(str(value))
        else:
            item[key] = value

    extracted["items"] = [item]
    return extracted


def prepare_chat_outcome(extracted: dict, user_input: str, past_entries: list) -> dict:
    missing = get_missing_required(extracted)
    if missing:
        return {
            "status": "followup",
            "question": generate_followup(extracted, missing[0]),
            "followup_field": missing[0],
            "is_optional_followup": False,
            "extracted": extracted,
            "autofilled_fields": extracted.get("autofilled_fields", []),
        }

    optional_done = bool(extracted.get("optional_followups_done"))
    optional_queue = get_missing_optional(extracted)

    if not optional_done:
        if optional_queue:
            next_optional = optional_queue[0]

            if next_optional == "bill_no":
                # Bill numbers are unique — never suggest from past entries
                suggestions = extracted.get("optional_assumption_suggestions", {})
                if "bill_no" in suggestions:
                    del suggestions["bill_no"]
                extracted["optional_assumption_suggestions"] = suggestions
                return {
                    "status": "followup",
                    "question": "Do you have a bill number? (optional - say 'skip' to continue)",
                    "followup_field": "bill_no",
                    "assumption_value": None,
                    "assumption_confidence": None,
                    "retrieval_stage": extracted.get(
                        "retrieval_stage", "semantic_fallback"
                    ),
                    "followup_options": ["Skip"],
                    "is_optional_followup": True,
                    "extracted": extracted,
                    "autofilled_fields": extracted.get("autofilled_fields", []),
                }

            item_data = (
                extracted.get("items", [{}])[0] if extracted.get("items") else {}
            )
            item_text = str(item_data.get("item") or "")
            raw_category = item_data.get("category")
            # Normalize null-like category payloads so keyword inference can fill/override.
            category_text = (
                ""
                if (
                    raw_category is None
                    or str(raw_category).strip().lower()
                    in {"null", "none", "", "other", "unspecified"}
                )
                else str(raw_category).strip()
            )
            original_query = str(extracted.get("original_query") or user_input or "")

            # Always run keyword check — override wrong extractions too
            # (e.g. vegetables -> Food should be overridden to Groceries)
            query_lower = original_query.lower()
            inferred_category = None
            for cat, keywords in CATEGORY_KEYWORDS.items():
                if any(kw in query_lower for kw in keywords):
                    inferred_category = cat
                    break

            # Use inferred category for field_query if:
            # 1. No category extracted at all
            # 2. Extracted category is "Other"
            # 3. Inferred category differs from extracted (override wrong extraction)
            if inferred_category:
                if (
                    not category_text
                    or category_text.lower() in {"other", ""}
                    or inferred_category.lower() != category_text.lower()
                ):
                    category_text = inferred_category

            field_query = " ".join(
                filter(
                    None,
                    [
                        original_query,
                        item_text,
                        category_text,
                    ],
                )
            ).strip()
            print(
                f"DEBUG field_query: '{field_query}' | category_text: '{category_text}' | item_text: '{item_text}'"
            )

            # Use the matched entries already produced by retrieve_assumptions.
            context_entries = (
                extracted.get("retrieved_context")
                if isinstance(extracted.get("retrieved_context"), list)
                else []
            )
            if not context_entries:
                context_entries = past_entries if isinstance(past_entries, list) else []

            intent = str(extracted.get("intent") or "").strip().lower()
            candidate_pool = [
                p
                for p in context_entries
                if isinstance(p, dict)
                and str(p.get("intent") or "").strip().lower() == intent
                and float(p.get("_similarity", 0.0)) >= 0.45
            ]

            # Build optional suggestions. If suggestions are empty, seed them via
            # build_assumptions so score_entry_for_field() ranking is applied.
            suggestions = extracted.get("optional_assumption_suggestions")
            if not isinstance(suggestions, dict):
                suggestions = {}

            if not suggestions:
                suggestions.update(
                    build_optional_assumption_suggestions(
                        extracted, optional_queue, user_input
                    )
                )

            if not suggestions:
                explicit_fields = detect_explicit_fields(user_input, extracted)
                assumptions, candidates, _ = build_assumptions(
                    extracted,
                    past_entries if isinstance(past_entries, list) else [],
                    explicit_fields,
                )

                seeded = {}
                for field in optional_queue:
                    if field in explicit_fields:
                        continue
                    if not is_empty(item_data.get(field)):
                        continue
                    value = assumptions.get(field)
                    if is_empty(value):
                        continue
                    cleaned = clean_assumed_prefix({field: value}).get(field, "")
                    if cleaned and cleaned.lower() not in FILLER_VALUES:
                        seeded[field] = cleaned

                suggestions.update(seeded)
                if isinstance(candidates, dict) and candidates:
                    extracted["optional_assumption_candidates"] = candidates

            if next_optional not in suggestions:
                explicit_fields = detect_explicit_fields(user_input, extracted)
                assumptions, candidates, _ = build_assumptions(
                    extracted,
                    past_entries if isinstance(past_entries, list) else [],
                    explicit_fields,
                )

                if next_optional not in explicit_fields and is_empty(
                    item_data.get(next_optional)
                ):
                    value = assumptions.get(next_optional)
                    if not is_empty(value):
                        cleaned = clean_assumed_prefix({next_optional: value}).get(
                            next_optional, ""
                        )
                        if cleaned and cleaned.lower() not in FILLER_VALUES:
                            suggestions[next_optional] = cleaned
                            if isinstance(candidates, dict) and candidates:
                                extracted["optional_assumption_candidates"] = candidates

            # If scorer-based assumptions don't produce a value for this field,
            # fallback to best semantic candidate from existing matched context.
            if next_optional not in suggestions:

                def get_raw_top(field: str) -> Optional[str]:
                    ranked_pool = candidate_pool
                    if field == "merchant" and not ranked_pool:
                        ranked_pool = context_entries
                    elif field == "merchant" and len(ranked_pool) < 3:
                        ranked_pool = context_entries or ranked_pool
                    if not ranked_pool:
                        return None

                    best_value, best_score = _pick_optional_candidate(
                        ranked_pool,
                        field,
                        field_query,
                        item_text,
                        category_text,
                    )

                    merchant_threshold = 0.35 if field == "merchant" else 0.45
                    if best_value and best_score >= merchant_threshold:
                        cleaned = clean_assumed_prefix({field: best_value}).get(
                            field, ""
                        )
                        if cleaned and cleaned.lower() not in FILLER_VALUES:
                            return cleaned
                    return None

                semantic_best = get_raw_top(next_optional)
                if semantic_best:
                    suggestions[next_optional] = semantic_best

            extracted["optional_assumption_suggestions"] = suggestions

            rejected_suggestion_fields = extracted.get(
                "rejected_optional_suggestion_fields"
            )
            if not isinstance(rejected_suggestion_fields, list):
                rejected_suggestion_fields = []

            if (
                next_optional in suggestions
                and next_optional not in rejected_suggestion_fields
            ):
                assumed_value = str(suggestions[next_optional]).strip()
                rag_assumptions = (
                    extracted.get("rag_assumptions")
                    if isinstance(extracted.get("rag_assumptions"), dict)
                    else {}
                )
                assumption_confidence = rag_assumptions.get(
                    f"{next_optional}_confidence"
                )
                return {
                    "status": "followup",
                    "question": generate_optional_assumption_followup(
                        next_optional, assumed_value
                    ),
                    "followup_field": next_optional,
                    "assumption_value": assumed_value,
                    "assumption_confidence": (
                        float(assumption_confidence)
                        if assumption_confidence is not None
                        else None
                    ),
                    "retrieval_stage": extracted.get(
                        "retrieval_stage", "semantic_fallback"
                    ),
                    "followup_options": ["Yes", "No", "Skip"],
                    "is_optional_followup": True,
                    "extracted": extracted,
                    "autofilled_fields": extracted.get("autofilled_fields", []),
                }

            return {
                "status": "followup",
                "question": generate_followup(extracted, next_optional),
                "followup_field": next_optional,
                "assumption_value": None,
                "assumption_confidence": None,
                "retrieval_stage": extracted.get(
                    "retrieval_stage", "semantic_fallback"
                ),
                "followup_options": ["Skip"],
                "is_optional_followup": True,
                "extracted": extracted,
                "autofilled_fields": extracted.get("autofilled_fields", []),
            }
        extracted["optional_followups_done"] = True
        extracted["autofilled_fields"] = []
        return {
            "status": "complete",
            "extracted": extracted,
            "autofilled_fields": [],
        }

    # Optional follow-up phase already finished in this session.
    extracted["autofilled_fields"] = []
    return {
        "status": "complete",
        "extracted": extracted,
        "autofilled_fields": [],
    }


def _prepare_and_record(extracted: dict, user_input: str, past_entries: list) -> dict:
    """Wrapper that records the last follow-up for the ephemeral fallback."""
    global last_followup_session, last_followup_time, last_followup_field
    outcome = prepare_chat_outcome(extracted, user_input, past_entries)
    try:
        if isinstance(outcome, dict) and outcome.get("status") == "followup":
            last_followup_session = outcome.get("extracted")
            last_followup_field = outcome.get("followup_field")
            last_followup_time = datetime.now()
        else:
            last_followup_session = None
            last_followup_field = None
            last_followup_time = None
    except Exception:
        last_followup_session = None
        last_followup_field = None
        last_followup_time = None
    return outcome


def save_to_db(entry: dict):
    """
    ── FIX 14: Never write "assumed*" prefixes into the DB.
    Normalize payment_method before storage so retrieval is always clean.
    """
    item = entry.get("items", [{}])[0] if entry.get("items") else {}

    # Strip any "assumed" prefix that might have snuck through
    cleaned_item = clean_assumed_prefix(item)

    # Normalize payment method before storage
    raw_pm = cleaned_item.get("payment_method")
    if raw_pm and not is_empty(raw_pm):
        cleaned_item["payment_method"] = normalize_payment_method(str(raw_pm))

    safe_entry = dict(entry)
    safe_item = dict(cleaned_item)
    safe_entry["items"] = [safe_item]
    text = build_embed_text(safe_entry)
    active_collection = ensure_collection()
    model = ensure_embed_model()
    embedding = model.encode(text).tolist() if model is not None else None

    # Only store fields with real values — skip None/empty to avoid
    # false-positive where-clause matches and noisy debug output.
    raw_fields = {
        "intent": entry.get("intent"),
        "amount": cleaned_item.get("amount"),
        "category": cleaned_item.get("category"),
        "source": cleaned_item.get("source"),
        "currency": cleaned_item.get("currency", "INR"),
        "item": cleaned_item.get("item"),
        "merchant": cleaned_item.get("merchant"),
        "payer": cleaned_item.get("payer"),
        "payment_method": cleaned_item.get("payment_method"),
        "datetime": cleaned_item.get("datetime"),
        "bill_no": cleaned_item.get("bill_no"),
        "remarks": cleaned_item.get("remarks"),
    }
    flat = {
        k: str(v)
        for k, v in raw_fields.items()
        if v is not None
        and str(v).strip() != ""
        and str(v).strip().lower() not in FILLER_VALUES
    }

    tx_id = str(uuid.uuid4())
    if embedding is not None:
        active_collection.add(
            documents=[text], metadatas=[flat], embeddings=[embedding], ids=[tx_id]
        )
    else:
        active_collection.add(documents=[text], metadatas=[flat], ids=[tx_id])
    return tx_id


@app.post("/chat")
async def chat(req: ChatRequest):
    if req.session_data and req.assumption_action in {"confirm", "edit", "reject"}:
        updated = apply_assumption_action(dict(req.session_data), req)
        return {
            "status": "complete",
            "extracted": updated,
            "autofilled_fields": updated.get("autofilled_fields", []),
        }

    # Robustness: frontend may sometimes omit followup_field when sending a
    # short follow-up answer like "cash" or "upi". If session_data is present
    # and looks like it's waiting on payment_method, accept single-word answers
    # that normalize to a known payment method and treat them as follow-up.
    # Fallback: if client omits session_data but user replies with a single
    # payment-method token soon after the server asked for it, merge into the
    # last recorded followup (best-effort, single-user/dev only).
    if not req.session_data and not req.followup_field:
        try:
            global last_followup_session, last_followup_time, last_followup_field
            if last_followup_session and last_followup_field == "payment_method":
                # 2-minute expiry for safety
                if (
                    last_followup_time
                    and (datetime.now() - last_followup_time).total_seconds() <= 120
                ):
                    pm_norm = normalize_payment_method(str(req.message or ""))
                    if pm_norm:
                        merged = merge_followup_answer(
                            dict(last_followup_session),
                            str(req.message or ""),
                            "payment_method",
                        )
                        orig_q = (
                            last_followup_session.get("original_query")
                            if isinstance(
                                last_followup_session.get("original_query"), str
                            )
                            else req.message
                        )
                        past = merged.get("retrieved_context") or []
                        # clear recorded followup after applying
                        last_followup_session = None
                        last_followup_field = None
                        last_followup_time = None
                        return _prepare_and_record(merged, orig_q, past)
        except Exception:
            pass

    if req.session_data and not req.followup_field:
        try:
            sess = (
                req.session_data
                if isinstance(req.session_data, dict)
                else dict(req.session_data)
            )
            if (
                isinstance(sess, dict)
                and isinstance(sess.get("items"), list)
                and len(sess.get("items") or []) > 0
            ):
                intent = str(sess.get("intent") or "expense").strip().lower()
                pending_optional = OPTIONAL_FOLLOWUP_FIELDS.get(intent, [])
                if "payment_method" in pending_optional:
                    cur_pm = (sess.get("items") or [{}])[0].get("payment_method")
                    if is_empty(cur_pm):
                        pm_norm = normalize_payment_method(str(req.message or ""))
                        if pm_norm:
                            try:
                                # perform merge immediately to avoid relying on client followup_field
                                session_copy = dict(sess)
                                merged = merge_followup_answer(
                                    session_copy,
                                    str(req.message or ""),
                                    "payment_method",
                                )
                                original_query = (
                                    session_copy.get("original_query")
                                    if isinstance(
                                        session_copy.get("original_query"), str
                                    )
                                    else req.message
                                )
                                past = merged.get("retrieved_context") or []
                                return _prepare_and_record(merged, original_query, past)
                            except Exception:
                                pass
        except Exception:
            pass

    # Merging a follow-up answer into existing session
    valid_followup_fields = set(REQUIRED_FIELDS)
    for optional_fields in OPTIONAL_FOLLOWUP_FIELDS.values():
        valid_followup_fields.update(optional_fields)

    has_valid_session_shape = (
        isinstance(req.session_data, dict)
        and isinstance(req.session_data.get("items"), list)
        and len(req.session_data.get("items") or []) > 0
        and isinstance((req.session_data.get("items") or [None])[0], dict)
    )

    if (
        has_valid_session_shape
        and req.followup_field
        and req.followup_field in valid_followup_fields
    ):
        if should_start_fresh_transaction(req.message):
            # Treat the input as a brand-new transaction instead of a follow-up
            # answer. This lets a new natural-language expense reach retrieval
            # even if the previous draft was still waiting on an optional field.
            has_valid_session_shape = False
        elif not is_query_like_message(req.message, is_followup=True):
            session = dict(req.session_data)
            field = req.followup_field
            intent = str(session.get("intent") or "").strip().lower()
            optional_field_set = {
                optional
                for fields in OPTIONAL_FOLLOWUP_FIELDS.values()
                for optional in fields
            }
            if field in optional_field_set:
                allowed_optional_fields = OPTIONAL_FOLLOWUP_FIELDS.get(intent, [])
                if field not in allowed_optional_fields:
                    past = (
                        session.get("retrieved_context")
                        if isinstance(session.get("retrieved_context"), list)
                        else []
                    )
                    original_query = (
                        req.session_data.get("original_query", "")
                        if isinstance(req.session_data, dict)
                        else ""
                    )
                    return _prepare_and_record(session, original_query, past)

            answer = req.message.strip()
            try:
                print(
                    f"DEBUG merging: field={field} answer={answer} session_before={session.get('items')}"
                )
            except Exception:
                pass
            answer_lower = answer.lower()

            # Also support the frontend chip 'rag_accepted' flag
            is_rag_accepted = getattr(req, "rag_accepted", False)

            # Determine the active suggestion based on existing state
            suggestions = session.get("optional_assumption_suggestions", {})
            rag_suggestion = suggestions.get(field)
            has_active_suggestion = not is_empty(rag_suggestion)

            # "no"/"n"/"nope" means SKIP when no suggestion is showing.
            # "no"/"n"/"nope" means REJECT when a suggestion is showing.
            is_skip = is_skip_answer(answer) or (
                is_negative_answer(answer) and not has_active_suggestion
            )

            # Detect yes — accept the RAG suggestion
            is_yes = is_affirmative_answer(answer) or answer_lower in {"sure", "✓"}

            # Detect no — reject suggestion, ask for correct value as plain followup
            is_no = (
                is_negative_answer(answer) or answer_lower in {"not", "x", "✗"}
            ) and has_active_suggestion

            # "no cash" on a payment-method suggestion should be treated as
            # a correction value, not a rejection-only response.
            if (
                has_active_suggestion
                and field == "payment_method"
                and re.match(r"^no\s+", answer_lower)
            ):
                candidate_method = re.sub(r"^no\s+", "", answer_lower).strip()
                if normalize_payment_method(candidate_method):
                    answer = candidate_method
                    answer_lower = candidate_method.lower()
                    is_no = False
                    is_skip = False

            if is_skip:
                # Revert manual bypass of merge_followup_answer for skip, let it process properly
                # so the field gets marked in skipped_optional_fields.
                session = merge_followup_answer(session, answer, field)
                try:
                    print(f"DEBUG merged skip session_after={session.get('items')}")
                except Exception:
                    pass

            elif (is_yes or is_rag_accepted) and not is_empty(rag_suggestion):
                # Accept RAG suggestion
                session = merge_followup_answer(session, str(rag_suggestion), field)
                try:
                    print(f"DEBUG merged yes session_after={session.get('items')}")
                except Exception:
                    pass

            elif is_no and not is_empty(rag_suggestion):
                # User rejected suggestion — ask plain followup for this field

                # Clear suggestion for this field so it isn't asked again
                rejected = session.get("rejected_optional_suggestion_fields")
                if not isinstance(rejected, list):
                    rejected = []
                if field not in rejected:
                    rejected.append(field)
                session["rejected_optional_suggestion_fields"] = rejected

                # Clear _cosine / _similarity on all entries so suggestion is not shown again
                retrieved = session.get("retrieved_context") or []
                for e in retrieved:
                    e["_cosine"] = None
                    e["_similarity"] = 0.0  # force below threshold
                session["retrieved_context"] = retrieved

                # Return plain followup question for same field with no suggestion
                plain_question = generate_followup(session, field)
                return {
                    "status": "followup",
                    "question": plain_question,
                    "followup_field": field,
                    "assumption_value": None,
                    "assumption_confidence": None,
                    "retrieval_stage": session.get(
                        "retrieval_stage", "semantic_fallback"
                    ),
                    "followup_options": ["Skip"],
                    "is_optional_followup": True,
                    "extracted": session,
                    "autofilled_fields": session.get("autofilled_fields", []),
                }

        else:
            # User typed a real value
            if field == "payment_method":
                answer = normalize_payment_method(answer) or answer
            session = merge_followup_answer(session, answer, field)
            try:
                print(f"DEBUG merged real session_after={session.get('items')}")
            except Exception:
                pass

        # Always preserve original retrieved_context — never overwrite with empty
        existing_context = session.get("retrieved_context")
        if isinstance(existing_context, list) and existing_context:
            past = existing_context
        else:
            past = []
        session["retrieved_context"] = past
        original_query = (
            req.session_data.get("original_query", "")
            if isinstance(req.session_data, dict)
            else ""
        )
        return _prepare_and_record(session, original_query, past)

    # Fresh input: extract -> RAG -> check
    extracted = extract_fields(req.message, [])

    # ── Safety net: correct obviously wrong categories before RAG lookup
    # This prevents Groceries category on medicine/tablet queries from
    # poisoning the RAG category guard and blocking Health suggestions
    item_for_correction = (
        extracted.get("items", [{}])[0] if extracted.get("items") else {}
    )
    current_cat = str(item_for_correction.get("category") or "").strip().lower()
    query_lower_correction = req.message.lower()

    STRONG_CATEGORY_OVERRIDES = {
        "health": [
            "tablet",
            "tablets",
            "medicine",
            "medicines",
            "supplement",
            "supplements",
            "vitamin",
            "vitamins",
            "gym",
            "doctor",
            "hospital",
            "pharmacy",
            "clinic",
            "pills",
            "capsule",
        ],
        "transport": [
            "petrol",
            "diesel",
            "fuel",
            "cab",
            "uber",
            "ola",
            "rapido",
            "metro",
            "bus",
            "train",
            "flight",
            "toll",
            "parking",
            "auto rickshaw",
        ],
        "entertainment": [
            "netflix",
            "spotify",
            "hotstar",
            "prime",
            "movie",
            "cinema",
            "pvr",
            "inox",
            "subscription",
            "gaming",
        ],
        "bills": [
            "electricity",
            "wifi",
            "internet",
            "mobile recharge",
            "water bill",
            "gas bill",
            "broadband",
            "recharge",
        ],
        "shopping": [
            "sneakers",
            "shoes",
            "shirt",
            "tshirt",
            "jeans",
            "dress",
            "jacket",
            "watch",
            "bag",
            "kurta",
        ],
    }

    for correct_cat, keywords in STRONG_CATEGORY_OVERRIDES.items():
        if any(kw in query_lower_correction for kw in keywords):
            if current_cat != correct_cat:
                item_for_correction["category"] = correct_cat.title()
                extracted["items"] = [item_for_correction]
            break

    # Set original_query BEFORE retrieve_assumptions so RAG scoring can use
    # item keyword overlap from the user's actual input text.
    extracted["original_query"] = req.message

    rag_result = retrieve_assumptions(extracted)
    past = rag_result.get("matched_entries", []) if isinstance(rag_result, dict) else []
    rag_assumptions = (
        rag_result.get("assumptions", {}) if isinstance(rag_result, dict) else {}
    )
    retrieval_stage = (
        rag_result.get("retrieval_stage", "semantic_fallback")
        if isinstance(rag_result, dict)
        else "semantic_fallback"
    )
    extracted["autofilled_fields"] = []
    extracted["retrieved_context"] = past
    extracted["rag_assumptions"] = rag_assumptions
    extracted["retrieval_stage"] = retrieval_stage
    return _prepare_and_record(extracted, extracted["original_query"], past)


@app.post("/save")
async def save(req: SaveRequest):
    tx_id = save_to_db(req.entry)
    return {"status": "saved", "id": tx_id}


@app.post("/analytics")
async def analytics(req: AnalyticsRequest):
    try:
        active_collection = ensure_collection()
        # Pre-filter at DB level when possible to avoid loading all rows.
        where_conditions = []
        qt = req.query_type
        if qt == "total_expense":
            where_conditions.append({"intent": "expense"})
        elif qt == "total_income":
            where_conditions.append({"intent": "income"})
        # For balance and breakdown we need both intents, so no intent filter.

        if req.filters.category:
            where_conditions.append({"category": req.filters.category})

        where_clause = None
        if len(where_conditions) == 1:
            where_clause = where_conditions[0]
        elif len(where_conditions) > 1:
            where_clause = {"$and": where_conditions}

        if where_clause:
            data = active_collection.get(where=where_clause, include=["metadatas"]).get(
                "metadatas", []
            )
        else:
            data = active_collection.get(include=["metadatas"]).get("metadatas", [])
    except Exception:
        data = []

    def parse_date_robust(ds):
        if not ds:
            return None
        try:
            return datetime.fromisoformat(ds.replace("Z", "+00:00")).replace(
                tzinfo=None
            )
        except Exception:
            return None

    def parse_amount(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    now = datetime.now()
    tr = req.filters.time_range
    if tr == "today":
        start_dt = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end_dt = now.replace(hour=23, minute=59, second=59, microsecond=999999)
    elif tr == "this_week":
        start_dt = (now - timedelta(days=now.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        end_dt = now.replace(hour=23, minute=59, second=59, microsecond=999999)
    elif tr == "this_month":
        start_dt = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_dt = now.replace(hour=23, minute=59, second=59, microsecond=999999)
    elif tr == "last_month":
        first_day = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_of_last = first_day - timedelta(seconds=1)
        start_dt = end_of_last.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_dt = end_of_last
    elif tr == "custom":
        start_dt = parse_date_robust(req.filters.start_date) or datetime.min
        end_dt = parse_date_robust(req.filters.end_date) or datetime.max
    else:
        start_dt = datetime.min
        end_dt = datetime.max

    filtered = []
    for tx in data:
        tx_dt = parse_date_robust(tx.get("datetime"))
        if not tx_dt:
            continue
        if not (start_dt <= tx_dt <= end_dt):
            continue
        if req.filters.category and tx.get("category"):
            if tx.get("category").lower() != req.filters.category.lower():
                continue
        filtered.append(tx)

    qt = req.query_type
    if qt == "total_expense":
        total = sum(
            parse_amount(tx.get("amount"))
            for tx in filtered
            if tx.get("intent") == "expense"
        )
        return {"answer": f"You spent ₹{total:,.0f}", "data": {"total": total}}
    elif qt == "total_income":
        total = sum(
            parse_amount(tx.get("amount"))
            for tx in filtered
            if tx.get("intent") == "income"
        )
        return {"answer": f"You received ₹{total:,.0f}", "data": {"total": total}}
    elif qt == "balance":
        expense = sum(
            parse_amount(tx.get("amount"))
            for tx in filtered
            if tx.get("intent") == "expense"
        )
        income = sum(
            parse_amount(tx.get("amount"))
            for tx in filtered
            if tx.get("intent") == "income"
        )
        bal = income - expense
        return {
            "answer": f"Your balance is ₹{bal:,.0f}",
            "data": {"total": bal, "income": income, "expense": expense},
        }
    elif qt == "category_breakdown":
        breakdown = {}
        for tx in filtered:
            if tx.get("intent") == "expense":
                cat = tx.get("category") or "Uncategorized"
                breakdown[cat] = breakdown.get(cat, 0) + parse_amount(tx.get("amount"))
        return {"answer": "Here is your spending breakdown", "data": breakdown}

    return {"answer": "Unknown query type", "data": {}}


@app.get("/transactions")
async def transactions():
    try:
        active_collection = ensure_collection()
        count = active_collection.count()
        if count == 0:
            return {"transactions": []}
        data = active_collection.get(include=["metadatas"])
        items = []
        for ds_id, meta in zip(data["ids"], data["metadatas"]):
            meta["id"] = ds_id
            items.append(meta)
        return {"transactions": items}
    except Exception:
        return {"transactions": []}


@app.delete("/transactions/{tx_id}")
async def delete_transaction(tx_id: str):
    try:
        active_collection = ensure_collection()
        active_collection.delete(ids=[tx_id])
        return {"status": "deleted"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/debug")
async def debug(req: ChatRequest):
    """Returns full extraction + RAG context — use to diagnose failures."""
    past = retrieve_context(req.message)

    raw_output = {"source": "llama_cpp" if llm is not None else "heuristic_fallback"}
    try:
        extracted = extract_fields(req.message, past)
    except Exception as e:
        raw_output = {"error": str(e), "source": "heuristic_fallback"}
        extracted = normalize_model_output(
            {"intent": infer_transaction_intent(req.message), "items": [empty_item()]},
            req.message,
        )

    extracted["retrieved_context"] = past
    extracted["original_query"] = req.message

    extracted_for_rag = json.loads(json.dumps(extracted))

    rag_result = retrieve_assumptions(extracted_for_rag)
    past = rag_result.get("matched_entries", []) if isinstance(rag_result, dict) else []
    enriched, autofilled_fields = autofill_from_context(
        json.loads(json.dumps(extracted)), past
    )
    missing = get_missing_required(enriched)
    explicit = detect_explicit_fields(req.message, enriched)

    # Show top-5 RAG candidates with their similarity scores
    rag_candidates = [
        {
            "intent": e.get("intent"),
            "merchant": e.get("merchant"),
            "payment_method": e.get("payment_method"),
            "category": e.get("category"),
            "similarity": round(float(e.get("_similarity", 0)), 3),
        }
        for e in sorted(
            past, key=lambda x: float(x.get("_similarity", 0)), reverse=True
        )[:5]
    ]

    rag_match_reason = [
        {
            "merchant": entry.get("merchant"),
            "category": entry.get("category"),
            "payment_method": entry.get("payment_method"),
            "similarity": round(float(entry.get("_similarity", 0)), 3),
            "base_similarity": round(float(entry.get("_base_similarity", 0)), 3),
            "keyword_boosted": float(entry.get("_similarity", 0))
            > float(entry.get("_base_similarity", 0)) + 0.05,
        }
        for entry in sorted(
            past, key=lambda x: float(x.get("_similarity", 0)), reverse=True
        )[:5]
    ]

    return {
        "raw_model_output": raw_output,
        "heuristic_intent_probe": infer_transaction_intent(req.message),
        "heuristic_source_probe": infer_income_source(req.message),
        "normalized_output": extracted,
        "raw_extracted": extracted,
        "after_autofill": enriched,
        "autofilled_fields": autofilled_fields,
        "explicit_fields_detected": list(explicit),
        "missing_required": missing,
        "will_ask_followup": len(missing) > 0,
        "next_question_about": missing[0] if missing else None,
        "rag_top_candidates": rag_candidates,
        "rag_match_reason": rag_match_reason,
        "rag_threshold": ASSUMPTION_SIMILARITY_THRESHOLD,
        "retrieval_stage": (
            rag_result.get("retrieval_stage", "semantic_fallback")
            if isinstance(rag_result, dict)
            else "semantic_fallback"
        ),
        "rag_assumptions": (
            rag_result.get("assumptions", {}) if isinstance(rag_result, dict) else {}
        ),
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "model_path": MODEL_PATH,
        "backend": "llama_cpp" if llm is not None else "heuristic_fallback",
        "intent_probe_gym": infer_transaction_intent("gym membership 1000 today"),
        "intent_probe_repay": infer_transaction_intent("Rahul paid me back 2000 today"),
    }


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "expense-backend",
        "message": "API is running. Use /chat, /save, /analytics, /transactions, /debug, /health",
    }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "9002"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    uvicorn.run("main:app", host=host, port=port, reload=reload)
