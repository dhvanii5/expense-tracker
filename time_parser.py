from datetime import datetime, timedelta
import re
 
 
def contains_relative_datetime_term(text: str) -> bool:
    """True when input contains a relative time phrase that should resolve at extraction time."""
    normalized = (text or "").lower()
    relative_terms = [
        "today",
        "yesterday",
        "just now",
        "this morning",
        "tonight",
        "last night",
        "this week",
    ]
    if any(term in normalized for term in relative_terms):
        return True
    if re.search(r"\b\d+\s+days?\s+(ago|back)\b", normalized):
        return True
    return False


def parse_datetime(text: str) -> str:
    """
    Always returns a datetime string.
    - If user mentions a time/date → return that specific datetime
    - Otherwise → return current machine datetime
    """
    now = datetime.now()
    text_lower = text.lower()

    if "just now" in text_lower:
        return now.strftime("%Y-%m-%d %H:%M")
 
    # --- Relative day keywords ---
    if "yesterday" in text_lower:
        return (now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M")
    if "tomorrow" in text_lower:
        return (now + timedelta(days=1)).strftime("%Y-%m-%d %H:%M")
    if "today" in text_lower:
        return now.strftime("%Y-%m-%d %H:%M")
    if "this week" in text_lower:
        start_of_week = (now - timedelta(days=now.weekday())).replace(hour=9, minute=0, second=0, microsecond=0)
        return start_of_week.strftime("%Y-%m-%d %H:%M")
 
    # --- Time of day keywords ---
    time_of_day = {
        "this morning":   now.replace(hour=9,  minute=0, second=0),
        "this afternoon": now.replace(hour=14, minute=0, second=0),
        "this evening":   now.replace(hour=18, minute=0, second=0),
        "tonight":        now.replace(hour=20, minute=0, second=0),
        "last night":     (now - timedelta(days=1)).replace(hour=21, minute=0, second=0),
        "morning":        now.replace(hour=9,  minute=0, second=0),
        "afternoon":      now.replace(hour=14, minute=0, second=0),
        "evening":        now.replace(hour=18, minute=0, second=0),
        "night":          now.replace(hour=21, minute=0, second=0),
    }
    for keyword, dt in time_of_day.items():
        if keyword in text_lower:
            return dt.strftime("%Y-%m-%d %H:%M")
 
    # --- Explicit time: "at 3pm", "at 14:30", "at 9:00 am" ---
    time_pattern = re.search(
        r'\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b',
        text_lower
    )
    if time_pattern:
        hour = int(time_pattern.group(1))
        minute = int(time_pattern.group(2)) if time_pattern.group(2) else 0
        meridiem = time_pattern.group(3)
        if meridiem == "pm" and hour != 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        try:
            return now.replace(hour=hour, minute=minute, second=0).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            pass
 
    # --- Explicit date: "12th March", "5 Jan", "12 march 2025" ---
    months = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4,
        "may": 5, "jun": 6, "jul": 7, "aug": 8,
        "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        "january": 1, "february": 2, "march": 3, "april": 4,
        "june": 6, "july": 7, "august": 8, "september": 9,
        "october": 10, "november": 11, "december": 12,
    }
    date_pattern = re.search(
        r'\b(\d{1,2})(?:st|nd|rd|th)?\s+(' + '|'.join(months.keys()) + r')(?:\s+(\d{4}))?\b',
        text_lower
    )
    if date_pattern:
        day = int(date_pattern.group(1))
        month = months[date_pattern.group(2)]
        year = int(date_pattern.group(3)) if date_pattern.group(3) else now.year
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            pass
 
    # --- ISO date: "2025-03-12" ---
    iso = re.search(r'\b(\d{4})-(\d{2})-(\d{2})\b', text)
    if iso:
        try:
            return datetime(int(iso.group(1)), int(iso.group(2)), int(iso.group(3))).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            pass
 
    # --- DD/MM/YYYY or DD-MM-YYYY ---
    dmy = re.search(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b', text)
    if dmy:
        try:
            day, month, year = int(dmy.group(1)), int(dmy.group(2)), int(dmy.group(3))
            if year < 100:
                year += 2000
            return datetime(year, month, day).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            pass
 
    # --- Relative: "2 days ago", "3 days back" ---
    days_ago = re.search(r'(\d+)\s+days?\s+(ago|back)', text_lower)
    if days_ago:
        return (now - timedelta(days=int(days_ago.group(1)))).strftime("%Y-%m-%d %H:%M")
 
    # --- Weeks ago ---
    weeks_ago = re.search(r'(\d+)\s+weeks?\s+(ago|back)', text_lower)
    if weeks_ago:
        return (now - timedelta(weeks=int(weeks_ago.group(1)))).strftime("%Y-%m-%d %H:%M")
 
    # --- Last Monday/Tuesday etc ---
    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for i, day in enumerate(weekdays):
        if f"last {day}" in text_lower:
            days_behind = (now.weekday() - i) % 7 or 7
            return (now - timedelta(days=days_behind)).strftime("%Y-%m-%d %H:%M")
 
    # --- Default: current machine datetime ---
    return now.strftime("%Y-%m-%d %H:%M")
 
 
def parse_bill_no(text: str) -> str | None:
    """Extract bill/invoice/receipt/transaction number from text."""
    patterns = [
        r'bill\s*(?:no|num|number|#)[:\s]*([A-Za-z0-9\-\/]+)',
        r'invoice\s*(?:no|num|number|#)?[:\s]*([A-Za-z0-9\-\/]+)',
        r'receipt\s*(?:no|num|number|#)?[:\s]*([A-Za-z0-9\-\/]+)',
        r'txn\s*(?:id|no|#)?[:\s]*([A-Za-z0-9\-\/]+)',
        r'transaction\s*(?:id|no|#)?[:\s]*([A-Za-z0-9\-\/]+)',
        r'order\s*(?:id|no|#)?[:\s]*([A-Za-z0-9\-\/]+)',
        r'#([A-Za-z0-9\-\/]{4,})',
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group(1).upper()
    return None
 
 
def build_remarks(user_input: str, model_remarks: str | None) -> str | None:
    """
    Always try to return a meaningful remarks string.
    Uses model's remarks if present, otherwise infers from input.
    """
    if model_remarks and str(model_remarks).strip() not in ("", "null", "None"):
        return model_remarks
 
    text_lower = user_input.lower()
 
    # Person payment context
    person_patterns = [
        (r'(?:paid|gave|sent|given)\s+(?:to\s+)?(\w+)', "Paid to {}"),
        (r'(?:for|to)\s+(mom|dad|sister|brother|friend|roommate|bro|sis)', "Paid to {}"),
    ]
    for pattern, template in person_patterns:
        match = re.search(pattern, text_lower)
        if match:
            name = match.group(1).capitalize()
            if name.lower() not in ("the", "a", "an", "my", "his", "her", "our"):
                return template.format(name)
 
    # Purpose context
    purpose_patterns = [
        (r'for\s+(birthday|anniversary|diwali|festival|wedding|gift)', "{}"),
        (r'(monthly|annual|yearly)\s+\w+', "{}"),
        (r'(tip|donation|charity)', "{}"),
        (r'from\s+(local|roadside|nearby)\s+\w+', "{}"),
    ]
    for pattern, template in purpose_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1).capitalize()
 
    return None
 
 
def enrich_expense(result: dict, user_input: str) -> dict:
    """
    Enrich expense result with:
    - datetime (always present, machine time if not mentioned)
    - bill_no (from input or null)
    - payment_method default = "cash" if null
    - remarks always present (never missing)
    """
    if result.get("intent") != "expense":
        return result
 
    dt = parse_datetime(user_input)
    bill = parse_bill_no(user_input)
 
    for item in result.get("items", []):
        # Default payment_method to cash
        if not item.get("payment_method"):
            item["payment_method"] = "cash"
 
        # Always ensure remarks is present and meaningful
        item["remarks"] = build_remarks(user_input, item.get("remarks"))
 
        # Always add datetime
        item["datetime"] = dt
 
        # Add bill_no
        item["bill_no"] = bill
 
    return result
 
 
# ── Quick test ──────────────────────────────────────────────────
if __name__ == "__main__":
    from pprint import pprint
 
    test_cases = [
        "I spent ₹250 at Zomato yesterday",
        "paid electricity bill 1200 this morning",
        "bought shoes 3000 on 12th March",
        "Uber 200 2 days ago",
        "bill no 1234 groceries 500 at DMart",
        "txn id TXN789 paid Netflix 649",
        "receipt #ABC123 dinner 800 rupees",
        "spent 500 on food last Monday",
        "Swiggy 350 at 7pm",
        "gave mom 1000",                        # no time → machine time
        "paid Rahul 500 at his shop",           # person payment
        "I spent ₹250 at Zomato",              # no time → machine time
        "gave sister 500 for birthday",         # purpose in remarks
        "roadside stall momos 50rs",            # local context
    ]
 
    for text in test_cases:
        print(f"INPUT:    {text}")
        print(f"DATETIME: {parse_datetime(text)}")
        print(f"BILL NO:  {parse_bill_no(text)}")
        print(f"REMARKS:  {build_remarks(text, None)}")
        print()