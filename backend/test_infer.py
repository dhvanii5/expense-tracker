import re

def infer_expense_item(text: str):
    normalized = re.sub(r"\s+", " ", (text or "").lower()).strip()
    stopwords = {"today","yesterday","tomorrow","at","from","via","using","with","by","in","on","for","and","to","the","a","an","my","our","his","her","their","took","got","had","went","was"}
    verbs = r"(?:spent|spend|paid|pay|bought|buy|ordered|order|purchased|purchase|took|got)"

    patterns = [
        rf"\b{verbs}\b(?:\s+\d+(?:[.,]\d+)?(?:\s*(?:k|lakh|lac|rs\.?|inr|rupees?))?)?\s+(?:on|for)\s+([a-z][a-z\s]{{1,40}}?)(?:\s+(?:today|yesterday|at|from|via|using|with|by|morning|evening|night|afternoon)\b|$)",
        r"(?:^|[\.!\s])([a-z]{2}[a-z\s]{1,35}?)\s+for\s+(?:\d+(?:[.,]\d+)?(?:\s*(?:k|lakh|lac|rs\.?|inr|rupees?))?)(?:\s+(?:today|yesterday|morning|evening|night)\b|$)",
        rf"\b{verbs}\b\s+([a-z][a-z\s]{{1,40}}?)(?:\s+(?:today|yesterday|at|from|via|using|with|by|morning|evening|night|afternoon)\b|$)",
        r"^(?:for|on)\s+([a-z]{2}[a-z\s]{1,35}?)(?:\s+(?:today|yesterday|\d+|morning|evening|night)\b|$)",
        r"(?:\s|^)(?:for|on)\s+([a-z]{2}[a-z\s]{1,35}?)(?:\s+(?:today|yesterday|morning|evening|night)\b|$)",
        r"\b(uber|ola|cab|taxi|auto|bus|train|movie|cinema|food|lunch|dinner|breakfast|coffee|tea|petrol|gas|grocery|groceries|gym|workout|book|ticket|subscription|music|medicine|medicines)\b",
        r"(?:a|an)\s+([a-z][a-z\s]{{1,35}}?)(?:\s+(?:today|yesterday|morning|evening)\b|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            phrase = re.sub(r"\s+", " ", match.group(1)).strip()
            tokens = [t for t in phrase.split() if t and t not in stopwords]
            verbs_list = ["spent", "spend", "paid", "pay", "bought", "buy", "ordered", "order", "purchased", "purchase", "took", "got"]
            while tokens and tokens[0].lower() in verbs_list:
                tokens.pop(0)
            if tokens:
                result = " ".join(tokens[:3]).title()
                if not any(t in result.lower() for t in ["am", "pm", "oclock", "o'clock"]):
                    return result
    return None

def infer_explicit_merchant(text: str):
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    pattern = (
        r"\b(?:from|at)\s+([a-zA-Z0-9][a-zA-Z0-9&'.\- ]{1,40}?)"
        r"(?=\s+(?:today|yesterday|for|via|using|with|by|on|in|paid|spent|bought|ordered|"
        r"received|got|rs\.?|inr|rupees?|\d)|$)"
    )
    match = re.search(pattern, normalized, re.IGNORECASE)
    if match:
        return match.group(1).strip(" .,-").title()
    return None

text = "spent 300 on lunch at swiggy today via UPI"
print("Item:", infer_expense_item(text))
print("Merchant:", infer_explicit_merchant(text))
