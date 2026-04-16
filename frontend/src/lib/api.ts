import { TransactionDraft } from "@/types/chat";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim() || "http://127.0.0.1:8000";

type BackendItem = {
  amount?: number | string | null;
  category?: string | null;
  currency?: string | null;
  item?: string | null;
  merchant?: string | null;
  source?: string | null;
  payer?: string | null;
  payment_method?: string | null;
  remarks?: string | null;
  bill_no?: string | null;
  datetime?: string | null;
};

type BackendExtracted = {
  intent?: "expense" | "income" | null;
  items?: BackendItem[];
  autofilled_fields?: string[];
};

export type BackendChatResponse = {
  status: "followup" | "complete";
  question?: string;
  followup_field?: string;
  extracted?: BackendExtracted;
};

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body || res.statusText}`);
  }

  return res.json() as Promise<T>;
}

function parseAmount(value: number | string | null | undefined): number | null {
  if (value === null || value === undefined || value === "") {
    return null;
  }

  if (typeof value === "number") {
    return Number.isFinite(value) ? value : null;
  }

  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function toDateOnly(value: string | null | undefined): string | undefined {
  if (!value) {
    return undefined;
  }

  const text = value.trim();
  const dateOnly = text.match(/^(\d{4}-\d{2}-\d{2})/);
  if (dateOnly) {
    return dateOnly[1];
  }

  const parsed = new Date(text);
  if (Number.isNaN(parsed.getTime())) {
    return undefined;
  }
  return parsed.toISOString().split("T")[0];
}

export function extractedToDraft(extracted?: BackendExtracted): TransactionDraft | null {
  if (!extracted || !extracted.intent) {
    return null;
  }

  const item = extracted.items?.[0] || {};
  const type = extracted.intent;
  const source = item.source || item.category || item.item || item.merchant || undefined;
  const itemLabel = item.item || undefined;
  const merchant = item.merchant || undefined;

  return {
    type,
    amount: parseAmount(item.amount),
    category: type === "expense" ? (item.category || undefined) : undefined,
    merchant: type === "expense" ? merchant : undefined,
    source: type === "income" ? source : undefined,
    payer: type === "income" ? (item.payer || merchant) : undefined,
    item: itemLabel,
    paymentMethod: item.payment_method || undefined,
    date: toDateOnly(item.datetime),
    description: itemLabel || merchant,
    currency: item.currency || "INR",
    remarks: item.remarks || undefined,
    bill_no: item.bill_no || undefined,
    confirmed: false,
  };
}

export function draftToEntry(draft: TransactionDraft): Record<string, unknown> {
  const itemValue = draft.item || draft.description || null;
  const merchantValue = draft.type === "expense" ? (draft.merchant || null) : null;
  const payerValue = draft.type === "income" ? (draft.payer || draft.merchant || null) : null;
  const sourceValue = draft.type === "income" ? (draft.source || draft.category || null) : null;

  return {
    intent: draft.type,
    items: [
      {
        amount: draft.amount,
        category: draft.type === "expense" ? draft.category || null : sourceValue,
        currency: draft.currency || "INR",
        item: itemValue,
        merchant: merchantValue,
        source: sourceValue,
        payer: payerValue,
        payment_method: draft.paymentMethod || null,
        remarks: draft.remarks || null,
        datetime: draft.date ? new Date(`${draft.date}T12:00:00`).toISOString() : null,
        bill_no: draft.bill_no || null,
      },
    ],
  };
}

export async function chatWithBackend(params: {
  message: string;
  sessionData?: Record<string, unknown> | null;
  followupField?: string | null;
}): Promise<BackendChatResponse> {
  return apiFetch<BackendChatResponse>("/chat", {
    method: "POST",
    body: JSON.stringify({
      message: params.message,
      session_data: params.sessionData || null,
      followup_field: params.followupField || null,
    }),
  });
}

export async function saveTransaction(entry: Record<string, unknown>): Promise<void> {
  await apiFetch<{ status: string }>("/save", {
    method: "POST",
    body: JSON.stringify({ entry }),
  });
}

export async function fetchTransactions(): Promise<TransactionDraft[]> {
  const data = await apiFetch<{ transactions: Array<Record<string, string>> }>("/transactions", {
    method: "GET",
  });

  return (data.transactions || []).map((row) => {
    const type = (row.intent === "income" ? "income" : "expense") as TransactionDraft["type"];
    const amount = parseAmount(row.amount);
    const date = toDateOnly(row.datetime);

    return {
      type,
      amount,
      category: type === "expense" ? (row.category || undefined) : undefined,
      merchant: type === "expense" ? (row.merchant || undefined) : undefined,
      source: type === "income" ? (row.source || row.category || row.item || row.merchant || undefined) : undefined,
      payer: type === "income" ? (row.payer || row.merchant || undefined) : undefined,
      paymentMethod: row.payment_method || undefined,
      date,
      item: row.item || undefined,
      description: row.item || row.merchant || undefined,
      currency: row.currency || "INR",
      remarks: row.remarks || undefined,
      bill_no: row.bill_no || undefined,
      confirmed: true,
    };
  });
}
