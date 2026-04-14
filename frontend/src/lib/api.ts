import { TransactionDraft } from "@/types/chat";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim() || "http://127.0.0.1:8000";

type BackendItem = {
  amount?: number | string | null;
  category?: string | null;
  item?: string | null;
  merchant?: string | null;
  payment_method?: string | null;
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

export function extractedToDraft(extracted?: BackendExtracted): TransactionDraft | null {
  if (!extracted || !extracted.intent) {
    return null;
  }

  const item = extracted.items?.[0] || {};
  const type = extracted.intent;
  const isoDate = item.datetime || undefined;

  return {
    type,
    amount: parseAmount(item.amount),
    category: type === "expense" ? (item.category || undefined) : undefined,
    source: type === "income" ? (item.category || item.item || item.merchant || undefined) : undefined,
    paymentMethod: item.payment_method || undefined,
    date: isoDate ? new Date(isoDate).toISOString().split("T")[0] : undefined,
    description: item.item || item.merchant || undefined,
    confirmed: false,
  };
}

export function draftToEntry(draft: TransactionDraft): Record<string, unknown> {
  return {
    intent: draft.type,
    items: [
      {
        amount: draft.amount,
        category: draft.type === "expense" ? draft.category || null : draft.source || null,
        currency: "INR",
        item: draft.description || null,
        merchant: null,
        payment_method: draft.paymentMethod || null,
        remarks: null,
        datetime: draft.date ? new Date(`${draft.date}T12:00:00`).toISOString() : null,
        bill_no: null,
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
    const rawDate = row.datetime || undefined;
    const date = rawDate ? new Date(rawDate).toISOString().split("T")[0] : undefined;

    return {
      type,
      amount,
      category: type === "expense" ? (row.category || undefined) : undefined,
      source: type === "income" ? (row.category || row.item || row.merchant || undefined) : undefined,
      paymentMethod: row.payment_method || undefined,
      date,
      description: row.item || undefined,
      confirmed: true,
    };
  });
}
