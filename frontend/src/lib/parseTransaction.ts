import { TransactionDraft, ChatMessage } from '@/types/chat';

const EXPENSE_KEYWORDS = ['spent', 'paid', 'bought', 'cost', 'expense', 'bill', 'rent', 'emi'];
const INCOME_KEYWORDS = ['received', 'earned', 'salary', 'income', 'credited', 'got paid', 'freelance'];

const CATEGORIES: Record<string, string[]> = {
  food: ['groceries', 'food', 'restaurant', 'lunch', 'dinner', 'breakfast', 'coffee', 'snacks', 'swiggy', 'zomato'],
  travel: ['travel', 'uber', 'ola', 'cab', 'fuel', 'petrol', 'diesel', 'flight', 'train', 'bus'],
  shopping: ['shopping', 'clothes', 'amazon', 'flipkart', 'electronics'],
  entertainment: ['movie', 'netflix', 'spotify', 'subscription', 'game'],
  bills: ['electricity', 'water', 'internet', 'phone', 'recharge', 'bill', 'rent', 'emi'],
  health: ['doctor', 'medicine', 'hospital', 'gym', 'pharmacy'],
};

const INCOME_SOURCES: Record<string, string[]> = {
  salary: ['salary', 'paycheck'],
  freelance: ['freelance', 'client', 'project', 'gig'],
  investment: ['dividend', 'interest', 'returns', 'investment'],
  refund: ['refund', 'cashback', 'reimbursement'],
};

const PAYMENT_METHODS: Record<string, string[]> = {
  UPI: ['upi', 'gpay', 'phonepe', 'paytm'],
  cash: ['cash'],
  card: ['card', 'credit', 'debit'],
  'bank transfer': ['bank', 'transfer', 'neft', 'imps'],
};

function extractAmount(text: string): number | null {
  const patterns = [
    /(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d{1,2})?)\s*k?/i,
    /([\d,]+(?:\.\d{1,2})?)\s*k?\s*(?:₹|rs\.?|rupees?|inr)/i,
    /(?:spent|paid|received|earned|got|cost|for)\s+(?:₹|rs\.?|inr)?\s*([\d,]+(?:\.\d{1,2})?)\s*k?/i,
    /([\d,]+(?:\.\d{1,2})?)\s*k\b/i,
    /\b([\d,]+(?:\.\d{1,2})?)\b/,
  ];

  for (const pattern of patterns) {
    const match = text.match(pattern);
    if (match) {
      let amount = parseFloat(match[1].replace(/,/g, ''));
      if (/k\b/i.test(text.slice(text.indexOf(match[1])))) {
        amount *= 1000;
      }
      if (amount > 0) return amount;
    }
  }
  return null;
}

function detectType(text: string): TransactionDraft['type'] {
  const lower = text.toLowerCase();
  if (INCOME_KEYWORDS.some(k => lower.includes(k))) return 'income';
  return 'expense';
}

function detectCategory(text: string): string | undefined {
  const lower = text.toLowerCase();
  for (const [cat, keywords] of Object.entries(CATEGORIES)) {
    if (keywords.some(k => lower.includes(k))) return cat;
  }
  return undefined;
}

function detectSource(text: string): string | undefined {
  const lower = text.toLowerCase();
  for (const [src, keywords] of Object.entries(INCOME_SOURCES)) {
    if (keywords.some(k => lower.includes(k))) return src;
  }
  return undefined;
}

function detectPaymentMethod(text: string): string | undefined {
  const lower = text.toLowerCase();
  for (const [method, keywords] of Object.entries(PAYMENT_METHODS)) {
    if (keywords.some(k => lower.includes(k))) return method;
  }
  return undefined;
}

function formatCurrency(amount: number): string {
  return new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(amount);
}

export function processUserMessage(text: string, pendingDraft?: TransactionDraft | null): ChatMessage {
  const id = crypto.randomUUID();
  const type = detectType(text);
  const amount = extractAmount(text);
  const category = type === 'expense' ? detectCategory(text) : undefined;
  const source = type === 'income' ? (detectSource(text) || detectCategory(text)) : undefined;
  const paymentMethod = detectPaymentMethod(text);

  // If there's a pending draft and user is answering a follow-up
  if (pendingDraft && !pendingDraft.confirmed) {
    const updated = { ...pendingDraft };
    if (amount && !updated.amount) updated.amount = amount;
    if (!updated.category && type === 'expense') updated.category = detectCategory(text) || text.trim();
    if (!updated.source && type === 'income') updated.source = detectSource(text) || text.trim();
    if (!updated.paymentMethod) updated.paymentMethod = detectPaymentMethod(text);

    const missing = getMissingFields(updated);
    if (missing.length === 0) {
      return {
        id, role: 'ai', timestamp: new Date(),
        content: `Recording ${updated.type === 'income' ? '📈' : '📉'} ${formatCurrency(updated.amount!)} for **${updated.type === 'expense' ? updated.category : updated.source}**${updated.paymentMethod ? ` via ${updated.paymentMethod}` : ''}. Correct?`,
        transaction: updated,
        confirmationNeeded: true,
        suggestions: ['Confirm', 'Edit'],
      };
    }
    return {
      id, role: 'ai', timestamp: new Date(),
      content: `Got it! Still need: **${missing.join(', ')}**`,
      transaction: updated,
      followUp: `What's the ${missing[0]}?`,
    };
  }

  const draft: TransactionDraft = {
    type, amount, category, source, paymentMethod,
    date: new Date().toISOString().split('T')[0],
    confirmed: false,
  };

  const missing = getMissingFields(draft);

  if (missing.length === 0) {
    return {
      id, role: 'ai', timestamp: new Date(),
      content: `Recording ${type === 'income' ? '📈' : '📉'} ${formatCurrency(amount!)} for **${type === 'expense' ? category : source}**${paymentMethod ? ` via ${paymentMethod}` : ''}. Correct?`,
      transaction: draft,
      confirmationNeeded: true,
      suggestions: ['Confirm', 'Edit'],
    };
  }

  return {
    id, role: 'ai', timestamp: new Date(),
    content: amount
      ? `I see a ${type} of ${formatCurrency(amount)}. I need a few more details.`
      : `I couldn't detect an amount. Could you clarify?`,
    transaction: draft,
    followUp: `What's the ${missing[0]}?`,
    suggestions: type === 'expense'
      ? ['Food', 'Travel', 'Shopping', 'Bills', 'Entertainment']
      : ['Salary', 'Freelance', 'Investment', 'Refund'],
  };
}

function getMissingFields(draft: TransactionDraft): string[] {
  const missing: string[] = [];
  if (!draft.amount) missing.push('amount');
  if (draft.type === 'expense' && !draft.category) missing.push('category');
  if (draft.type === 'income' && !draft.source) missing.push('source');
  return missing;
}
