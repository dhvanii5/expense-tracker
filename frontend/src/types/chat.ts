export type TransactionType = 'expense' | 'income';

export interface TransactionDraft {
  id?: string;
  type: TransactionType;
  amount: number | null;

  // Expense-specific
  category?: string | null;
  merchant?: string | null;

  // Income-specific
  source?: string | null;
  payer?: string | null;

  // Shared fields
  paymentMethod?: string | null;
  date?: string | null;
  description?: string | null;
  currency?: string | null;
  item?: string | null;
  remarks?: string | null;
  bill_no?: string | null;

  /** Fields that were auto-filled / assumed by the backend */
  assumedFields?: string[];

  confirmed: boolean;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'ai';
  content: string;
  timestamp: Date;
  transaction?: TransactionDraft;
  transactions?: TransactionDraft[];
  queryResult?: { type?: string; answer: string; breakdown?: any };
  followUp?: string;
  suggestions?: string[];
  confirmationNeeded?: boolean;
  allowRejectAssumptions?: boolean;
  assumptionOptions?: Record<string, string[]>;
}
