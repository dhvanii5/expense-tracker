import { useState } from 'react';
import { TransactionDraft } from '@/types/chat';
import { ArrowUpRight, ArrowDownLeft, ChevronDown, ChevronUp, Trash2 } from 'lucide-react';

interface Props {
  transactions: TransactionDraft[];
  onDelete?: (id: string) => void;
}

function formatCurrency(amount: number, currency?: string | null): string {
  const cur = currency?.trim() || 'INR';
  try {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: cur,
      maximumFractionDigits: 0,
    }).format(amount);
  } catch {
    return `${cur} ${amount.toLocaleString('en-IN')}`;
  }
}

export function RecentTransactions({ transactions, onDelete }: Props) {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  function buildDetails(t: TransactionDraft) {
    return [
      { label: 'Type', value: t.type },
      { label: t.type === 'income' ? 'Source' : 'Category', value: t.type === 'income' ? t.source : t.category },
      { label: t.type === 'income' ? 'Payer' : 'Merchant', value: t.type === 'income' ? t.payer : t.merchant },
      { label: 'Item', value: t.item || t.description },
      { label: 'Payment', value: t.paymentMethod },
      { label: 'Date', value: t.date },
      { label: 'Bill No', value: t.bill_no },
      { label: 'Remarks', value: t.remarks || 'Not provided' },
      { label: 'Transaction ID', value: t.id },
    ].filter((entry) => !!entry.value);
  }

  if (transactions.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground/60">
        <p className="text-sm">No transactions yet</p>
        <p className="text-xs mt-1">Start chatting to add one!</p>
      </div>
    );
  }

  const total = transactions.reduce((acc, t) => {
    if (!t.amount) return acc;
    return t.type === 'income' ? acc + t.amount : acc - t.amount;
  }, 0);

  return (
    <div className="space-y-3">
      <div className="bg-muted/50 rounded-xl p-4">
        <p className="text-xs text-muted-foreground mb-1">Net Balance (all saved transactions)</p>
        <p className={`text-2xl font-bold ${total >= 0 ? 'text-income' : 'text-expense'}`}>
          {total >= 0 ? '+' : ''}{formatCurrency(total)}
        </p>
      </div>

      <div className="space-y-1">
        {transactions.slice().reverse().map((t, i) => {
          const rowId = t.id || `${t.type}-${t.amount}-${t.date}-${i}`;
          const isExpanded = expandedId === rowId;
          const isIncome = t.type === 'income';
          const Icon = isIncome ? ArrowDownLeft : ArrowUpRight;
          // Primary label: source/category; subtitle: item or payer/merchant, then date
          const primaryLabel = isIncome ? (t.source ?? '—') : (t.category ?? '—');
          const subtitle = t.item || (isIncome ? t.payer : t.merchant) || t.date || '';
          const remarksPreview = t.remarks?.trim() || 'Not provided';
          const details = buildDetails(t);

          return (
            <div
              key={rowId}
              className="rounded-lg border border-transparent hover:bg-muted/40 transition-colors"
            >
              <div
                onClick={() => setExpandedId(isExpanded ? null : rowId)}
                className="flex items-center gap-3 px-3 py-2.5 cursor-pointer"
              >
                <div
                  className={`w-7 h-7 rounded-md flex items-center justify-center shrink-0 ${
                    isIncome ? 'bg-income/15 text-income' : 'bg-expense/15 text-expense'
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium capitalize truncate">{primaryLabel}</p>
                  {subtitle && (
                    <p className="text-[10px] text-muted-foreground truncate">{subtitle}</p>
                  )}
                  <p className="text-[10px] text-muted-foreground truncate">Remarks: {remarksPreview}</p>
                </div>
                <span className={`text-sm font-semibold shrink-0 ${isIncome ? 'text-income' : 'text-expense'}`}>
                  {isIncome ? '+' : '-'}{formatCurrency(t.amount!, t.currency)}
                </span>
                {onDelete && t.id && (
                  <button
                    type="button"
                    className="shrink-0 p-1 rounded-md text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-colors"
                    onClick={(e) => {
                      e.stopPropagation();
                      const ok = window.confirm('Delete this transaction permanently?');
                      if (ok) {
                        onDelete(t.id);
                      }
                    }}
                    aria-label="Delete transaction"
                    title="Delete transaction"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                )}
                <button
                  type="button"
                  className="shrink-0 p-1 rounded-md text-muted-foreground hover:bg-muted"
                  onClick={(e) => {
                    e.stopPropagation();
                    setExpandedId(isExpanded ? null : rowId);
                  }}
                  aria-label={isExpanded ? 'Hide transaction details' : 'Show transaction details'}
                >
                  {isExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                </button>
              </div>

              {isExpanded && (
                <div className="mx-3 mb-2 rounded-md border bg-muted/30 px-3 py-2 space-y-1.5">
                  <div className="text-[10px] uppercase tracking-wide text-muted-foreground">Transaction details</div>
                  <div className="grid grid-cols-1 gap-1">
                    <div className="text-xs flex justify-between">
                      <span className="text-muted-foreground">Amount</span>
                      <span className="font-medium">{formatCurrency(t.amount!, t.currency)}</span>
                    </div>
                    {details.map((detail) => (
                      <div key={`${rowId}-${detail.label}`} className="text-xs flex justify-between gap-3">
                        <span className="text-muted-foreground">{detail.label}</span>
                        <span className="text-right break-all">{detail.value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
