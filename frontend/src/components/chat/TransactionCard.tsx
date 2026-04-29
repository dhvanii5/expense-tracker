import { useState } from 'react';
import { TransactionDraft } from '@/types/chat';
import { motion, AnimatePresence } from 'framer-motion';
import { Check, Pencil, X, ArrowUpRight, ArrowDownLeft, AlertCircle, Trash2 } from 'lucide-react';

interface Props {
  transaction: TransactionDraft;
  onConfirm: (t: TransactionDraft) => void;
  onEdit: (t: TransactionDraft) => void;
  onCancelDraft?: () => void;
  onReject?: (t: TransactionDraft) => void;
  onDelete?: (t: TransactionDraft) => void;
  confirmationNeeded?: boolean;
  assumptionOptions?: Record<string, string[]>;
  allowRejectAssumptions?: boolean;
}

function formatCurrency(amount: number, currency?: string | null): string {
  const cur = currency?.trim() || 'INR';
  try {
    const formatted = new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: cur,
      maximumFractionDigits: 2,
    }).format(amount);
    
    if (cur.toUpperCase() === 'INR') {
      return formatted.replace(/[A-Za-z]+|Rs\.?/g, '₹').replace(/\s+/g, '');
    }
    return formatted;
  } catch {
    return `${cur === 'INR' ? '₹' : cur}${amount.toLocaleString('en-IN')}`;
  }
}

function formatReadableDate(dateString: string | null | undefined): string | null {
  if (!dateString) return null;

  // Compare using local date parts only to avoid timezone shifts.
  let year: number, month: number, day: number;

  const isoMatch = dateString.match(/^(\d{4})-(\d{2})-(\d{2})/);
  if (isoMatch) {
    year = Number(isoMatch[1]);
    month = Number(isoMatch[2]) - 1;
    day = Number(isoMatch[3]);
  } else {
    const d = new Date(dateString);
    if (isNaN(d.getTime())) return dateString;
    year = d.getFullYear();
    month = d.getMonth();
    day = d.getDate();
  }

  const date = new Date(year, month, day);
  const now = new Date();
  const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterdayStart = new Date(todayStart);
  yesterdayStart.setDate(todayStart.getDate() - 1);

  if (date.getTime() === todayStart.getTime()) return 'Today';
  if (date.getTime() === yesterdayStart.getTime()) return 'Yesterday';
  return new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric', year: 'numeric' }).format(date);
}

function cleanValue(value: string | null | undefined): string | null {
  if (value === null || value === undefined || value === '') return null;
  const s = String(value).trim();
  if (s.toLowerCase().startsWith('assumed') && s.length > 7) return s.slice(7).trim();
  return s;
}

// ─── FieldRow ────────────────────────────────────────────────────────────────

interface FieldRowProps {
  label: string;
  value: string | number | null | undefined;
  isAmount?: boolean;
  amountClass?: string;
  assumed?: boolean;
}

function FieldRow({ label, value, isAmount, amountClass, assumed }: FieldRowProps) {
  const displayValue = isAmount
    ? (value === null || value === undefined || value === '' ? null : String(value))
    : cleanValue(value as string | null | undefined);

  if (displayValue === null) return null;

  return (
    <div className="flex items-start justify-between gap-4 py-1.5 border-b border-border/40 last:border-0">
      <span className="text-xs text-muted-foreground shrink-0 w-28">{label}</span>
      <div className="flex items-center gap-1.5 min-w-0 justify-end">
        {assumed && (
          <span className="text-[9px] font-medium bg-amber-500/15 text-amber-600 dark:text-amber-400 px-1.5 py-0.5 rounded-full shrink-0">
            assumed
          </span>
        )}
        <span
          className={`text-sm font-medium text-right break-words max-w-[180px] ${
            isAmount ? amountClass : 'text-foreground'
          }`}
        >
          {displayValue}
        </span>
      </div>
    </div>
  );
}

// ─── ConfirmationBox ─────────────────────────────────────────────────────────

interface ConfirmationBoxProps {
  assumedFields: string[];
  assumptionOptions?: Record<string, string[]>;
  selectedAssumptions?: Record<string, string>;
  onSelectAssumption?: (field: string, value: string) => void;
  onConfirm: () => void;
  onEdit: () => void;
  onReject?: () => void;
  showReject?: boolean;
}

function fieldLabelFromBackend(field: string): string {
  const map: Record<string, string> = {
    currency: 'Currency',
    payment_method: 'Payment Method',
    category: 'Category',
    source: 'Source',
    merchant: 'Merchant',
    payer: 'Payer',
    item: 'Item',
    remarks: 'Remarks',
    bill_no: 'Bill No',
    datetime: 'Date & Time',
  };
  return map[field] ?? field;
}

function ConfirmationBox({
  assumedFields,
  assumptionOptions,
  selectedAssumptions,
  onSelectAssumption,
  onConfirm,
  onEdit,
  onReject,
  showReject,
}: ConfirmationBoxProps) {
  if (assumedFields.length > 0) {
    const sortedOptions = Object.entries(assumptionOptions || {}).map(([field, values]) => {
      const rawValues = Array.isArray(values) ? values : [String(values)];
      const cleanedValues = rawValues
        .map(v => cleanValue(v) ?? v)
        .map(v => String(v).trim())
        .filter(Boolean);
      return [field, cleanedValues] as [string, string[]];
    });
    return (
      <div className="mt-3 bg-amber-500/8 border border-amber-500/25 rounded-lg p-3 space-y-2">
        <div className="flex items-start gap-2">
          <AlertCircle className="w-3.5 h-3.5 text-amber-500 mt-0.5 shrink-0" />
          <p className="text-xs text-amber-700 dark:text-amber-400 leading-relaxed">
            We assumed: <span className="font-semibold">{assumedFields.join(', ')}</span>. Is this correct?
          </p>
        </div>
        {sortedOptions.length > 0 && (
          <div className="space-y-2">
            {sortedOptions.map(([field, values]) => {
              const safeValues = Array.isArray(values) ? values : [String(values)];
              if (!safeValues.length) return null;
              const selected = selectedAssumptions?.[field] || safeValues[0];
              return (
                <div key={field} className="space-y-1">
                  <p className="text-[11px] font-medium text-muted-foreground">{fieldLabelFromBackend(field)}</p>
                  <div className="flex flex-wrap gap-1.5">
                    {safeValues.map((option) => (
                      <button
                        key={`${field}-${option}`}
                        onClick={() => onSelectAssumption?.(field, option)}
                        className={`text-[11px] px-2.5 py-1 rounded-full border transition-colors ${
                          selected === option
                            ? 'bg-primary text-primary-foreground border-primary'
                            : 'bg-background text-foreground border-border hover:bg-muted'
                        }`}
                      >
                        {option}
                      </button>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        )}
        <div className="flex gap-2">
          <button
            onClick={onConfirm}
            className="flex items-center gap-1.5 bg-primary text-primary-foreground px-3 py-1.5 rounded-md text-xs font-medium hover:opacity-90 transition-opacity"
          >
            <Check className="w-3 h-3" /> Confirm
          </button>
          <button
            onClick={onEdit}
            className="flex items-center gap-1.5 bg-muted text-muted-foreground px-3 py-1.5 rounded-md text-xs font-medium hover:bg-muted/80 transition-colors"
          >
            <Pencil className="w-3 h-3" /> Edit
          </button>
          {showReject && onReject && (
            <button
              onClick={onReject}
              className="flex items-center gap-1.5 bg-destructive/90 text-destructive-foreground px-3 py-1.5 rounded-md text-xs font-medium hover:opacity-90 transition-opacity"
            >
              <X className="w-3 h-3" /> Reject
            </button>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="flex gap-2 mt-3">
      <button
        onClick={onConfirm}
        className="flex items-center gap-1.5 bg-primary text-primary-foreground px-3 py-1.5 rounded-md text-xs font-medium hover:opacity-90 transition-opacity"
      >
        <Check className="w-3 h-3" /> Confirm
      </button>
      <button
        onClick={onEdit}
        className="flex items-center gap-1.5 bg-muted text-muted-foreground px-3 py-1.5 rounded-md text-xs font-medium hover:bg-muted/80 transition-colors"
      >
        <Pencil className="w-3 h-3" /> Edit
      </button>
      {showReject && onReject && (
        <button
          onClick={onReject}
          className="flex items-center gap-1.5 bg-destructive/90 text-destructive-foreground px-3 py-1.5 rounded-md text-xs font-medium hover:opacity-90 transition-opacity"
        >
          <X className="w-3 h-3" /> Reject
        </button>
      )}
    </div>
  );
}

// ─── EditableField ────────────────────────────────────────────────────────────

interface EditableFieldProps {
  label: string;
  value: string | number | null | undefined;
  type?: 'text' | 'number';
  onChange: (val: string) => void;
}

function EditableField({ label, value, type = 'text', onChange }: EditableFieldProps) {
  const displayValue = type === 'number'
    ? (value ?? '')
    : (cleanValue(value as string | null | undefined) ?? '');

  return (
    <label className="block">
      <span className="text-xs text-muted-foreground">{label}</span>
      <input
        type={type}
        className="w-full bg-muted rounded-md px-3 py-1.5 mt-0.5 text-sm text-foreground outline-none focus:ring-2 focus:ring-ring/50 transition-shadow"
        value={displayValue}
        onChange={(e) => onChange(e.target.value)}
      />
    </label>
  );
}

// ─── TransactionCard ──────────────────────────────────────────────────────────

export function TransactionCard({
  transaction,
  onConfirm,
  onEdit,
  onCancelDraft,
  onReject,
  onDelete,
  confirmationNeeded,
  assumptionOptions,
  allowRejectAssumptions,
}: Props) {
  const [editing, setEditing] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [draftCancelled, setDraftCancelled] = useState(false);
  const [draft, setDraft] = useState<TransactionDraft>(() => ({
    ...transaction,
    merchant: cleanValue(transaction.merchant) ?? null,
    payer: cleanValue(transaction.payer) ?? null,
    paymentMethod: cleanValue(transaction.paymentMethod) ?? null,
    category: cleanValue(transaction.category) ?? null,
    source: cleanValue(transaction.source) ?? null,
    item: cleanValue(transaction.item) ?? null,
    currency: cleanValue(transaction.currency) ?? null,
    remarks: cleanValue(transaction.remarks) ?? null,
    bill_no: cleanValue(transaction.bill_no) ?? null,
  }));
  const [selectedAssumptions, setSelectedAssumptions] = useState<Record<string, string>>({});

  function applyAssumptionToDraft(field: string, value: string) {
    const clean = cleanValue(value) ?? value;
    setDraft((prev) => {
      const next = { ...prev };
      switch (field) {
        case 'payment_method':
          next.paymentMethod = clean;
          break;
        case 'category':
          if (next.type === 'income') {
            next.source = clean;
          } else {
            next.category = clean;
          }
          break;
        case 'source':
          next.source = clean;
          break;
        case 'merchant':
          if (next.type === 'income') {
            next.payer = clean;
          } else {
            next.merchant = clean;
          }
          break;
        case 'payer':
          next.payer = clean;
          break;
        case 'item':
          next.item = clean;
          if (!next.description) next.description = clean;
          break;
        case 'currency':
          next.currency = clean;
          break;
        case 'remarks':
          next.remarks = clean;
          break;
        case 'bill_no':
          next.bill_no = clean;
          break;
        case 'datetime':
          next.date = clean;
          break;
        default:
          break;
      }
      return next;
    });
  }

  const selectAssumption = (field: string, value: string) => {
    setSelectedAssumptions((prev) => ({ ...prev, [field]: value }));
    applyAssumptionToDraft(field, value);
  };

  const isIncome = draft.type === 'income';
  const Icon = isIncome ? ArrowDownLeft : ArrowUpRight;
  const assumed = draft.assumedFields ?? [];
  const badgeAllowedLabels = new Set([
    'Category',
    'Source',
    'Merchant',
    'Payer',
    'Item',
    'Payment Method',
    'Currency',
    'Bill No',
  ]);

  const isAssumed = (key: string) => {
    const label = fieldLabel(key);
    return badgeAllowedLabels.has(label) && assumed.includes(label);
  };

  function fieldLabel(key: string): string {
    const map: Record<string, string> = {
      currency: 'Currency',
      paymentMethod: 'Payment Method',
      category: 'Category',
      source: 'Source',
      merchant: 'Merchant',
      payer: 'Payer',
      item: 'Item',
      remarks: 'Remarks',
      bill_no: 'Bill No',
      date: 'Date & Time',
    };
    return map[key] ?? key;
  }

  const amountStr =
    draft.amount != null
      ? `${formatCurrency(draft.amount, draft.currency)}`
      : null;

  const handleSave = () => {
    setEditing(false);
    onEdit(draft);
  };

  const handleCancel = () => {
    setEditing(false);
    setDraft({
      ...transaction,
      merchant: cleanValue(transaction.merchant) ?? null,
      payer: cleanValue(transaction.payer) ?? null,
      paymentMethod: cleanValue(transaction.paymentMethod) ?? null,
      category: cleanValue(transaction.category) ?? null,
      source: cleanValue(transaction.source) ?? null,
      item: cleanValue(transaction.item) ?? null,
      currency: cleanValue(transaction.currency) ?? null,
      remarks: cleanValue(transaction.remarks) ?? null,
      bill_no: cleanValue(transaction.bill_no) ?? null,
    });
  };

  const handleCancelDraft = () => {
    setDraftCancelled(true);
    onCancelDraft?.();
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-card border rounded-xl p-4 mt-2 w-full max-w-sm shadow-sm"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div
            className={`w-8 h-8 rounded-lg flex items-center justify-center ${
              isIncome ? 'bg-income/15 text-income' : 'bg-expense/15 text-expense'
            }`}
          >
            <Icon className="w-4 h-4" />
          </div>
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            {draft.type}
          </span>
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          {confirmationNeeded && onCancelDraft && !draftCancelled && (
            <button
              onClick={handleCancelDraft}
              className="text-muted-foreground hover:text-destructive hover:bg-destructive/10 p-1.5 rounded-md transition-colors"
              title="Cancel this draft"
              aria-label="Cancel this draft"
            >
              <X className="w-4 h-4" />
            </button>
          )}
          {onDelete && draft.id && (
            confirmDelete ? (
              <div className="flex items-center gap-1.5 shrink-0">
                <button
                  onClick={() => onDelete(draft)}
                  className="text-[10px] font-medium bg-destructive text-destructive-foreground px-2 py-1 rounded hover:opacity-90 transition-opacity"
                >
                  Confirm
                </button>
                <button
                  onClick={() => setConfirmDelete(false)}
                  className="text-[10px] font-medium bg-muted text-muted-foreground px-2 py-1 rounded hover:opacity-90 transition-opacity"
                >
                  Cancel
                </button>
              </div>
            ) : (
              <button
                onClick={() => setConfirmDelete(true)}
                className="text-muted-foreground hover:text-destructive hover:bg-destructive/10 p-1.5 rounded-md transition-colors"
                title="Delete this transaction"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            )
          )}
        </div>
      </div>

      {/* ── VIEW MODE ── */}
      <AnimatePresence mode="wait">
        {!editing ? (
          <motion.div
            key="view"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
          >
            <div className="space-y-0">
              <FieldRow
                label="Amount"
                value={amountStr}
                isAmount
                amountClass={isIncome ? 'text-income font-bold text-3xl tracking-tight' : 'text-foreground font-bold text-3xl tracking-tight'}
                assumed={false}
              />
              <FieldRow
                label={isIncome ? 'Source' : 'Category'}
                value={isIncome ? draft.source : draft.category}
                assumed={isAssumed(isIncome ? 'source' : 'category')}
              />
              <FieldRow
                label={isIncome ? 'Payer' : 'Merchant'}
                value={isIncome ? draft.payer : draft.merchant}
                assumed={isAssumed(isIncome ? 'payer' : 'merchant')}
              />
              <FieldRow
                label="Item"
                value={draft.item}
                assumed={isAssumed('item')}
              />
              <FieldRow
                label="Payment Method"
                value={draft.paymentMethod}
                assumed={isAssumed('paymentMethod')}
              />
              <FieldRow
                label="Currency"
                value={draft.currency}
                assumed={isAssumed('currency')}
              />
              <FieldRow
                label="Bill No"
                value={draft.bill_no}
                assumed={isAssumed('bill_no')}
              />
              <FieldRow
                label="Date & Time"
                value={formatReadableDate(draft.date)}
                assumed={false}
              />
              <FieldRow
                label="Remarks"
                value={draft.remarks ?? 'Not provided'}
                assumed={isAssumed('remarks')}
              />
            </div>

            {confirmationNeeded && !draftCancelled && (
              <ConfirmationBox
                assumedFields={assumed}
                assumptionOptions={assumptionOptions}
                selectedAssumptions={selectedAssumptions}
                onSelectAssumption={selectAssumption}
                onConfirm={() => onConfirm({ ...draft, confirmed: true })}
                onEdit={() => setEditing(true)}
                onReject={onReject ? () => onReject({ ...draft, confirmed: false }) : undefined}
                showReject={!!allowRejectAssumptions}
              />
            )}

            {confirmationNeeded && draftCancelled && (
              <p className="text-xs text-muted-foreground mt-3">Entry cancelled.</p>
            )}
          </motion.div>
        ) : (
          /* ── EDIT MODE ── */
          <motion.div
            key="edit"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="space-y-2.5 text-sm"
          >
            <EditableField
              label="Amount"
              value={draft.amount ?? ''}
              type="number"
              onChange={(v) => setDraft((d) => ({ ...d, amount: Number(v) || null }))}
            />
            <EditableField
              label={isIncome ? 'Source' : 'Category'}
              value={(isIncome ? draft.source : draft.category) ?? ''}
              onChange={(v) =>
                setDraft((d) =>
                  isIncome ? { ...d, source: v || null } : { ...d, category: v || null }
                )
              }
            />
            <EditableField
              label={isIncome ? 'Payer' : 'Merchant'}
              value={(isIncome ? draft.payer : draft.merchant) ?? ''}
              onChange={(v) =>
                setDraft((d) =>
                  isIncome ? { ...d, payer: v || null } : { ...d, merchant: v || null }
                )
              }
            />
            <EditableField
              label="Item"
              value={draft.item ?? ''}
              onChange={(v) => setDraft((d) => ({ ...d, item: v || null }))}
            />
            <EditableField
              label="Payment Method"
              value={draft.paymentMethod ?? ''}
              onChange={(v) => setDraft((d) => ({ ...d, paymentMethod: v || null }))}
            />
            <EditableField
              label="Currency"
              value={draft.currency ?? ''}
              onChange={(v) => setDraft((d) => ({ ...d, currency: v || null }))}
            />
            <EditableField
              label="Bill No"
              value={draft.bill_no ?? ''}
              onChange={(v) => setDraft((d) => ({ ...d, bill_no: v || null }))}
            />
            <EditableField
              label="Date (YYYY-MM-DD)"
              value={draft.date ?? ''}
              onChange={(v) => setDraft((d) => ({ ...d, date: v || null }))}
            />
            <EditableField
              label="Remarks"
              value={draft.remarks ?? ''}
              onChange={(v) => setDraft((d) => ({ ...d, remarks: v || null }))}
            />

            <div className="flex gap-2 pt-1">
              <button
                onClick={handleSave}
                className="flex items-center gap-1.5 bg-primary text-primary-foreground px-3 py-1.5 rounded-md text-xs font-medium hover:opacity-90 transition-opacity"
              >
                <Check className="w-3 h-3" /> Save
              </button>
              <button
                onClick={handleCancel}
                className="flex items-center gap-1.5 bg-muted text-muted-foreground px-3 py-1.5 rounded-md text-xs font-medium hover:bg-muted/80 transition-colors"
              >
                <X className="w-3 h-3" /> Cancel
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
