import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import { ChatMessage } from '@/types/chat';
import { TransactionCard } from './TransactionCard';
import { TransactionDraft } from '@/types/chat';

interface Props {
  message: ChatMessage;
  onConfirm: (t: TransactionDraft) => void;
  onEdit: (t: TransactionDraft) => void;
  onCancelDraft?: () => void;
  onReject?: (t: TransactionDraft) => void;
  onDelete?: (t: TransactionDraft) => void;
  onSuggestionClick: (s: string) => void;
}

export function ChatBubble({ message, onConfirm, onEdit, onCancelDraft, onReject, onDelete, onSuggestionClick }: Props) {
  const isUser = message.role === 'user';
  const showQueryAnswer = message.queryResult && message.queryResult.answer !== message.content;

  function getQueryTitle(type?: string): string {
    switch (type) {
      case 'category_breakdown':
        return 'Expense breakdown by category';
      case 'total_expense':
        return 'Total expenses';
      case 'total_income':
        return 'Total income';
      case 'balance':
        return 'Net balance across all saved transactions';
      default:
        return 'Query result';
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className={`flex ${isUser ? 'justify-end' : 'justify-start'} px-4 py-1`}
    >
      <div className={`flex gap-3 max-w-[85%] md:max-w-[70%] ${isUser ? 'flex-row-reverse' : ''}`}>
        {!isUser && (
          <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center text-primary-foreground text-xs font-semibold shrink-0 mt-1">
            AI
          </div>
        )}
        <div>
          <div className={`rounded-2xl px-4 py-2.5 ${
            isUser
              ? 'bg-chat-user text-chat-user-foreground rounded-tr-sm'
              : 'bg-chat-ai text-chat-ai-foreground rounded-tl-sm'
          }`}>
            <div className="text-sm leading-relaxed prose prose-sm max-w-none prose-p:m-0 prose-strong:text-inherit">
              <ReactMarkdown>{message.content}</ReactMarkdown>
            </div>
          </div>

          {message.followUp && (
            <p className="text-xs text-muted-foreground mt-1.5 ml-1">{message.followUp}</p>
          )}

          {message.transaction && (
            <TransactionCard
              transaction={message.transaction}
              onConfirm={onConfirm}
              onEdit={onEdit}
              onReject={onReject}
              onDelete={onDelete}
              confirmationNeeded={message.confirmationNeeded}
              assumptionOptions={message.assumptionOptions}
              allowRejectAssumptions={message.allowRejectAssumptions}
              onCancelDraft={message.confirmationNeeded ? onCancelDraft : undefined}
            />
          )}

          {message.transactions && message.transactions.length > 0 && (
            <div className="flex flex-col gap-2 mt-2">
              {message.transactions.map((tx, idx) => (
                <TransactionCard
                  key={tx.id || idx}
                  transaction={tx}
                  onConfirm={onConfirm}
                  onEdit={onEdit}
                  onReject={onReject}
                  onDelete={onDelete}
                  confirmationNeeded={message.confirmationNeeded}
                  assumptionOptions={message.assumptionOptions}
                  allowRejectAssumptions={message.allowRejectAssumptions}
                  onCancelDraft={message.confirmationNeeded ? onCancelDraft : undefined}
                />
              ))}
            </div>
          )}

          {message.queryResult && (
            <div className="mt-3 bg-primary/5 border border-primary/20 rounded-xl p-4 w-full shadow-sm">
              <p className="text-[11px] uppercase tracking-wide text-muted-foreground font-semibold">
                {getQueryTitle(message.queryResult.type)}
              </p>
              {showQueryAnswer && (
                <p className="text-sm font-medium text-foreground leading-relaxed mt-1">
                  {message.queryResult.answer}
                </p>
              )}
              {message.queryResult.breakdown && Object.keys(message.queryResult.breakdown).length > 0 && (
                <div className="mt-3 text-xs text-muted-foreground font-mono bg-background p-2.5 rounded-md border border-border shadow-inner whitespace-pre-wrap">
                  {typeof message.queryResult.breakdown === 'string' 
                    ? message.queryResult.breakdown 
                    : JSON.stringify(message.queryResult.breakdown, null, 2)}
                </div>
              )}
            </div>
          )}

          {message.suggestions && !message.confirmationNeeded && (
            <div className="flex flex-wrap gap-1.5 mt-2">
              {message.suggestions.map(s => (
                <button
                  key={s}
                  onClick={() => onSuggestionClick(s)}
                  className="bg-card border text-foreground text-xs px-3 py-1.5 rounded-full hover:bg-muted transition-colors font-medium"
                >
                  {s}
                </button>
              ))}
            </div>
          )}

          <p className="text-[10px] text-muted-foreground/50 mt-1 ml-1">
            {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </p>
        </div>
      </div>
    </motion.div>
  );
}
