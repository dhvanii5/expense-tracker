import { useState, useRef, useEffect } from 'react';
import { Send, X } from 'lucide-react';

interface Props {
  onSend: (text: string) => void;
  onCancelEntry?: () => void;
  canCancelEntry?: boolean;
  disabled?: boolean;
}

export function ChatInput({ onSend, onCancelEntry, canCancelEntry, disabled }: Props) {
  const [value, setValue] = useState('');
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, [disabled]);

  const handleSubmit = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue('');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="border-t bg-card/80 backdrop-blur-sm px-4 py-3">
      <div className="max-w-2xl mx-auto space-y-2">
        {canCancelEntry && onCancelEntry && (
          <div className="flex justify-end">
            <button
              onClick={onCancelEntry}
              className="inline-flex items-center gap-1.5 text-xs bg-muted text-muted-foreground px-2.5 py-1.5 rounded-md hover:bg-muted/80 transition-colors"
              title="Cancel current entry"
              aria-label="Cancel current entry"
            >
              <X className="w-3 h-3" /> Cancel entry
            </button>
          </div>
        )}
        <div className="flex items-end gap-2">
        <textarea
          ref={inputRef}
          value={value}
          onChange={e => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type something like &quot;Spent 500 on groceries&quot;..."
          rows={1}
          disabled={disabled}
          className="flex-1 bg-muted rounded-xl px-4 py-3 text-sm text-foreground placeholder:text-muted-foreground/60 outline-none resize-none focus:ring-2 focus:ring-ring/30 transition-shadow disabled:opacity-50"
        />
        <button
          onClick={handleSubmit}
          disabled={!value.trim() || disabled}
          className="bg-primary text-primary-foreground w-10 h-10 rounded-xl flex items-center justify-center hover:opacity-90 transition-opacity disabled:opacity-40 shrink-0"
        >
          <Send className="w-4 h-4" />
        </button>
        </div>
      </div>
    </div>
  );
}
