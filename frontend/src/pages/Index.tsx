import { useState, useRef, useEffect, useCallback } from 'react';
import { ChatMessage, TransactionDraft } from '@/types/chat';
import { chatWithBackend, draftToEntry, extractedToDraft, fetchTransactions, saveTransaction } from '@/lib/api';
import { ChatBubble } from '@/components/chat/ChatBubble';
import { ChatInput } from '@/components/chat/ChatInput';
import { TypingIndicator } from '@/components/chat/TypingIndicator';
import { RecentTransactions } from '@/components/chat/RecentTransactions';
import { Wallet, PanelRightOpen, PanelRightClose } from 'lucide-react';

const WELCOME: ChatMessage = {
  id: 'welcome',
  role: 'ai',
  content: "Hey! 👋 I'm your personal finance assistant. Tell me about your expenses or income in natural language — like **\"Spent 500 on groceries\"** or **\"Received 25k salary\"**.",
  timestamp: new Date(),
  suggestions: ['Spent 200 on food', 'Received 50k salary', 'Paid 1500 rent'],
};

export default function Index() {
  const [messages, setMessages] = useState<ChatMessage[]>([WELCOME]);
  const [typing, setTyping] = useState(false);
  const [pendingDraft, setPendingDraft] = useState<TransactionDraft | null>(null);
  const [confirmedTransactions, setConfirmedTransactions] = useState<TransactionDraft[]>([]);
  const [sessionData, setSessionData] = useState<Record<string, unknown> | null>(null);
  const [followupField, setFollowupField] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = useCallback(() => {
    setTimeout(() => scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' }), 50);
  }, []);

  useEffect(scrollToBottom, [messages, typing, scrollToBottom]);

  useEffect(() => {
    let isMounted = true;
    fetchTransactions()
      .then((items) => {
        if (isMounted) {
          setConfirmedTransactions(items);
        }
      })
      .catch(() => {
        addMessage({
          id: crypto.randomUUID(),
          role: 'ai',
          content: 'I can chat, but I could not load previous transactions from the backend yet.',
          timestamp: new Date(),
        });
      });

    return () => {
      isMounted = false;
    };
  }, []);

  const addMessage = (msg: ChatMessage) => setMessages(prev => [...prev, msg]);

  const handleSend = useCallback(async (text: string) => {
    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: text,
      timestamp: new Date(),
    };
    addMessage(userMsg);
    setTyping(true);

    try {
      const response = await chatWithBackend({
        message: text,
        sessionData,
        followupField,
      });
      const nextDraft = extractedToDraft(response.extracted);

      if (response.status === 'followup') {
        setPendingDraft(nextDraft);
        setSessionData((response.extracted as Record<string, unknown>) || null);
        setFollowupField(response.followup_field || null);
        addMessage({
          id: crypto.randomUUID(),
          role: 'ai',
          content: response.question || 'Need one more detail to continue.',
          timestamp: new Date(),
          transaction: nextDraft || undefined,
          followUp: response.followup_field ? `Please share ${response.followup_field}.` : undefined,
        });
      } else {
        setPendingDraft(nextDraft);
        setSessionData(null);
        setFollowupField(null);
        addMessage({
          id: crypto.randomUUID(),
          role: 'ai',
          content: 'I have all required details. Please confirm to save this transaction.',
          timestamp: new Date(),
          transaction: nextDraft || undefined,
          confirmationNeeded: !!nextDraft,
          suggestions: nextDraft ? ['Confirm', 'Edit'] : undefined,
        });
      }
    } catch {
      addMessage({
        id: crypto.randomUUID(),
        role: 'ai',
        content: 'Backend is not reachable right now. Start the API server and try again.',
        timestamp: new Date(),
      });
    } finally {
      setTyping(false);
    }
  }, [followupField, sessionData]);

  const handleConfirm = useCallback(async (t: TransactionDraft) => {
    try {
      await saveTransaction(draftToEntry(t));
      setConfirmedTransactions(prev => [...prev, { ...t, confirmed: true }]);
      setPendingDraft(null);
      setSessionData(null);
      setFollowupField(null);
      addMessage({
        id: crypto.randomUUID(),
        role: 'ai',
        content: 'Transaction saved to backend and database successfully. What else can I log?',
        timestamp: new Date(),
      });
    } catch {
      addMessage({
        id: crypto.randomUUID(),
        role: 'ai',
        content: 'I could not save this transaction to the backend. Please retry once the API is up.',
        timestamp: new Date(),
      });
    }
  }, []);

  const handleEdit = useCallback((t: TransactionDraft) => {
    setPendingDraft(t);
    addMessage({
      id: crypto.randomUUID(),
      role: 'ai',
      content: 'Updated draft captured. Confirm to save, or send more details to adjust fields.',
      timestamp: new Date(),
      transaction: t,
      confirmationNeeded: true,
      suggestions: ['Confirm', 'Edit'],
    });
  }, []);

  return (
    <div className="flex h-screen bg-background overflow-hidden">
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <header className="border-b bg-card/80 backdrop-blur-sm px-4 py-3 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-primary flex items-center justify-center">
              <Wallet className="w-5 h-5 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-sm font-semibold">FinChat</h1>
              <p className="text-[10px] text-muted-foreground">AI Finance Assistant</p>
            </div>
          </div>
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="md:hidden w-9 h-9 rounded-lg flex items-center justify-center hover:bg-muted transition-colors"
          >
            {sidebarOpen ? <PanelRightClose className="w-4 h-4" /> : <PanelRightOpen className="w-4 h-4" />}
          </button>
        </header>

        {/* Messages */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto scrollbar-thin py-4 space-y-1">
          {messages.map(msg => (
            <ChatBubble
              key={msg.id}
              message={msg}
              onConfirm={handleConfirm}
              onEdit={handleEdit}
              onSuggestionClick={handleSend}
            />
          ))}
          {typing && <TypingIndicator />}
        </div>

        {/* Input */}
        <ChatInput onSend={handleSend} disabled={typing} />
      </div>

      {/* Right Sidebar - Recent Transactions */}
      <aside className={`${sidebarOpen ? 'translate-x-0' : 'translate-x-full'} md:translate-x-0 fixed md:static right-0 top-0 h-full z-30 w-72 border-l bg-card p-4 transition-transform duration-200 overflow-y-auto scrollbar-thin`}>
        <h2 className="text-sm font-semibold mb-4">Recent Transactions</h2>
        <RecentTransactions transactions={confirmedTransactions} />
      </aside>

      {/* Mobile overlay */}
      {sidebarOpen && (
        <div className="fixed inset-0 bg-foreground/20 z-20 md:hidden" onClick={() => setSidebarOpen(false)} />
      )}
    </div>
  );
}
