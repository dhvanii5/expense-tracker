# Expense Tracker Frontend

This folder contains the Vite + React frontend for the expense tracker. It provides the chat UI, confirmation flow, and recent transaction list used with the backend API.

## Development

Install dependencies:

```bash
npm install
```

Start the dev server:

```bash
npm run dev
```

The frontend dev server runs on `http://127.0.0.1:8080` by default.

## Backend Connection

The frontend talks to the backend JSON API.

- Preferred: set `VITE_API_BASE_URL`
- Fallbacks used by the app: `http://127.0.0.1:8000` and `http://127.0.0.1:9002`

Example `.env` value:

```bash
VITE_API_BASE_URL=http://127.0.0.1:9002
```

## Typical Commands

Run the app in development mode:

```bash
npm run dev
```

Build for production:

```bash
npm run build
```

Preview the production build locally:

```bash
npm run preview
```

Run lint checks:

```bash
npm run lint
```

Run tests:

```bash
npm run test
```

## App Structure

- `src/pages/Index.tsx` - main chat experience and recent transactions sidebar
- `src/lib/api.ts` - backend API client and payload mapping
- `src/components/chat/` - chat bubbles, input, transaction cards, and recent list UI
- `src/types/chat.ts` - shared frontend transaction and chat types

## Notes

- The app uses React, Vite, TypeScript, Tailwind, and shadcn/ui-style components.
- The main user flow is: send natural-language message, review extracted transaction, confirm, then save to backend.
- If the backend is unavailable, the UI falls back to inline error messaging instead of crashing.
