"use client";
import { useState } from "react";
import type { Verdict } from "@/lib/types";

const BUTTONS: { verdict: Verdict; label: string; icon: string }[] = [
  { verdict: "like", label: "Like", icon: "♥" },
  { verdict: "dislike", label: "Not for me", icon: "✕" },
  { verdict: "not_this_vibe", label: "Not this vibe", icon: "↺" },
];

// Every press is saved to the person's history and updates their taste profile.
export default function ReactionBar({ movieId, source, contextId }: { movieId: number; source: "search" | "feed"; contextId?: string }) {
  const [picked, setPicked] = useState<Verdict | null>(null);
  const [busy, setBusy] = useState(false);

  async function react(verdict: Verdict) {
    setBusy(true);
    const res = await fetch("/api/react", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ movie_id: movieId, verdict, source, context_id: contextId }),
    });
    setBusy(false);
    if (res.ok) setPicked(verdict);
  }

  return (
    <div className="flex flex-wrap gap-2">
      {BUTTONS.map((b) => (
        <button
          key={b.verdict}
          disabled={busy}
          onClick={() => react(b.verdict)}
          className={`rounded-full border px-3 py-1 text-xs transition ${
            picked === b.verdict
              ? "border-accent bg-accent text-accent-ink"
              : "border-line bg-panel-2 text-muted hover:border-accent/60 hover:text-ink"
          }`}
        >
          <span aria-hidden className="mr-1">{b.icon}</span>
          {b.label}
        </button>
      ))}
    </div>
  );
}
