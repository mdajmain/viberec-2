"use client";
import { useRouter } from "next/navigation";
import { useState } from "react";

export const EXAMPLES = [
  "A light, funny romance like How to Lose a Guy in 10 Days",
  "Something emotional like Forrest Gump, but all positive",
  "Like The Hangover: extremely funny, genuine comedy",
  "A superhero movie like The Amazing Spider-Man, maybe animated",
];

// The prompt box: a film you love plus the feel you want. Sends you to /search?q=...
export default function SearchBox({ initial = "", onSearch, big = false }: { initial?: string; onSearch?: (q: string) => void; big?: boolean }) {
  const router = useRouter();
  const [q, setQ] = useState(initial);
  const go = (text: string) => {
    const t = text.trim();
    if (t.length < 3) return;
    if (onSearch) onSearch(t);
    else router.push(`/search?q=${encodeURIComponent(t)}`);
  };
  return (
    <div className="w-full">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          go(q);
        }}
        className={`flex w-full items-center gap-2 rounded-2xl border border-line bg-panel p-2 shadow-xl shadow-black/30 focus-within:border-accent/70 ${big ? "sm:p-2.5" : ""}`}
      >
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Something like How to Lose a Guy in 10 Days, but more urban…"
          className={`min-w-0 flex-1 bg-transparent px-3 outline-none placeholder:text-muted/70 ${big ? "py-3 text-base sm:text-lg" : "py-2 text-sm"}`}
        />
        <button className={`rounded-xl bg-accent font-semibold text-accent-ink hover:brightness-110 ${big ? "px-5 py-3" : "px-4 py-2 text-sm"}`}>Find</button>
      </form>
      <div className="mt-3 flex flex-wrap gap-2">
        {EXAMPLES.map((x) => (
          <button key={x} onClick={() => { setQ(x); go(x); }} className="rounded-full border border-line px-3 py-1 text-xs text-muted hover:border-accent/50 hover:text-ink">
            {x}
          </button>
        ))}
      </div>
    </div>
  );
}
