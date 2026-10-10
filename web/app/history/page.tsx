"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import Loading from "@/components/Loading";
import Poster from "@/components/Poster";
import type { History, Verdict } from "@/lib/types";

const VERDICT: Record<Verdict, { label: string; style: string }> = {
  like: { label: "Liked", style: "bg-accent text-accent-ink" },
  dislike: { label: "Not for me", style: "bg-panel-2 text-muted" },
  unseen: { label: "Haven't seen", style: "bg-panel-2 text-muted" },
  not_this_vibe: { label: "Not this vibe", style: "bg-panel-2 text-muted" },
};

const when = (iso: string) => new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });

// "My history": every search with what came back, and every film you reacted to.
export default function HistoryPage() {
  const [h, setH] = useState<History | null>(null);
  const [tab, setTab] = useState<"searches" | "reactions">("searches");
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/api/history", { cache: "no-store" }).then(async (res) => {
      const body = await res.json();
      if (!res.ok) setError(body.detail ?? "Could not load your history.");
      else setH(body);
    });
  }, []);

  if (error) return <p className="py-20 text-center text-bad">{error}</p>;
  if (!h) return <Loading message="Loading your history…" />;
  return (
    <div className="flex flex-col gap-6">
      <h1 className="text-3xl font-semibold tracking-tight">My history</h1>
      <div className="flex gap-2">
        {(["searches", "reactions"] as const).map((t) => (
          <button key={t} onClick={() => setTab(t)} className={`rounded-full px-4 py-1.5 text-sm capitalize ${tab === t ? "bg-panel-2 text-ink" : "text-muted hover:text-ink"}`}>
            {t} ({t === "searches" ? h.searches.length : h.reactions.length})
          </button>
        ))}
      </div>
      {tab === "searches" &&
        (h.searches.length ? (
          <div className="flex flex-col gap-4">
            {h.searches.map((s) => (
              <div key={s.search_id} className="rounded-2xl border border-line bg-panel p-4">
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <Link href={`/search?q=${encodeURIComponent(s.prompt)}`} className="font-medium hover:text-accent">&ldquo;{s.prompt}&rdquo;</Link>
                  <span className="text-xs text-muted">{when(s.created)}</span>
                </div>
                <div className="mt-3 grid grid-cols-5 gap-2 sm:w-1/2">
                  {s.results.map((r) => (
                    <Poster key={r.movie_id} src={r.poster} title={r.title} />
                  ))}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-muted">No searches yet. <Link href="/search" className="text-accent">Try one</Link>.</p>
        ))}
      {tab === "reactions" &&
        (h.reactions.length ? (
          <div className="grid grid-cols-3 gap-3 sm:grid-cols-5 lg:grid-cols-8">
            {h.reactions.map((r, i) => (
              <div key={`${r.movie_id}-${i}`} className="flex flex-col gap-1">
                <Poster src={r.poster} title={r.title} />
                <span className={`self-start rounded-full px-2 py-0.5 text-[11px] ${VERDICT[r.verdict].style}`}>{VERDICT[r.verdict].label}</span>
                <span className="line-clamp-2 text-xs text-muted">{r.title}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-muted">No reactions yet. <Link href="/quiz" className="text-accent">Take the taste quiz</Link>.</p>
        ))}
    </div>
  );
}
