"use client";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import Loading from "@/components/Loading";
import MovieCard from "@/components/MovieCard";
import type { Feed } from "@/lib/types";

// "For you": the tested Step 8 method from this person's likes, each pick with a reason.
export default function ForYou() {
  const [feed, setFeed] = useState<Feed | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    const res = await fetch("/api/feed?k=10", { cache: "no-store" });
    const body = await res.json();
    if (!res.ok) setError(body.detail ?? "Could not load your picks.");
    else setFeed(body);
    setLoading(false);
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight">Picked for you</h1>
          <p className="mt-1 text-sm text-muted">From the films you liked. Every like or pass below sharpens the next list.</p>
        </div>
        <button onClick={load} disabled={loading} className="rounded-full border border-line px-4 py-1.5 text-sm text-muted hover:border-accent/60 hover:text-ink">
          Refresh picks
        </button>
      </div>
      {feed && !feed.personal && !loading && (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-accent/40 bg-accent/10 p-4 text-sm">
          <span>These are popular films for now. Take the taste quiz to make them yours.</span>
          <Link href="/quiz" className="rounded-full bg-accent px-4 py-1.5 font-semibold text-accent-ink">Take the quiz</Link>
        </div>
      )}
      {loading && <Loading message="Choosing films for you and writing why each one fits…" />}
      {error && <p className="py-10 text-center text-bad">{error}</p>}
      {!loading && feed && (
        <div className="grid gap-4 lg:grid-cols-2">
          {feed.results.map((m, i) => (
            <MovieCard key={m.movie_id} movie={m} source="feed" contextId={feed.feed_id} rank={i + 1} />
          ))}
        </div>
      )}
    </div>
  );
}
