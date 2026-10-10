"use client";
import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import Chips from "@/components/Chips";
import Loading from "@/components/Loading";
import MovieCard from "@/components/MovieCard";
import SearchBox from "@/components/SearchBox";
import type { SearchResult } from "@/lib/types";

export default function SearchPage() {
  const params = useSearchParams();
  const router = useRouter();
  const q = params.get("q") ?? "";
  const [result, setResult] = useState<SearchResult | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const run = useCallback(async (prompt: string) => {
    setLoading(true);
    setError("");
    setResult(null);
    const res = await fetch("/api/search", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt }) });
    const body = await res.json();
    if (!res.ok) setError(body.detail ?? "Search failed.");
    else setResult(body);
    setLoading(false);
  }, []);

  useEffect(() => {
    if (q) run(q);
  }, [q, run]);

  const u = result?.understood;
  return (
    <div className="flex flex-col gap-6">
      <h1 className="text-3xl font-semibold tracking-tight">Search by vibe</h1>
      <SearchBox key={q} initial={q} onSearch={(t) => router.push(`/search?q=${encodeURIComponent(t)}`)} />
      {loading && <Loading message="Reading your prompt, finding films with that vibe, and writing why each fits…" />}
      {error && <p className="py-10 text-center text-bad">{error}</p>}
      {u && (
        <div className="flex flex-wrap gap-x-5 gap-y-2 rounded-2xl border border-line bg-panel p-3">
          <span className="text-xs text-muted">Understood as</span>
          <Chips label="Like" items={u.sample_movies} />
          <Chips label="Genre" items={u.genres} />
          <Chips label="More" items={u.more_of} />
          <Chips label="Less" items={u.less_of} />
          <Chips label="About" items={u.about} />
          {!u.sample_movies.length && <span className="text-xs text-muted">No sample film found, so this searched by feel alone.</span>}
        </div>
      )}
      {result && (
        <div className="grid gap-4 lg:grid-cols-2">
          {result.results.map((m, i) => (
            <MovieCard key={m.movie_id} movie={m} source="search" contextId={result.search_id} rank={i + 1} />
          ))}
        </div>
      )}
    </div>
  );
}
