"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import Loading from "@/components/Loading";
import type { Taste } from "@/lib/types";

const count = (n: number, one: string, many = one + "s") => `${n} ${n === 1 ? one : many}`;

// "Your taste": the readable side of the taste profile, recomputed after every like or pass.
export default function TastePage() {
  const [taste, setTaste] = useState<Taste | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/api/taste", { cache: "no-store" }).then(async (res) => {
      const body = await res.json();
      if (!res.ok) setError(body.detail ?? "Could not load your taste.");
      else setTaste(body);
    });
  }, []);

  if (error) return <p className="py-20 text-center text-bad">{error}</p>;
  if (!taste) return <Loading message="Reading your taste…" />;
  if (!taste.likes)
    return (
      <div className="mx-auto flex max-w-md flex-col items-center gap-4 py-16 text-center">
        <h1 className="text-3xl font-semibold">No taste yet</h1>
        <p className="text-muted">Like a few films and VibeRec will describe the feel you lean toward.</p>
        <Link href="/quiz" className="rounded-xl bg-accent px-5 py-3 font-semibold text-accent-ink">Take the taste quiz</Link>
      </div>
    );

  const facts: [string, string | null][] = [
    ["Light or heavy", taste.light_heavy],
    ["Pace", taste.pacing],
    ["Era you like most", taste.era],
  ];
  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-6">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">Your taste</h1>
        <p className="mt-1 text-sm text-muted">
          From {count(taste.likes, "like")}, {count(taste.dislikes, "pass", "passes")} and {count(taste.unseen, "film")} you haven&apos;t seen.
        </p>
      </div>
      <section className="rounded-2xl border border-line bg-panel p-6">
        <h2 className="text-xs uppercase tracking-wide text-muted">The feel you lean toward</h2>
        <div className="mt-3 flex flex-wrap gap-2">
          {taste.top_vibes.map((v) => (
            <span key={v} className="rounded-full bg-accent/15 px-4 py-1.5 text-base font-medium text-accent">{v}</span>
          ))}
        </div>
      </section>
      <section className="grid gap-4 sm:grid-cols-3">
        {facts.map(([k, v]) => (
          <div key={k} className="rounded-2xl border border-line bg-panel p-5">
            <p className="text-xs uppercase tracking-wide text-muted">{k}</p>
            <p className="mt-2 text-lg font-semibold capitalize">{v ?? "—"}</p>
          </div>
        ))}
      </section>
      {taste.top_genres.length > 0 && (
        <section className="rounded-2xl border border-line bg-panel p-5">
          <p className="text-xs uppercase tracking-wide text-muted">Genres you like most</p>
          <p className="mt-2 text-lg font-semibold">{taste.top_genres.join(" · ")}</p>
        </section>
      )}
      <div className="flex flex-wrap gap-3">
        <Link href="/for-you" className="rounded-xl bg-accent px-5 py-3 font-semibold text-accent-ink">See my picks</Link>
        <Link href="/quiz" className="rounded-xl border border-line px-5 py-3 text-muted hover:text-ink">Answer more of the quiz</Link>
      </div>
    </div>
  );
}
