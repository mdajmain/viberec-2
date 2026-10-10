"use client";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import Loading from "@/components/Loading";
import Poster from "@/components/Poster";
import type { QuizState, Verdict } from "@/lib/types";

const ANSWERS: { verdict: Verdict; label: string; key: string; style: string }[] = [
  { verdict: "dislike", label: "Not for me", key: "N", style: "border-line bg-panel-2 hover:border-bad/60" },
  { verdict: "unseen", label: "Haven't seen", key: "U", style: "border-line bg-panel-2 hover:border-ink/40" },
  { verdict: "like", label: "Like", key: "L", style: "border-accent bg-accent text-accent-ink hover:brightness-110" },
];

// The taste quiz (Step 8): one film at a time, in the tested order, until 20 likes (at most 100 questions).
export default function Quiz() {
  const [state, setState] = useState<QuizState | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    const res = await fetch("/api/quiz", { cache: "no-store" });
    const body = await res.json();
    if (!res.ok) setError(body.detail ?? "Could not load the quiz.");
    else setState(body);
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const answer = useCallback(
    async (verdict: Verdict) => {
      if (!state?.movie || busy) return;
      setBusy(true);
      await fetch("/api/react", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ movie_id: state.movie.movie_id, verdict, source: "quiz" }),
      });
      await load();
      setBusy(false);
    },
    [state, busy, load],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const a = ANSWERS.find((x) => x.key === e.key.toUpperCase());
      if (a) answer(a.verdict);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [answer]);

  if (error) return <p className="py-20 text-center text-bad">{error}</p>;
  if (!state) return <Loading message="Loading the quiz…" />;

  const pct = Math.min(100, Math.round((state.likes / state.target_likes) * 100));
  return (
    <div className="mx-auto flex max-w-md flex-col gap-6">
      <div>
        <div className="flex items-baseline justify-between text-sm">
          <h1 className="text-xl font-semibold">Taste quiz</h1>
          <span className="text-muted">
            <span className="font-semibold text-ink">{state.likes}</span> of {state.target_likes} likes · {state.answered} answered
          </span>
        </div>
        <div className="mt-3 h-2 overflow-hidden rounded-full bg-panel-2">
          <div className="h-full rounded-full bg-accent transition-all" style={{ width: `${pct}%` }} />
        </div>
      </div>

      {state.done || !state.movie ? (
        <div className="flex flex-col items-center gap-4 rounded-2xl border border-line bg-panel p-8 text-center">
          <h2 className="text-2xl font-semibold">Your taste is set</h2>
          <p className="text-muted">You liked {state.likes} films. Your picks are ready.</p>
          <Link href="/for-you" className="rounded-xl bg-accent px-5 py-3 font-semibold text-accent-ink hover:brightness-110">See my picks</Link>
        </div>
      ) : (
        <>
          <div className="relative mx-auto w-64 sm:w-72">
            <Poster src={state.movie.poster} title={state.movie.title} className={busy ? "opacity-60" : ""} />
          </div>
          <div className="text-center">
            <h2 className="text-xl font-semibold">{state.movie.title.replace(/ \(\d{4}\)$/, "")}</h2>
            <p className="text-sm text-muted">
              {[state.movie.year, state.movie.genres.slice(0, 3).join(" · ")].filter(Boolean).join(" · ")}
            </p>
          </div>
          <div className="grid grid-cols-3 gap-2">
            {ANSWERS.map((a) => (
              <button key={a.verdict} disabled={busy} onClick={() => answer(a.verdict)} className={`rounded-xl border px-3 py-3 text-sm font-semibold transition ${a.style}`}>
                {a.label}
                <span className="ml-1 hidden text-xs font-normal opacity-60 sm:inline">({a.key})</span>
              </button>
            ))}
          </div>
          {state.likes >= 5 && (
            <Link href="/for-you" className="text-center text-sm text-muted underline-offset-4 hover:text-ink hover:underline">
              I&apos;m done, show my picks
            </Link>
          )}
        </>
      )}
    </div>
  );
}
