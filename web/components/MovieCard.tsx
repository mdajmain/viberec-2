import type { Movie } from "@/lib/types";
import Poster from "./Poster";
import ReactionBar from "./ReactionBar";

// One recommended film: poster, title, the "why", the premise on demand, and reaction buttons.
export default function MovieCard({ movie, source, contextId, rank }: { movie: Movie; source: "search" | "feed"; contextId?: string; rank?: number }) {
  return (
    <article className="group flex gap-4 rounded-2xl border border-line bg-panel p-3 transition hover:border-accent/40 sm:gap-5 sm:p-4">
      <div className="relative w-24 shrink-0 sm:w-32">
        <Poster src={movie.poster} title={movie.title} />
        {rank !== undefined && (
          <span className="absolute -left-2 -top-2 flex h-7 w-7 items-center justify-center rounded-full bg-accent text-xs font-bold text-accent-ink">
            {rank}
          </span>
        )}
      </div>
      <div className="flex min-w-0 flex-1 flex-col gap-2">
        <div>
          <h3 className="text-base font-semibold leading-tight sm:text-lg">{movie.title.replace(/ \(\d{4}\)$/, "")}</h3>
          <p className="text-xs text-muted">
            {[movie.year, movie.genres.slice(0, 3).join(" · ")].filter(Boolean).join(" · ")}
          </p>
        </div>
        {movie.why && <p className="text-sm leading-relaxed text-ink/90">{movie.why}</p>}
        {movie.premise && (
          <details className="text-sm text-muted">
            <summary className="cursor-pointer select-none text-xs uppercase tracking-wide hover:text-ink">The setup</summary>
            <p className="mt-1 leading-relaxed">{movie.premise}</p>
          </details>
        )}
        <div className="mt-auto pt-1">
          <ReactionBar movieId={movie.movie_id} source={source} contextId={contextId} />
        </div>
      </div>
    </article>
  );
}
