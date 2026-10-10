import Link from "next/link";

export default function Footer() {
  return (
    <footer className="border-t border-line/70">
      <div className="mx-auto flex max-w-6xl flex-col gap-3 px-4 py-6 text-xs text-muted sm:px-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <p>VibeRec recommends films by how they feel. Ratings: MovieLens. Film data and posters: TMDB.</p>
          <nav className="flex gap-4">
            <Link href="/privacy" className="whitespace-nowrap underline-offset-4 hover:text-ink hover:underline">Privacy</Link>
            <Link href="/account" className="whitespace-nowrap underline-offset-4 hover:text-ink hover:underline">Forget me</Link>
          </nav>
        </div>
        <p className="text-muted/70">This product uses the TMDB API but is not endorsed or certified by TMDB.</p>
      </div>
    </footer>
  );
}
