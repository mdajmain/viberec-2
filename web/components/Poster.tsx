/* eslint-disable @next/next/no-img-element */
// Posters come straight from TMDB's image server (plain <img>, lazy-loaded). Films without one get a calm placeholder.
export default function Poster({ src, title, className = "" }: { src: string | null; title: string; className?: string }) {
  if (!src) {
    return (
      <div className={`flex aspect-[2/3] items-center justify-center rounded-xl bg-gradient-to-br from-panel-2 to-panel p-3 text-center text-xs text-muted ${className}`}>
        {title}
      </div>
    );
  }
  return <img src={src} alt={`${title} poster`} loading="lazy" className={`aspect-[2/3] w-full rounded-xl object-cover shadow-lg shadow-black/40 ${className}`} />;
}
