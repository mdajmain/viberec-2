// Small labelled chips, e.g. "More: funny · romantic". Vibe keys arrive as snake_case and are shown as words.
export function words(key: string) {
  return key.replaceAll("_", " ").replace("dark bleak", "dark").replace("deadpan witty", "witty");
}

export default function Chips({ label, items }: { label: string; items: string[] }) {
  if (!items.length) return null;
  return (
    <span className="inline-flex flex-wrap items-center gap-1.5">
      <span className="text-xs uppercase tracking-wide text-muted">{label}</span>
      {items.map((x) => (
        <span key={x} className="rounded-full border border-line bg-panel px-2.5 py-0.5 text-xs text-ink">
          {words(x)}
        </span>
      ))}
    </span>
  );
}
