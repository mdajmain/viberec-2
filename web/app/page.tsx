import Link from "next/link";
import SearchBox from "@/components/SearchBox";

const STEPS = [
  { href: "/quiz", title: "Take the taste quiz", body: "Like or pass on films you know until you've liked 20. About two minutes.", cta: "Start the quiz" },
  { href: "/for-you", title: "Get picks for you", body: "Films chosen from your taste, each with a one-line reason it suits you.", cta: "See my picks" },
  { href: "/taste", title: "See your taste", body: "The feel you lean toward, the eras and genres you like, in plain words.", cta: "My taste" },
];

export default function Home() {
  return (
    <div className="flex flex-col gap-16">
      <section className="mx-auto flex max-w-3xl flex-col items-center gap-6 pt-10 text-center sm:pt-16">
        <p className="rounded-full border border-line bg-panel px-3 py-1 text-xs text-muted">Recommendations by how a film feels</p>
        <h1 className="text-balance text-4xl font-semibold leading-tight tracking-tight sm:text-6xl">
          Find films with the <span className="text-accent">vibe</span> you want.
        </h1>
        <p className="max-w-xl text-base text-muted sm:text-lg">
          Name a film you love and the feel you&apos;re after. VibeRec finds films that match, and tells you why each one fits.
        </p>
        <SearchBox big />
      </section>
      <section className="grid gap-4 sm:grid-cols-3">
        {STEPS.map((s, i) => (
          <Link key={s.href} href={s.href} className="group rounded-2xl border border-line bg-panel p-5 transition hover:border-accent/50">
            <p className="text-xs text-accent">0{i + 1}</p>
            <h2 className="mt-2 text-lg font-semibold">{s.title}</h2>
            <p className="mt-1 text-sm text-muted">{s.body}</p>
            <p className="mt-4 text-sm font-medium text-ink group-hover:text-accent">{s.cta} &rarr;</p>
          </Link>
        ))}
      </section>
    </div>
  );
}
