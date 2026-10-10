export const metadata = { title: "Privacy — VibeRec" };

const ROWS: [string, string][] = [
  ["Who you are", "Guests: a random id in a cookie in this browser (no IP address, no fingerprint). Members: the account id from our sign-in provider, Clerk. Your email and password stay with Clerk."],
  ["What you react to", "Every like, pass, \"haven't seen\" and \"not this vibe\", with the page it happened on."],
  ["What you search", "Each prompt, what VibeRec understood from it, and the films it showed you."],
  ["Your taste profile", "A summary of your likes as numbers, used to pick films for you, plus the plain-words version on Your taste."],
];

export default function Privacy() {
  return (
    <div className="mx-auto flex max-w-2xl flex-col gap-6">
      <h1 className="text-3xl font-semibold tracking-tight">What VibeRec stores</h1>
      <p className="text-muted">VibeRec saves what you do here so it can learn your taste and improve recommendations. Nothing is sold or shared.</p>
      <dl className="flex flex-col gap-3">
        {ROWS.map(([k, v]) => (
          <div key={k} className="rounded-2xl border border-line bg-panel p-4">
            <dt className="font-medium">{k}</dt>
            <dd className="mt-1 text-sm text-muted">{v}</dd>
          </div>
        ))}
      </dl>
      <p className="text-sm text-muted">
        &ldquo;Forget me&rdquo; (on the Account page) cuts the link to you: your history moves to a random id that points to no one,
        your taste profile is deleted, and a member&apos;s account is removed. The anonymous reactions remain to help recommendations.
        Searches use Claude (Anthropic, via AWS) to read your prompt and write the explanations.
      </p>
    </div>
  );
}
