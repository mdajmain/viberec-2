"use client";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { SignedIn, SignedOut, SignInButton, useClerk, useUser } from "@clerk/nextjs";

// Account and "Forget me". Forgetting anonymises: your history moves to a random id that points to no one
// (the reactions still help recommendations), your taste profile is deleted, and a member's account is removed.
export default function Account() {
  const { user } = useUser();
  const { signOut } = useClerk();
  const router = useRouter();
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState(false);

  async function forget() {
    setBusy(true);
    const res = await fetch("/api/forget", { method: "POST" });
    if (res.ok && user) {
      try {
        await user.delete();
      } catch {
        await signOut();
      }
    }
    setBusy(false);
    setDone(res.ok);
    setConfirm(false);
    router.refresh();
  }

  return (
    <div className="mx-auto flex max-w-2xl flex-col gap-6">
      <h1 className="text-3xl font-semibold tracking-tight">Account</h1>
      <SignedIn>
        <p className="text-muted">Signed in as <span className="text-ink">{user?.primaryEmailAddress?.emailAddress}</span>. Your taste and history follow you on every device.</p>
      </SignedIn>
      <SignedOut>
        <div className="rounded-2xl border border-line bg-panel p-5">
          <p>You&apos;re browsing as a guest. Your taste and history are kept for this browser only.</p>
          <p className="mt-1 text-sm text-muted">Sign in to keep them on every device. Anything you did as a guest moves into your account.</p>
          <SignInButton mode="modal">
            <button className="mt-4 rounded-xl bg-accent px-5 py-2.5 font-semibold text-accent-ink">Sign in or sign up</button>
          </SignInButton>
        </div>
      </SignedOut>
      <section className="rounded-2xl border border-bad/40 bg-panel p-5">
        <h2 className="text-lg font-semibold">Forget me</h2>
        <p className="mt-1 text-sm text-muted">
          Removes the link between you and everything you did here, and deletes your taste profile{user ? " and your account" : ""}.
          Your likes and passes stay only as anonymous data that helps improve recommendations; nothing can trace them back to you.{" "}
          <Link href="/privacy" className="underline underline-offset-4">What VibeRec stores</Link>
        </p>
        {done ? (
          <p className="mt-4 text-good">Done. You start fresh from here.</p>
        ) : confirm ? (
          <div className="mt-4 flex flex-wrap gap-2">
            <button onClick={forget} disabled={busy} className="rounded-xl bg-bad px-5 py-2.5 font-semibold text-bg">{busy ? "Forgetting…" : "Yes, forget me"}</button>
            <button onClick={() => setConfirm(false)} className="rounded-xl border border-line px-5 py-2.5 text-muted">Cancel</button>
          </div>
        ) : (
          <button onClick={() => setConfirm(true)} className="mt-4 rounded-xl border border-bad/60 px-5 py-2.5 text-bad hover:bg-bad/10">Forget me</button>
        )}
      </section>
    </div>
  );
}
