"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { SignedIn, SignedOut, SignInButton, UserButton } from "@clerk/nextjs";

const LINKS: [string, string][] = [
  ["/search", "Search"],
  ["/quiz", "Taste quiz"],
  ["/for-you", "For you"],
  ["/taste", "Your taste"],
  ["/history", "History"],
];

export default function Nav() {
  const path = usePathname();
  const link = (href: string, label: string) => (
    <Link
      key={href}
      href={href}
      className={`whitespace-nowrap rounded-full px-3 py-1.5 text-sm transition ${
        path.startsWith(href) ? "bg-panel-2 text-ink" : "text-muted hover:text-ink"
      }`}
    >
      {label}
    </Link>
  );
  return (
    <header className="sticky top-0 z-30 border-b border-line/70 bg-bg/80 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-6xl items-center gap-4 px-4 sm:px-6">
        <Link href="/" className="text-lg font-semibold tracking-tight">
          Vibe<span className="text-accent">Rec</span>
        </Link>
        <nav className="hidden gap-1 md:flex">{LINKS.map(([h, l]) => link(h, l))}</nav>
        <div className="ml-auto flex items-center gap-3">
          <SignedOut>
            <span className="hidden text-xs text-muted sm:inline">Browsing as a guest</span>
            <SignInButton mode="modal">
              <button className="rounded-full bg-accent px-4 py-1.5 text-sm font-semibold text-accent-ink hover:brightness-110">Sign in</button>
            </SignInButton>
          </SignedOut>
          <SignedIn>
            <Link href="/account" className="text-sm text-muted hover:text-ink">Account</Link>
            <UserButton />
          </SignedIn>
        </div>
      </div>
      <nav className="flex gap-1 overflow-x-auto px-4 pb-2 md:hidden">{LINKS.map(([h, l]) => link(h, l))}</nav>
    </header>
  );
}
