import "server-only";
import { auth } from "@clerk/nextjs/server";
import { cookies } from "next/headers";
import { NextResponse } from "next/server";

// The VibeRec API (FastAPI on the server). Only this server-side code calls it, with the secret key; browsers never do.
const API = process.env.VIBEREC_API_URL ?? "http://localhost:8000";
const KEY = process.env.VIBEREC_API_KEY ?? "";

export class BackendError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export async function backend<T>(
  path: string,
  init: { method?: "GET" | "POST"; body?: unknown; query?: Record<string, string | number> } = {},
): Promise<T> {
  const url = new URL(path, API);
  for (const [k, v] of Object.entries(init.query ?? {})) url.searchParams.set(k, String(v));
  const res = await fetch(url, {
    method: init.method ?? "GET",
    headers: { "X-VibeRec-Key": KEY, "Content-Type": "application/json" },
    body: init.body === undefined ? undefined : JSON.stringify(init.body),
    cache: "no-store",
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {}
    throw new BackendError(res.status, String(detail));
  }
  return res.json() as Promise<T>;
}

export function newGuestId() {
  return "g_" + crypto.randomUUID().replaceAll("-", "");
}

/** The visitor's profile: the member profile when signed in (merging this browser's guest history), else the guest. */
export async function currentProfile(): Promise<{ profile_id: string; kind: "guest" | "member" }> {
  const { userId } = await auth();
  const jar = await cookies();
  let guest = jar.get("vr_guest")?.value;
  if (!guest) {
    guest = newGuestId();
    jar.set("vr_guest", guest, { httpOnly: true, sameSite: "lax", secure: process.env.NODE_ENV === "production", maxAge: 31536000, path: "/" });
  }
  return backend("/profile", { method: "POST", body: userId ? { clerk_user_id: userId, guest_id: guest } : { guest_id: guest } });
}

/** Run an API route body and turn back-end errors into JSON answers with the same status. */
export async function respond(fn: () => Promise<unknown>) {
  try {
    return NextResponse.json(await fn());
  } catch (err) {
    if (err instanceof BackendError) return NextResponse.json({ detail: err.message }, { status: err.status });
    console.error(err);
    return NextResponse.json({ detail: "Something went wrong. Please try again." }, { status: 500 });
  }
}
