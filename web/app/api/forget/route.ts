import { cookies } from "next/headers";
import { backend, currentProfile, newGuestId, respond } from "@/lib/backend";

// "Forget me": the back end moves this profile's history to a fresh anonymous id (the reactions stay for training,
// nothing points to the person). The browser then gets a brand-new guest id; a member is also signed out and their
// Clerk account deleted by the page.
export async function POST() {
  return respond(async () => {
    const { profile_id, kind } = await currentProfile();
    await backend("/forget", { method: "POST", body: { profile_id } });
    (await cookies()).set("vr_guest", newGuestId(), { httpOnly: true, sameSite: "lax", secure: process.env.NODE_ENV === "production", maxAge: 31536000, path: "/" });
    return { forgotten: true, was: kind };
  });
}
