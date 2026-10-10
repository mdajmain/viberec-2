import { backend, currentProfile, respond } from "@/lib/backend";

export async function GET(req: Request) {
  return respond(async () => {
    const k = Number(new URL(req.url).searchParams.get("k") ?? 10);
    const { profile_id } = await currentProfile();
    return backend("/feed", { query: { profile_id, k } });
  });
}
