import { backend, currentProfile, respond } from "@/lib/backend";

export async function POST(req: Request) {
  return respond(async () => {
    const { prompt } = await req.json();
    const { profile_id } = await currentProfile();
    return backend("/search", { method: "POST", body: { prompt, k: 5, profile_id } });
  });
}
