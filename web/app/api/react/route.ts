import { backend, currentProfile, respond } from "@/lib/backend";

export async function POST(req: Request) {
  return respond(async () => {
    const { movie_id, verdict, source, context_id } = await req.json();
    const { profile_id } = await currentProfile();
    return backend("/react", { method: "POST", body: { profile_id, movie_id, verdict, source, context_id: context_id ?? null } });
  });
}
