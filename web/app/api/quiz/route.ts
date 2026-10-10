import { backend, currentProfile, respond } from "@/lib/backend";

export async function GET() {
  return respond(async () => {
    const { profile_id } = await currentProfile();
    return backend("/quiz/next", { query: { profile_id } });
  });
}
