import { backend, currentProfile, respond } from "@/lib/backend";

export async function GET() {
  return respond(async () => {
    const { profile_id } = await currentProfile();
    return backend("/history", { query: { profile_id } });
  });
}
