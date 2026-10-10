import { currentProfile, respond } from "@/lib/backend";

export async function GET() {
  return respond(async () => ({ kind: (await currentProfile()).kind }));
}
