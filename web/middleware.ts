import { clerkMiddleware } from "@clerk/nextjs/server";

// Clerk runs on every page and API call. The guest cookie (a random id, no IP or fingerprint) is set by the API routes
// on a visitor's first request (lib/backend.ts currentProfile), so it works the same with or without Clerk keys.
export default clerkMiddleware();

export const config = {
  matcher: [
    "/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)",
    "/(api|trpc)(.*)",
  ],
};
