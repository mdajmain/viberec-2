import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // posters are shown with plain <img> straight from TMDB's image server, so no image optimisation quota is used
  poweredByHeader: false,
};

export default nextConfig;
