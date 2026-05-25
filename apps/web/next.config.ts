import type { NextConfig } from "next";

const apiBackend = process.env.API_BACKEND_URL ?? "http://127.0.0.1:8001";

const nextConfig: NextConfig = {
  output: "standalone",
  images: {
    remotePatterns: [
      { protocol: "http", hostname: "localhost", port: "8001" },
      { protocol: "http", hostname: "127.0.0.1", port: "8001" },
    ],
  },
  async rewrites() {
    return [
      {
        source: "/api/v1/:path*",
        destination: `${apiBackend}/api/v1/:path*`,
      },
    ];
  },
};

export default nextConfig;
