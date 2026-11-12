const nextConfig = {
  experimental: {
    optimizeCss: true,
    optimizePackageImports: ["@react-three/fiber", "@react-three/drei"],
  },
  images: {
    domains: ["localhost", "api.exaipanc.dev"],
  },
  async headers() {
    return [
      {
        source: "/api/v1/(.*)",
        headers: [
          {
            key: "X-Content-Type-Options",
            value: "nosniff",
          },
          {
            key: "X-Frame-Options",
            value: "DENY",
          },
          {
            key: "X-XSS-Protection",
            value: "1; mode=block",
          },
        ],
      },
    ];
  },
  async redirects() {
    return [
      {
        source: "/login",
        destination: "/auth/login",
        permanent: false,
      },
      {
        source: "/register",
        destination: "/auth/register",
        permanent: false,
      },
    ];
  },
};

export default nextConfig;