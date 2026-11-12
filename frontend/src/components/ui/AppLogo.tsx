'use client';

export function AppLogo({ size = 44, title = 'ExAI-PanC' }: { size?: number; title?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 44 44"
      role="img"
      aria-label={title}
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <rect x="2" y="2" width="40" height="40" rx="12" fill="#0d9488" />
      <path
        d="M14 14 L30 30 M30 14 L14 30"
        stroke="#f8fafc"
        strokeWidth="3.4"
        strokeLinecap="round"
        strokeOpacity="0.9"
      />
      <circle cx="22" cy="22" r="3.2" fill="#3b82f6" />
    </svg>
  );
}
