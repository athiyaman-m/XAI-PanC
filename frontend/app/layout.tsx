import type { Metadata, Viewport } from 'next';
import '../src/styles/globals.css';

export const metadata: Metadata = {
  title: 'ExAI-PanC | Clinical Decision Support System',
  description: 'AI-powered Clinical Digital Twin for Pancreatic Cancer. Research use only.',
  keywords: ['pancreatic cancer', 'digital twin', 'clinical decision support', 'AI', 'PDAC'],
  authors: [{ name: 'ExAI-PanC' }],
  robots: 'noindex, nofollow',
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1,
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased">{children}</body>
    </html>
  );
}
