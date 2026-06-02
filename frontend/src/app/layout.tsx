import type { Metadata } from "next";
import "./globals.css";
import Link from "next/link";

export const metadata: Metadata = {
  title: "SimCity — AI Digital Twin of the Internet",
  description: "Real-time autonomous simulation engine for internet behavior, virality, influence cascades, and collective dynamics.",
};

const NAV_ITEMS = [
  { href: "/", label: "Dashboard", icon: "◉" },
  { href: "/graph", label: "Graph", icon: "◎" },
  { href: "/simulate", label: "Simulate", icon: "⚡" },
  { href: "/trends", label: "Trends", icon: "📈" },
  { href: "/narratives", label: "Narratives", icon: "📖" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="bg-[#0a0a0a] text-white font-sans antialiased">
        <div className="flex h-screen">
          {/* Sidebar Navigation */}
          <nav className="w-16 bg-[#0f0f0f] border-r border-gray-800 flex flex-col items-center py-4 gap-2">
            <div className="text-lg font-bold mb-4 text-blue-400">SC</div>
            {NAV_ITEMS.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="w-10 h-10 rounded-lg flex items-center justify-center text-sm hover:bg-gray-800 transition-colors group relative"
                title={item.label}
              >
                <span>{item.icon}</span>
                <span className="absolute left-14 bg-gray-800 text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none z-50">
                  {item.label}
                </span>
              </Link>
            ))}
          </nav>

          {/* Main Content */}
          <div className="flex-1 overflow-hidden">
            {children}
          </div>
        </div>
      </body>
    </html>
  );
}
