import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { ClerkProvider } from "@clerk/nextjs";
import Nav from "@/components/Nav";
import Footer from "@/components/Footer";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "VibeRec — movies by vibe",
  description: "Name a film you love and the feel you want. VibeRec finds films with that vibe, and tells you why.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <ClerkProvider
      appearance={{
        variables: {
          colorPrimary: "#f5a524",
          colorBackground: "#16141c",
          colorText: "#f4f1ea",
          colorTextSecondary: "#a8a29e",
          colorInputBackground: "#0c0b10",
          colorInputText: "#f4f1ea",
        },
      }}
    >
      <html lang="en" className={inter.variable}>
        <body className="flex min-h-dvh flex-col font-sans antialiased">
          <div className="page-glow pointer-events-none fixed inset-x-0 top-0 h-96" />
          <Nav />
          <main className="relative mx-auto w-full max-w-6xl flex-1 px-4 pb-20 pt-8 sm:px-6">{children}</main>
          <Footer />
        </body>
      </html>
    </ClerkProvider>
  );
}
