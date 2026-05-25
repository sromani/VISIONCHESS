"use client";

import { useEffect } from "react";

export function Providers({ children }: { children: React.ReactNode }) {
  useEffect(() => {
    const stored = localStorage.getItem("theme");
    document.documentElement.classList.toggle("dark", stored !== "light");
  }, []);

  return <>{children}</>;
}
