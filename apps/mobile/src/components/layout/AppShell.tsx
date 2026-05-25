import { WEB_DEBUG } from "@/shared/config/flags";

import { MobileHeader } from "@/components/layout/MobileHeader";
import { ProductHeader } from "@/components/layout/ProductHeader";
import { Sidebar } from "@/components/layout/Sidebar";

export function AppShell({ children }: { children: React.ReactNode }) {
  if (WEB_DEBUG) {
    return (
      <div className="flex min-h-screen bg-background">
        <div className="hidden md:flex">
          <Sidebar />
        </div>
        <div className="flex min-h-screen flex-1 flex-col">
          <MobileHeader />
          <main className="flex flex-1 flex-col">{children}</main>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen flex-col bg-background">
      <ProductHeader />
      <main className="flex flex-1 flex-col">{children}</main>
    </div>
  );
}
