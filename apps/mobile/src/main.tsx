import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import AppRoot from "./AppRoot";
import { initNativeShell } from "./lib/native/initNativeShell";
import "./index.css";

void initNativeShell();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <AppRoot />
  </StrictMode>,
);
