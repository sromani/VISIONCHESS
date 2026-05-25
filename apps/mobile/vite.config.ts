import path from "node:path";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
    /** Use external wasm files from public/ort (not JSEP bundle). */
    conditions: [
      "onnxruntime-web-use-extern-wasm",
      "import",
      "module",
      "browser",
      "default",
    ],
  },
  build: {
    outDir: "dist",
    target: "es2020",
  },
  optimizeDeps: {
    exclude: ["onnxruntime-web", "@techstark/opencv-js"],
  },
  worker: {
    format: "es",
  },
  server: {
    port: 5173,
    host: true,
  },
});
