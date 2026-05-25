/**
 * ONNX Runtime Web on iOS WKWebView / Capacitor must use the **non-JSEP** wasm build.
 * Default ORT 1.22 picks JSEP and dynamic-imports ort-wasm-simd-threaded.jsep.mjs → fails on device.
 */
import type * as Ort from "onnxruntime-web";

import { workerAssetUrl } from "./workerAssets";

let configured = false;

export function configureOrtWasmEnv(ort: typeof Ort): void {
  if (configured) return;
  configured = true;

  ort.env.wasm.wasmPaths = {
    wasm: workerAssetUrl("ort/ort-wasm-simd-threaded.wasm"),
    mjs: workerAssetUrl("ort/ort-wasm-simd-threaded.mjs"),
  };
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
}
