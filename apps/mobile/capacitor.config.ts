import type { Config } from "@capacitor/cli";

const config: Config = {
  appId: "com.visionchess.app",
  appName: "VisionChess",
  webDir: "dist",
  bundledWebRuntime: false,
  server: {
    /** https://localhost — required for ORT wasm fetch on iOS/Android WKWebView */
    androidScheme: "https",
    iosScheme: "https",
  },
  plugins: {
    SplashScreen: {
      launchAutoHide: false,
      launchShowDuration: 0,
      backgroundColor: "#09090b",
      androidSplashResourceName: "splash",
      androidScaleType: "FIT_XY",
      showSpinner: false,
      splashFullScreen: true,
      splashImmersive: true,
    },
    StatusBar: {
      style: "DARK",
      backgroundColor: "#09090b",
    },
    Camera: {
      permissions: ["camera", "photos"],
    },
  },
};

export default config;
