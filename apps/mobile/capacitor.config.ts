import type { Config } from "@capacitor/cli";

const config: Config = {
  appId: "com.visionchess.app",
  appName: "VisionChess",
  webDir: "dist",
  bundledWebRuntime: false,
  server: {
    androidScheme: "https",
  },
  plugins: {
    SplashScreen: {
      launchAutoHide: true,
      launchShowDuration: 2000,
      backgroundColor: "#09090b",
      androidSplashResourceName: "splash",
      androidScaleType: "CENTER_CROP",
      showSpinner: false,
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
