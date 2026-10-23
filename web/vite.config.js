import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Served from https://parbproject.github.io/vol-forecast-ml/
export default defineConfig({
  base: "/vol-forecast-ml/",
  plugins: [react()],
  build: { target: "es2022", outDir: "dist", emptyOutDir: true },
});
