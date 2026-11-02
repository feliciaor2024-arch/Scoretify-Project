import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
export default defineConfig({
  plugins: [react()],
  server: {
    watch: { ignored: ["**/test-results/**", "**/playwright-report/**"] },
    // The back end (backend/app.py), so the page and the API share one origin.
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
  test: {
    include: ["src/**/*.test.{ts,tsx}"],
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
  },
});
