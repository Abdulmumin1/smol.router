import { defineConfig } from "astro/config";

export default defineConfig({
  site: "https://router.ai-query.dev",
  output: "static",
  build: {
    format: "directory"
  },
  vite: {
    build: {
      cssMinify: true
    }
  }
});
