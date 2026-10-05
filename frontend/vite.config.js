import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import { fileURLToPath, URL } from 'node:url'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  server: {
    // Flask backend (app.py, port 5000): chat jobs + SSE progress stream.
    proxy: { '/api': { target: 'http://127.0.0.1:5000', changeOrigin: true } },
  },
})
