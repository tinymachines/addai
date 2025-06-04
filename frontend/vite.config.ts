import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  root: 'www',
  resolve: {
    alias: {
      '/src': resolve(__dirname, 'src')
    }
  },
  server: {
    fs: {
      allow: ['..']
    },
    proxy: {
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
        changeOrigin: true
      }
    }
  },
  build: {
    target: 'esnext'
  },
  optimizeDeps: {
    exclude: ['@wasm-tool/wasm-pack-plugin']
  }
});