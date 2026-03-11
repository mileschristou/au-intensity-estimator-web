/**
 * coi-serviceworker.js
 *
 * Adds Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy headers to
 * every response so the page becomes cross-origin isolated.  This enables
 * SharedArrayBuffer, which onnxruntime-web needs to load its threaded JSEP wasm
 * module and register the WebGPU execution provider.
 *
 * GitHub Pages cannot set custom HTTP headers, so this service-worker trick is
 * the standard workaround.  The page reloads once after the SW activates; on
 * that second load the SW intercepts the HTML document and injects the headers.
 */

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));

self.addEventListener('fetch', e => {
  // Ignore non-GET requests and opaque same-origin cache reads
  if (e.request.method !== 'GET') return;
  if (e.request.cache === 'only-if-cached' && e.request.mode !== 'same-origin') return;

  e.respondWith(
    fetch(e.request).then(resp => {
      if (!resp || resp.status === 0 || !resp.headers) return resp;

      const headers = new Headers(resp.headers);
      headers.set('Cross-Origin-Opener-Policy',   'same-origin');
      // 'credentialless' allows cross-origin fetches (HF, jsDelivr) without
      // requiring explicit Cross-Origin-Resource-Policy on each remote server.
      headers.set('Cross-Origin-Embedder-Policy',  'credentialless');

      return new Response(resp.body, {
        status:     resp.status,
        statusText: resp.statusText,
        headers,
      });
    }).catch(() => fetch(e.request))
  );
});
