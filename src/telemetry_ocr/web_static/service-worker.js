"use strict";

const CACHE_NAME = "telemetry-ocr-pwa-v16";
const APP_SHELL = [
  "/",
  "/manifest.webmanifest",
  "/assets/app.css",
  "/assets/app.js",
  "/assets/detection-math.js",
  "/assets/icons/icon-192.png",
  "/assets/icons/icon-512.png",
  "/assets/icons/icon-maskable-512.png",
];
// Served network-first so a running page never lags behind the server build.
const CODE_PATHS = new Set([
  "/assets/app.js",
  "/assets/detection-math.js",
  "/assets/app.css",
]);

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL)));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  const url = new URL(request.url);
  if (request.method !== "GET" || url.origin !== self.location.origin || url.pathname.startsWith("/api/")) return;

  if (request.mode === "navigate") {
    event.respondWith(networkFirst(request, "/"));
    return;
  }

  // The application code is served network-first, the same as the page itself.
  // Cache-first was wrong for it: the cache only turns over when CACHE_NAME
  // changes, so a shipped fix silently kept running the previous build until
  // someone remembered to bump the constant and reload twice. This app cannot
  // do anything useful offline anyway -- recognition needs the local server --
  // so the cached copy is a fallback for showing the shell, not the source of
  // truth for behaviour.
  if (CODE_PATHS.has(url.pathname)) {
    event.respondWith(networkFirst(request, request));
    return;
  }

  // Icons and the manifest are content-addressed by name and effectively
  // immutable, so they stay cache-first.
  event.respondWith(
    caches.match(request).then((cached) => cached || fetchAndCache(request)),
  );
});

function networkFirst(request, cacheKey) {
  return fetch(request)
    .then((response) => {
      if (response.ok) {
        const copy = response.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(cacheKey, copy));
      }
      return response;
    })
    .catch(() => caches.match(cacheKey));
}

function fetchAndCache(request) {
  return fetch(request).then((response) => {
    if (response.ok) {
      const copy = response.clone();
      caches.open(CACHE_NAME).then((cache) => cache.put(request, copy));
    }
    return response;
  });
}
