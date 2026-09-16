"use strict";

/*
 * The seams between the three files that make the page.
 *
 * app.js reaches into the markup by id and builds nodes by class name, and
 * neither the HTML nor the CSS knows it. A redesign that renames a wrapper or
 * styles a child that does not exist breaks the page in a way nothing catches:
 * the module still parses, the server still serves it, and the strip renders as
 * empty outlines. These are the two seams that have actually broken.
 */

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const STATIC_DIR = path.join(__dirname, "..", "src", "telemetry_ocr", "web_static");
const read = (name) => fs.readFileSync(path.join(STATIC_DIR, name), "utf8");

const appJs = read("app.js");
const indexHtml = read("index.html");
const appCss = read("app.css");

const matchAll = (source, pattern) => [...source.matchAll(pattern)].map((m) => m[1]);

test("every element app.js binds by id exists in the markup", () => {
  // `const ui = { video: $("sourceVideo"), ... }` runs at module scope, so an id
  // that is not in the document is not a missing feature -- it is a null that
  // throws on the first addEventListener and takes the whole page with it.
  const wanted = new Set(matchAll(appJs, /\$\("([^"]+)"\)/g));
  const present = new Set(matchAll(indexHtml, /id="([^"]+)"/g));
  const missing = [...wanted].filter((id) => !present.has(id)).sort();
  assert.deepEqual(missing, [], `app.js binds ids the markup does not define: ${missing}`);
  assert.ok(wanted.size > 50, "the id sweep found suspiciously few bindings");
});

test("every class app.js creates is styled", () => {
  // Nodes built in script carry classes that appear nowhere in the HTML, so a
  // stylesheet rewrite has nothing to grep for and drops them silently.
  const created = new Set(
    matchAll(appJs, /className\s*=\s*"([^"]+)"/g).flatMap((value) => value.split(/\s+/))
  );
  const unstyled = [...created]
    .filter((name) => name && !name.startsWith("is-"))
    .filter((name) => !appCss.includes(`.${name}`))
    .sort();
  assert.deepEqual(unstyled, [], `classes app.js creates with no rule: ${unstyled}`);
});

test("the capture thumbnail is sized by the button, not by a child image", () => {
  /*
   * app.js paints the thumbnail with `open.style.backgroundImage`. A background
   * gives the element no intrinsic size, so the button has to be stretched over
   * a card that has a height of its own. Styling `.shot-thumb-open img` instead
   * -- there is no img -- collapsed the button to zero and the strip rendered as
   * three empty outlines with a checkbox floating in each.
   */
  assert.match(
    appJs,
    /open\.style\.backgroundImage/,
    "app.js no longer paints the thumbnail as a background; this test is stale"
  );

  const rule = (selector) => {
    const match = appCss.match(
      new RegExp(`\\${selector}\\s*\\{([^}]*)\\}`)
    );
    assert.ok(match, `no rule for ${selector}`);
    return match[1];
  };

  assert.match(rule(".shot-thumb"), /height:\s*\d/, ".shot-thumb needs a height of its own");
  const open = rule(".shot-thumb-open");
  assert.match(open, /position:\s*absolute/, ".shot-thumb-open must fill the card");
  assert.match(open, /height:\s*100%/, ".shot-thumb-open must fill the card");
  assert.match(open, /cover/, "the background has to be sized, or it tiles at its natural size");
  assert.doesNotMatch(
    appCss,
    /\.shot-thumb-open\s+img\b/,
    "there is no <img> inside the thumbnail button"
  );
});

test("hidden wins over any display a class also sets", () => {
  /*
   * The enlarged-shot overlay, the install button and the
   * Restart worker button are all hidden with the attribute and shown by script.
   * Every one of them also carries a class, and a `display:` on that class beats
   * the user-agent's `[hidden]` rule -- which is how a button that should appear
   * only when the detection worker has died ended up on screen permanently.
   */
  assert.match(
    appCss,
    /\[hidden\]\s*\{\s*display:\s*none\s*!important/,
    "app.css needs a global [hidden] rule with !important"
  );
  const guardAt = appCss.indexOf("[hidden]");
  const firstDisplay = appCss.search(/^\s*(?:\.|#)[^{}]*\{[^}]*display:/m);
  assert.ok(
    guardAt < firstDisplay || firstDisplay === -1,
    "the [hidden] guard must come before the rules it has to beat"
  );
});

test("the downloads export the same thing the buttons are enabled by", () => {
  /*
   * The two buttons are enabled from `state.shots.length` and used to serialise
   * `state.records`, which only the stream's telemetry poll ever filled. In file
   * mode -- load a video, hold Capture, press Download -- the JSONL was a single
   * newline and the CSV was a header row, with no error and no warning.
   *
   * `state.records` is gone entirely now: nothing read it, and it grew to
   * 200,000 entries on a long stream to be read by nobody.
   */
  // Comment lines are stripped first: the ones explaining why it went name it.
  const code = appJs
    .split("\n")
    .filter((line) => !line.trim().startsWith("//"))
    .join("\n");
  assert.doesNotMatch(
    code,
    /\bstate\.records\b/,
    "state.records is referenced again outside a comment"
  );

  const exportRows = appJs.match(/function exportRows\(\)\s*\{[\s\S]*?\n\}/);
  assert.ok(exportRows, "exportRows() is missing");
  assert.match(exportRows[0], /state\.shots/, "the export must read the captured shots");

  for (const name of ["downloadJsonl", "downloadCsv"]) {
    const fn = appJs.match(new RegExp(`function ${name}\\(\\)\\s*\\{[\\s\\S]*?\\n\\}`));
    assert.ok(fn, `${name}() is missing`);
    assert.match(fn[0], /exportRows\(\)/, `${name}() must serialise exportRows()`);
    // An empty export says so rather than handing over an empty file.
    assert.match(fn[0], /rows\.length/, `${name}() must refuse an empty export`);
  }
});

test("the page has no stream mode left", () => {
  // Removed from the web app by decision: the page plays a file. The CLI still
  // reads rtsp:// through `telemetry-ocr run` and `capture`, which is a separate
  // process and not this page.
  for (const needle of ["/api/stream", "streamMode", "streamImage", "startStream"]) {
    assert.ok(!appJs.includes(needle), `app.js still refers to ${needle}`);
  }
  for (const needle of ["streamUrl", "streamButton", "streamBadge", "rtsp://"]) {
    assert.ok(!indexHtml.includes(needle), `index.html still carries ${needle}`);
  }
});

test("a model is named from the registry, never sent as a file", () => {
  /*
   * A .pt is a pickle that runs code as it loads. The page offers only the
   * checkpoints the server registered, and adds one by the id of a file the
   * server itself found -- never by uploading bytes or naming a path.
   */
  assert.match(indexHtml, /<select[^>]*id="modelSelect"/, "the model dropdown is missing");
  assert.doesNotMatch(indexHtml, /type="file"/, "a file picker is back");
  assert.doesNotMatch(appJs, /new FormData\(/, "app.js builds a multipart upload again");
  assert.match(appJs, /\/api\/models\/candidates/, "models are no longer added from discovered files");
  assert.match(appJs, /candidate_id: candidate\.id/, "a model is added by something other than its id");
  assert.match(appJs, /JSON\.stringify\(\{ name \}\)/, "the default is not chosen by name");
});

test("the page plays what the model rendered and draws no boxes itself", () => {
  /*
   * The live overlay painted boxes a frame or three late over a video that had
   * moved on, from a tracker fed an uneven fraction of the frames. Renders run
   * every frame through the model once and bake the boxes in, so the page must
   * not grow a second, lagging drawing path back.
   */
  for (const needle of ["detectionOverlay", "requestVideoFrameCallback", "/api/detection", "DetectionMath"]) {
    assert.ok(!appJs.includes(needle), `app.js still refers to ${needle}`);
  }
  assert.doesNotMatch(indexHtml, /detection-math\.js|<canvas/, "the overlay canvas is back");
  assert.match(appJs, /\/api\/renders/, "the page no longer starts or plays renders");
  // Watched while it is made: neither Chrome nor Edge plays a playlist that is
  // still growing, so the page feeds the segments to Media Source itself.
  assert.match(appJs, /\/api\/renders\/stream/, "the page waits for a render to finish again");
  assert.match(appJs, /TelemetryRenderPlayer/, "nothing plays the segments");
  assert.match(indexHtml, /render-player\.js/, "the segment player is not loaded");
  assert.match(appJs, /\/api\/videos\/source/, "the original is no longer served by the server");
  // A capture reads its frame on the server, from the original; it posts no picture.
  const capture = appJs.match(/async function takeCaptureShot\(\)\s*\{[\s\S]*?\n\}/);
  assert.ok(capture, "takeCaptureShot() is missing");
  assert.doesNotMatch(capture[0], /body:/, "a capture uploads the frame again");
});

test("every script the page loads is cached by the service worker", () => {
  // A script in the markup but not in the shell is fetched from the network on
  // an installed copy, and the page is broken exactly when the server is not up.
  const sw = read("service-worker.js");
  for (const src of matchAll(indexHtml, /<script src="([^"]+)"/g)) {
    assert.ok(sw.includes(src), `${src} is not in the service worker shell`);
  }
});

test("the service worker cache turns over when the shell changes", () => {
  // An installed PWA serves the cached shell until CACHE_NAME changes, so a
  // redesign that leaves it alone ships to a browser that will not show it.
  const sw = read("service-worker.js");
  const name = sw.match(/CACHE_NAME\s*=\s*"([^"]+)"/);
  assert.ok(name, "service-worker.js has no CACHE_NAME");
  assert.match(name[1], /-v(\d+)$/, "CACHE_NAME should end in a version this file can bump");
});
