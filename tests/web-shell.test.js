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
   * The stream image, the enlarged-shot overlay, the install button and the
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

test("the service worker cache turns over when the shell changes", () => {
  // An installed PWA serves the cached shell until CACHE_NAME changes, so a
  // redesign that leaves it alone ships to a browser that will not show it.
  const sw = read("service-worker.js");
  const name = sw.match(/CACHE_NAME\s*=\s*"([^"]+)"/);
  assert.ok(name, "service-worker.js has no CACHE_NAME");
  assert.match(name[1], /-v(\d+)$/, "CACHE_NAME should end in a version this file can bump");
});
