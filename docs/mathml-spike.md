# MathML cross-browser verification spike

Tracking issue: [#18](https://github.com/alcazar90/personal_website/issues/18)

A 5-minute manual checklist to validate that the SSG's MathML output
renders cleanly on the browsers our readers actually use, before we
commit to MathML-only (no KaTeX/MathJax fallback).

---

## 1. What we're testing

The Rust SSG pipeline converts LaTeX (`$...$` and `$$...$$` blocks in
Markdown) to MathML at build time via
[`pulldown-latex`](https://crates.io/crates/pulldown-latex). The
relevant module is [`ssg/src/render/math.rs`](../ssg/src/render/math.rs)
— it calls `pulldown_latex::mathml::push_mathml` and emits a raw
MathML fragment (no `<script>`, no client-side JS).

Browsers are expected to render MathML natively per the [MathML Core
spec](https://www.w3.org/TR/mathml-core/). Coverage matrix:
[MDN — Web/MathML](https://developer.mozilla.org/en-US/docs/Web/MathML).
Live status snapshot at verification time:
[caniuse.com/mathml](https://caniuse.com/mathml).

**Browsers in scope**

- Chrome (macOS, latest stable)
- Firefox (macOS, latest stable)
- Safari (macOS, latest stable)
- Safari (iOS, latest stable)

The math-heaviest post in the archive is
`content/posts/2024-10-02-reinforcement-learning` (lots of inline
expressions plus several display blocks with fractions, sums, and
expectations). That's the canonical "if this looks good, we're good"
target.

---

## 2. Generate the test page

### Option A — Render the real post (preferred once templates land)

On the `rust-ssg` branch with the templates + Flexoki PR merged (or any
future commit where `cargo run -p ssg -- build` produces post HTML):

```bash
cargo run --release -p ssg -- build
open public/posts/reinforcement-learning/index.html   # macOS
# Linux: xdg-open public/posts/reinforcement-learning/index.html
```

Spot-check inline math in paragraphs, the value-function display block,
any summations / expectations, and the policy-gradient derivation.

### Option B — Standalone fixture (works today)

If the templates aren't merged yet, open the self-contained fixture
shipped alongside this doc:

```bash
open docs/mathml-spike.html
```

It contains five representative MathML samples that mimic
`pulldown-latex`'s output, styled to roughly approximate the site
chrome (system font, ~640px content column). No JS, no external CSS,
no network — just open the file.

For iOS Safari: easiest path is `python3 -m http.server` from the repo
root and load `http://<mac-LAN-IP>:8000/docs/mathml-spike.html` from
the iPhone (both devices on the same Wi-Fi).

---

## 3. Verification checklist

Per browser × per sample, confirm each of:

- [ ] Renders at all (not raw `<math>` markup as text)
- [ ] Correct vertical alignment (fractions don't collapse, baselines look right)
- [ ] Subscripts / superscripts at the correct size and position
- [ ] Integrals and summations sized appropriately (display mode visibly bigger than inline)
- [ ] Matrices have proper inter-cell spacing and visible brackets

Fill in `OK`, `minor` (with a note), or `BROKEN` (with a note):

| Sample                       | Chrome (macOS) | Firefox (macOS) | Safari (macOS) | Safari (iOS) |
| ---------------------------- | -------------- | --------------- | -------------- | ------------ |
| 1. Inline `a + b`            |                |                 |                |              |
| 2. Display fraction          |                |                 |                |              |
| 3. Σ subscript / superscript |                |                 |                |              |
| 4. Integral bounds           |                |                 |                |              |
| 5. Matrix                    |                |                 |                |              |

Paste the filled-in table back into [#18](https://github.com/alcazar90/personal_website/issues/18)
when done.

---

## 4. Known browser quirks

Snapshot of what's publicly documented; treat
[caniuse.com/mathml](https://caniuse.com/mathml) and
[MDN's MathML Core compatibility tables](https://developer.mozilla.org/en-US/docs/Web/MathML)
as the source of truth at verification time.

### Chrome (and other Chromium browsers)

- Native MathML Core support landed in Chromium 109 (January 2023),
  closing the long-standing gap after MathML was removed from Blink
  back in 2013.
- Covers the MathML Core subset: `mrow`, `mfrac`, `msqrt`, `mroot`,
  `msub`, `msup`, `msubsup`, `munder`, `mover`, `munderover`, `mtable`,
  `mspace`, basic operator stretching, and `display="block"` centering.
- Older MathML 3 features outside Core (e.g. `mlabeledtr`, `mglyph`,
  elementary-math layout) are intentionally not supported. None of
  these are emitted by `pulldown-latex` for the LaTeX subset we use.

### Firefox

- The most-mature implementation. Ships MathML 3 plus Core. Has
  rendered math natively for ~20 years.
- Generally the reference for what "good" MathML looks like.

### Safari (macOS + iOS)

- WebKit historically had partial MathML; MathML Core conformance has
  been improving with each Safari release through 17.x and 18.x.
- Known rough edges (verify against current Safari at test time):
  - `mfrac` line thickness and vertical padding sometimes differs from
    Chrome/Firefox; rarely unreadable, occasionally a hair too tight.
  - Large operators (`∑`, `∫`, `∏`) in display mode may not stretch as
    aggressively as in Firefox — they render, but feel "inline-sized".
  - `mtable` column spacing can be slightly off; brackets (`mfenced` or
    `<mo stretchy="true">`) usually stretch correctly in modern Safari
    but historically did not.
  - iOS Safari typically inherits the desktop Safari engine, so issues
    track desktop Safari; double-check, don't assume.

If verification turns up something not listed here, add a note to #18
and (if it's a Safari issue) cross-reference WebKit bug tracker.

---

## 5. Decision tree

```
All four browsers render all five samples cleanly?
  YES → Keep pulldown-latex / MathML-only.
        Close #18, no follow-up.

Safari is the only outlier with MINOR spacing / sizing issues?
  YES → Accept. Add a short "Known limitations" note to README.
        Close #18.

Safari (or any browser) produces UNREADABLE output on any sample?
  YES → Add a build-time KaTeX fallback (no runtime JS).
        Open a follow-up issue:
          "Add build-time KaTeX fallback for math rendering"
        Estimated effort: ~½ day.
        Approach:
          - Replace pulldown-latex calls in ssg/src/render/math.rs
            with a KaTeX invocation (either katex-rs, or shelling out
            to a small Node script that calls katex.renderToString).
          - Emit KaTeX's HTML + ship katex.min.css as a site asset.
          - Pipeline change is local: only render/math.rs and the
            asset-copy step are affected.
```

---

## 6. Recommendation

**Default to MathML-only.** As of mid-2026, Chromium and Firefox have
mature MathML Core implementations, and Safari has been steadily
closing the gap over the last several releases. The cost of adding a
KaTeX fallback isn't huge, but it brings in either a Node toolchain
dependency or another Rust crate plus a stylesheet to ship, and it
locks our math output to whatever HTML KaTeX happens to emit (harder to
restyle, harder to debug). For a small personal blog with a handful of
math-heavy posts, the right default is to trust the browser, run this
checklist, and only reach for KaTeX if Safari turns up something
genuinely unreadable rather than merely a few pixels off.
