# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Source for [alkzar.cl](https://alkzar.cl), a personal blog. It's a **custom Rust static site generator** (Cargo workspace, not Node) that reads Markdown content and renders it to static HTML for deployment on Cloudflare Pages.

The repo also contains a one-shot `migrate` binary used to convert legacy Hugo/Rmd content into the clean Markdown this SSG consumes.

## Commands

```sh
# Build the site: content/ -> public/
cargo run --release -p ssg -- build

# Scaffold a new post at content/posts/YYYY-MM-DD-<slug>.md
cargo run --release -p ssg -- new-post "My Post Title"

# Run all tests (both crates)
cargo test --workspace

# Run tests for one crate/module
cargo test -p ssg
cargo test -p ssg render::   # filter by path prefix, e.g. render:: or templates::

# One-shot content migration (Hugo -> this SSG's content format)
cargo run --release -p migrate
```

There is no lint/format CI step defined in this repo beyond `cargo test`; use `cargo fmt`/`cargo clippy` as normal Rust hygiene if touching code.

## Workspace layout

- `ssg/` — the site generator binary (`cargo run -p ssg -- build`). All production logic lives here.
- `migrate/` — standalone one-shot binary for converting legacy Hugo content; not part of the build pipeline.
- `content/` — the actual source of truth for the site: `content/config.toml` (site config), `content/posts/*.md`, `content/pages/*.md`, `content/static/` (copied verbatim to `public/`).
- `templates/*.html` — minijinja templates, baked into the `ssg` binary via `include_str!` (not read from disk at runtime).
- `styles/main.css` — single stylesheet, inlined into every page's `<head>` at build time via `include_str!` (no external CSS request, no separate CSS file ships in `public/`).
- `public/` — build output (gitignored). Deleted and regenerated on every `build`.
- `posts/` — legacy/scratch content from the migration, intentionally still tracked (see `.gitignore` comment); not what the SSG reads (it reads `content/posts/`).
- `docs/cutover-guide.md` — runbook for the Hugo→Rust/Netlify→Cloudflare Pages DNS cutover.

## Build pipeline (`ssg/src/`)

`main.rs` orchestrates the whole build:

1. Load `content/config.toml` (`config.rs`) — flat schema, one struct, no nesting. Includes optional `[giscus]` block for comments (present/absent toggles whether post pages render a giscus mount).
2. Walk `content/posts/` and `content/pages/` (`content.rs`) — parses both `---` YAML and `+++` TOML frontmatter leniently (legacy Hugo posts have inconsistent shapes like `slug: []` or `tags: "single"`), skips drafts, sorts by date descending. Files under `pages/` become `SourceKind::Page`, everything else is a `Post`.
3. Render each source through the Markdown pipeline (`render/mod.rs`), which walks `pulldown-cmark`'s event stream and intercepts:
   - **Headings** → auto `id=` anchors + collected into a Table of Contents (rendered inline as a `<nav class="toc">` block if a post has ≥2 headings).
   - **Fenced code blocks** → `render/code.rs`, syntect-based syntax highlighting emitting CSS classes (not inline colors), so theming can change without a rebuild. Unknown languages fall back to a plain unhighlighted block — highlighting must never fail the build.
   - **Inline/display math** → `render/math.rs`, LaTeX → MathML via `pulldown-latex`. Also does a source-level preprocessing pass: normalizes legacy MathJax delimiters (`\(...\)`, `\[...\]`) to `$`/`$$`, resolves `\label`/`\ref`/`\eqref` into sequential equation numbers, wraps standalone `\begin{equation}` blocks, and fixes `\\` line breaks in display math (pulldown-latex rejects top-level `\\` outside an environment — see `fix_display_math_newlines`) and bare `<br>` spacer lines that CommonMark would otherwise swallow as an HTML block (`isolate_bare_br_tags`). Malformed math falls back to an escaped `<code>` block rather than panicking. Each `\label`'d display-math block gets an `id=` anchor, so `\ref`/`\eqref` render as links (`<a href="#key">N</a>`) that jump to the defining equation. Sections have no equation-style label/counter — cross-reference a heading with a plain markdown link to its auto-generated id, not `\ref`.
   - **Figures** → `render/figure.rs`. `<figure>`/`<figcaption>` are CommonMark "type 6" HTML blocks, so pulldown-cmark treats them as opaque raw HTML and skips inline parsing (including math) on their contents; this module renders `$...$`/`$$...$$` inside figures before the main parse. It also mirrors the equation label system with an independent figure counter: every `<figure id="...">` is auto-numbered in document order, `\figref{key}` resolves to a linked "Figure N", and each `<figcaption>` gets a "Figure N. " prefix inserted.
   - **Bibliography** — if a `<post-stem>.refs.yaml` sidecar exists next to a post (see `render/bibliography.rs`), `\cite{key}`/`\citep{key}` become numbered superscript links and a `<section class="references">` is appended.
   - **Embedded tweets** (`render/tweet.rs`) — `<blockquote class="twitter-tweet">` blocks (X's standard embed snippet) are replaced with a static, pre-themed `.tweet-card` div built from a build-time snapshot of the tweet, rather than upgraded client-side by X's `widgets.js`. The snapshot is fetched from X's public (unofficial) syndication endpoint (`cdn.syndication.twimg.com/tweet-result`, the same one `widgets.js` calls, requiring a `token` derived from the tweet id — see `syndication_token`) and cached at `content/tweet-cache.json`, keyed by tweet id; that cache is committed, so rebuilds are fully offline once a tweet has been fetched once. If a tweet isn't cached and the fetch fails (offline build, deleted tweet, endpoint changes), the original `<blockquote>` markup passes through untouched rather than failing the build. The card reuses the site's own Flexoki CSS variables, so it matches light/dark instantly with the rest of the page — no client-side script, no re-theming flash.
   - **Relative image paths** → rewritten to `/posts/<slug>/<filename>`; absolute/schemed/root-relative URLs pass through untouched.
4. Render through minijinja templates (`templates.rs`) — one context struct per page kind (`PostContext`, `PageContext`, `IndexContext`, `Render404Context`), each wrapping a shared `RenderEnv` (site config + inlined CSS + build year).
5. Emit `public/index.html` (post listing), `public/posts/<slug>/index.html`, `public/<slug>/index.html` (pages), `public/feed.xml` (Atom, hand-rolled in `feed.rs`), `public/sitemap.xml` (also `feed.rs`), and `public/404.html`.
6. Copy `content/static/` verbatim into `public/`.

Every build fully deletes and recreates `public/` — there's no incremental build.

## Content conventions

- Post files: `content/posts/YYYY-MM-DD-<slug>.md`. The slug is derived from the filename (date prefix stripped) unless overridden by a `slug:` frontmatter field.
- Frontmatter (YAML `---` or TOML `+++`) supports: `title`, `date`, `slug`, `tags`, `description`, `draft`, `lang`. `draft: true` posts are skipped entirely.
- Bibliography sidecar: `content/posts/<post-stem>.refs.yaml`, keyed by citation key, each entry has `author`, `title`, `year` required, `url`/`journal`/`booktitle`/`note` optional.
- Tweet cache: `content/tweet-cache.json`, keyed by tweet id — committed to the repo (see `render/tweet.rs`). Delete an entry to force a re-fetch on the next build.
- `new-post` scaffolds files with a `draft: true` frontmatter stub; it refuses to overwrite an existing file at the target path.

## Design principles evident in the code (follow these when extending)

- **Never crash the build on content-level problems.** Bad math, unknown syntax-highlighting languages, and malformed frontmatter shapes all degrade gracefully (fallback rendering, warning to stderr, or skipping the file) rather than aborting `cargo run -p ssg -- build`.
- **Templates and CSS are compiled into the binary** (`include_str!`), not read from disk at runtime — there's no template hot-reload story here.
- Frontmatter parsing is deliberately lenient (see `de_string_lenient`/`de_string_list_lenient` in `content.rs`, duplicated verbatim in `migrate/src/main.rs`) because legacy Hugo content has inconsistent field shapes. Keep both copies in sync if you change this logic.

## Deployment

`.github/workflows/deploy.yml` builds with `cargo run --release -p ssg -- build` and deploys `public/` to Cloudflare Pages (project name `alkzar`) via `wrangler-action`. Currently triggers on push to `master` and `rust-ssg`. `docs/cutover-guide.md` has the full runbook for the one-time Hugo/Netlify → Rust/Cloudflare Pages migration (DNS, API tokens, etc.) if that context is ever needed.
