# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Source for [alkzar.cl](https://alkzar.cl), a personal blog. This repo holds
content and site config only — the Markdown-to-HTML build itself is
[rustoky](https://github.com/alcazar90/rustoky), a Rust static site
generator developed as its own project so it can be installed and reused
independently. Deployment target is Cloudflare Pages.

For how the build pipeline works internally (frontmatter parsing, math/code/
figure/bibliography/tweet rendering, image optimization, templating), see
rustoky's own `CLAUDE.md` — none of that logic lives in this repo anymore.

## Commands

```sh
# Install the generator (once, or after a rustoky update)
cargo install --git https://github.com/alcazar90/rustoky --locked

# Build the site: content/ -> public/
rustoky build

# Build including draft: true posts, for local preview only (deploy never passes this flag)
rustoky build --drafts

# Build (optionally --drafts), then serve public/ at http://127.0.0.1:8000/ (--port <n> to override)
rustoky serve --drafts

# Scaffold a new post at content/posts/YYYY-MM-DD-<slug>.md
rustoky new-post "My Post Title"
```

## Layout

- `content/` — the source of truth for the site: `content/config.toml` (site config), `content/posts/*.md`, `content/pages/*.md`, `content/static/` (copied verbatim to `public/`).
- `public/` — build output (gitignored). Deleted and regenerated on every `build`.
- `docs/cutover-guide.md` — runbook for the Hugo→Rust/Netlify→Cloudflare Pages DNS cutover.

## Content conventions

- Post files: `content/posts/YYYY-MM-DD-<slug>.md`. The slug is derived from the filename (date prefix stripped) unless overridden by a `slug:` frontmatter field.
- Frontmatter (YAML `---` or TOML `+++`) supports: `title`, `date`, `slug`, `tags`, `description`, `draft`, `lang`. `draft: true` posts are skipped by a plain `build`; pass `--drafts` to render them locally (with a "Draft" badge on the post page and index listing) while still excluding them from `feed.xml`/`sitemap.xml`. The deploy workflow never passes `--drafts`, so drafts can't reach production regardless of local build flags.
- Bibliography sidecar: `content/posts/<post-stem>.refs.yaml`, keyed by citation key, each entry has `author`, `title`, `year` required, `url`/`journal`/`booktitle`/`note` optional. **Never hand-type a `## References` section in a post's Markdown** — rustoky unconditionally deletes a `## References` heading and everything after it from *every* post body at build time (sidecar or not), so hand-written content there is silently discarded. To add references: create the `.refs.yaml` sidecar and mark citation points inline with `\cite{key}`/`\citep{key}`; the build generates the numbered `<section class="references">` itself.
- Tweet cache: `content/tweet-cache.json`, keyed by tweet id — committed to the repo. Delete an entry to force a re-fetch on the next build.
- Images: drop full-resolution exports into `content/static/img/` and reference them normally — sizing is the build's job, not the author's. Derivatives are cached in `.image-cache/files/` (gitignored, mirrors the `content/static/` tree). Requires `cwebp`/`gif2webp` on PATH (`brew install webp`, `apt-get install -y webp`); without them the build still succeeds but ships images unoptimized, so CI verifies they're present.
- `new-post` scaffolds files with a `draft: true` frontmatter stub; it refuses to overwrite an existing file at the target path.

## Deployment

`.github/workflows/deploy.yml` installs `rustoky` from its GitHub repo, runs `rustoky build`, and deploys `public/` to Cloudflare Pages (project name `alkzar`) via `wrangler-action`. Currently triggers on push to `master` and `rust-ssg`. `docs/cutover-guide.md` has the full runbook for the one-time Hugo/Netlify → Rust/Cloudflare Pages migration (DNS, API tokens, etc.) if that context is ever needed.
