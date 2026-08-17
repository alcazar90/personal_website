# alkzar.cl

Source for [alkzar.cl](https://alkzar.cl). Content and config for the site;
rendered to static HTML by [rustoky](https://github.com/alcazar90/rustoky), a
Rust static site generator developed as a separate project. See `CLAUDE.md`
for design.

## Setup

Install the generator once (or after a `rustoky` update):

```
cargo install --git https://github.com/alcazar90/rustoky --locked
```

## Build

```
rustoky build
```

Produces a static site in `public/`.

## Writing

```
rustoky new-post "Title"
```

Scaffolds `content/posts/YYYY-MM-DD-<slug>.md` as a draft. Drop `draft: true` to publish.

To preview a draft locally without publishing it, build with `--drafts`:

```
rustoky build --drafts
```

Draft posts render into `public/` with a "Draft" badge (on the post page and the home listing) but are still excluded from `feed.xml` and `sitemap.xml`. The deploy workflow never passes this flag, so drafts can't reach production regardless of what you build locally.

To build and preview in the browser in one step:

```
rustoky serve --drafts
```

Serves `public/` at `http://127.0.0.1:8000/` (`--port <n>` to change it). It's a one-shot build, not a watcher — re-run the command to pick up new edits.

## Images

Put full-size exports in `content/static/img/` and reference them normally. Sizing is the build's job: it downscales to 1400px, encodes WebP, stamps dimensions and lazy-loading, and links each image back to the untouched original. A 2.4 MB screenshot ships as 25 KB; readers who click get the full-resolution file.

Requires libwebp:

```
brew install webp             # macOS
sudo apt-get install -y webp  # Debian/Ubuntu
```

Without it the build still succeeds, but images ship unoptimized.

## Deploy

Deployed to Cloudflare Pages on push to `master`.
