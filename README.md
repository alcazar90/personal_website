# alkzar.cl

Source for [alkzar.cl](https://alkzar.cl). Custom Rust static site generator; see `CLAUDE.md` for design.

## Build

```
cargo run --release -p ssg -- build
```

Produces a static site in `public/`.

## Writing

```
cargo run --release -p ssg -- new-post "Title"
```

Scaffolds `content/posts/YYYY-MM-DD-<slug>.md` as a draft. Drop `draft: true` to publish.

To preview a draft locally without publishing it, build with `--drafts`:

```
cargo run --release -p ssg -- build --drafts
```

Draft posts render into `public/` with a "Draft" badge (on the post page and the home listing) but are still excluded from `feed.xml` and `sitemap.xml`. The deploy workflow never passes this flag, so drafts can't reach production regardless of what you build locally.

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
