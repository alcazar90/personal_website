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

## Images

Put full-size exports in `content/static/img/` and reference them normally. Sizing is the build's job: it downscales to 1400px, encodes WebP, stamps dimensions and lazy-loading, and links each image back to the untouched original. A 2.4 MB screenshot ships as 25 KB; readers who click get the full-resolution file.

Requires libwebp:

```
brew install webp             # macOS
sudo apt-get install -y webp  # Debian/Ubuntu
```

Without it the build still succeeds, but images ship unoptimized.

## Deploy

Deployed to Cloudflare Pages on push to `master`. See `docs/cutover-guide.md` for the Hugo→Rust migration history.
