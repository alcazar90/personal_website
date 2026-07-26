//! Build-time image optimization.
//!
//! Authoring images should not require thinking about file size: drop a
//! full-resolution export into `content/static/img/` and reference it
//! normally. This module downscales and re-encodes every raster image to
//! WebP at build time, and `rewrite_images` rewrites the rendered HTML to
//! point at the derivative — stamping intrinsic `width`/`height` (so images
//! never cause layout shift), `loading`/`decoding` hints, and a link back to
//! the untouched original so readers can still click through to full
//! resolution.
//!
//! Encoding shells out to `cwebp` / `gif2webp` (Google's libwebp tools)
//! rather than pulling an encoder into the dependency graph: WebP encoding
//! would mean `libwebp-sys` and a C toolchain in the build, for work that
//! only happens when the author adds an image.
//!
//! Derivatives are cached in `.image-cache/` (gitignored, mirroring the
//! `content/static/` tree) and regenerated when the source is newer. A clean
//! checkout re-encodes everything on the first build; nothing derived is
//! tracked in git.
//!
//! Following the same posture as `render::tweet`, every failure degrades
//! instead of aborting: missing tools, a failed encode, or a derivative that
//! came out bigger than the source all leave that image untouched in the
//! output, with a warning on stderr.

use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

/// Where derivatives are written. Mirrors the `content/static/` tree so the
/// whole directory can be copied into `public/` alongside the originals.
pub const CACHE_DIR: &str = ".image-cache";

/// Images wider than this are downscaled. The content column is 700px
/// (`styles/main.css`), so this still covers 2x retina.
const MAX_WIDTH: u32 = 1400;

/// cwebp quality. 82 is visually transparent for screenshots and plots while
/// cutting typical matplotlib PNGs by >95%.
const QUALITY: &str = "82";

const RASTER_EXTS: [&str; 4] = ["png", "jpg", "jpeg", "gif"];

/// One optimized image: where the derivative lives, its intrinsic size, and
/// the original it was derived from.
#[derive(Debug, Clone)]
pub struct ImageEntry {
    /// Site-absolute URL of the WebP derivative, e.g. `/img/foo.webp`.
    pub optimized_url: String,
    /// Site-absolute URL of the untouched original, e.g. `/img/foo.png`.
    pub original_url: String,
    /// Intrinsic dimensions *of the derivative*, for `width`/`height`.
    pub width: usize,
    pub height: usize,
}

/// Lookup from original site-absolute URL → optimized derivative.
#[derive(Debug, Default, Clone)]
pub struct ImageManifest {
    entries: HashMap<String, ImageEntry>,
}

impl ImageManifest {
    pub fn get(&self, url: &str) -> Option<&ImageEntry> {
        self.entries.get(url)
    }

    #[cfg(test)]
    fn insert(&mut self, key: &str, entry: ImageEntry) {
        self.entries.insert(key.to_string(), entry);
    }
}

/// Summary of an optimization pass, for the build log.
pub struct OptimizeReport {
    pub optimized: usize,
    pub reused: usize,
    pub skipped: usize,
    pub pruned: usize,
    pub original_bytes: u64,
    pub optimized_bytes: u64,
}

/// Walk `static_root`, encode every raster image to WebP under `cache_root`,
/// and return the manifest. Never fails the build: on missing tools this
/// returns an empty manifest and the site ships its originals.
pub fn optimize(static_root: &Path, cache_root: &Path) -> (ImageManifest, OptimizeReport) {
    let mut manifest = ImageManifest::default();
    let mut report = OptimizeReport {
        optimized: 0,
        reused: 0,
        skipped: 0,
        pruned: 0,
        original_bytes: 0,
        optimized_bytes: 0,
    };

    if !static_root.is_dir() {
        return (manifest, report);
    }

    let have_cwebp = tool_available("cwebp");
    let have_gif2webp = tool_available("gif2webp");
    if !have_cwebp {
        eprintln!(
            "ssg: warning: `cwebp` not found on PATH; shipping unoptimized images. \
             Install it with `brew install webp` or `apt-get install -y webp`."
        );
    }
    if !have_gif2webp {
        eprintln!("ssg: warning: `gif2webp` not found on PATH; GIFs will ship unoptimized.");
    }
    if !have_cwebp && !have_gif2webp {
        return (manifest, report);
    }

    report.pruned = prune_stale(static_root, cache_root);

    for src in walk_images(static_root) {
        let Ok(rel) = src.strip_prefix(static_root) else {
            continue;
        };
        let is_gif = has_ext(&src, "gif");
        if (is_gif && !have_gif2webp) || (!is_gif && !have_cwebp) {
            report.skipped += 1;
            continue;
        }

        let out = cache_root.join(rel).with_extension("webp");
        let src_bytes = file_len(&src);

        let fresh = is_fresh(&src, &out);
        if !fresh {
            if let Some(parent) = out.parent() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    eprintln!("ssg: warning: creating {}: {e}", parent.display());
                    report.skipped += 1;
                    continue;
                }
            }
            if let Err(e) = encode(&src, &out, is_gif) {
                eprintln!("ssg: warning: optimizing {}: {e}", src.display());
                let _ = std::fs::remove_file(&out);
                report.skipped += 1;
                continue;
            }
        }

        let out_bytes = file_len(&out);
        // A derivative that came out bigger is worse than doing nothing —
        // already-optimized PNGs and tiny icons land here.
        if out_bytes == 0 || out_bytes >= src_bytes {
            report.skipped += 1;
            continue;
        }

        let Some(dims) = imagesize::size(&out).ok() else {
            eprintln!("ssg: warning: couldn't read dimensions of {}", out.display());
            report.skipped += 1;
            continue;
        };

        let original_url = to_url(rel);
        let optimized_url = to_url(&rel.with_extension("webp"));
        manifest.entries.insert(
            original_url.clone(),
            ImageEntry {
                optimized_url,
                original_url,
                width: dims.width,
                height: dims.height,
            },
        );

        if fresh {
            report.reused += 1;
        } else {
            report.optimized += 1;
        }
        report.original_bytes += src_bytes;
        report.optimized_bytes += out_bytes;
    }

    (manifest, report)
}

/// Run the encoder. `cwebp` handles stills (and downscales); `gif2webp`
/// handles GIFs, which may be animated — it has no resize flag, so animated
/// sources are re-encoded at their original dimensions.
fn encode(src: &Path, out: &Path, is_gif: bool) -> Result<(), String> {
    if is_gif {
        return encode_gif(src, out);
    }
    let mut cmd = Command::new("cwebp");
    cmd.args(["-q", QUALITY, "-mt", "-quiet"]);
    // Only downscale; never upscale a small source.
    if let Ok(dims) = imagesize::size(src) {
        if dims.width as u32 > MAX_WIDTH {
            cmd.args(["-resize", &MAX_WIDTH.to_string(), "0"]);
        }
    }
    cmd.arg(src).arg("-o").arg(out);
    run(cmd)
}

/// Animated GIFs are unpredictable: whether lossless or lossy WebP wins
/// depends entirely on the source. Measured on this site's three GIFs,
/// lossless beat lossy by 40% on one and lost by 79% on another. So encode
/// both ways and keep the smaller — there are only ever a handful of GIFs,
/// and guessing wrong costs megabytes.
fn encode_gif(src: &Path, out: &Path) -> Result<(), String> {
    let lossy_path = out.with_extension("lossy.webp");

    let mut lossless = Command::new("gif2webp");
    lossless.args(["-q", QUALITY, "-m", "4", "-mt"]);
    lossless.arg(src).arg("-o").arg(out);
    let lossless_ok = run(lossless).is_ok();

    let mut lossy = Command::new("gif2webp");
    lossy.args(["-lossy", "-q", QUALITY, "-m", "4", "-mt"]);
    lossy.arg(src).arg("-o").arg(&lossy_path);
    let lossy_ok = run(lossy).is_ok();

    let keep_lossy = match (lossless_ok, lossy_ok) {
        (true, true) => file_len(&lossy_path) < file_len(out),
        (false, true) => true,
        (true, false) => false,
        (false, false) => return Err("both gif2webp passes failed".to_string()),
    };
    if keep_lossy {
        std::fs::rename(&lossy_path, out).map_err(|e| format!("keeping lossy encode: {e}"))?;
    } else {
        let _ = std::fs::remove_file(&lossy_path);
    }
    Ok(())
}

fn run(mut cmd: Command) -> Result<(), String> {
    let output = cmd.output().map_err(|e| format!("spawning encoder: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "encoder exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(())
}

/// Delete cached derivatives whose source image is gone. Without this the
/// cache only ever grows, and deleted images keep getting copied into
/// `public/` long after they stop being referenced.
fn prune_stale(static_root: &Path, cache_root: &Path) -> usize {
    if !cache_root.is_dir() {
        return 0;
    }
    let mut pruned = 0;
    for entry in walkdir::WalkDir::new(cache_root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let Ok(rel) = entry.path().strip_prefix(cache_root) else {
            continue;
        };
        // A derivative is live if any supported source extension still
        // exists for its stem.
        let live = RASTER_EXTS.iter().any(|ext| {
            static_root.join(rel).with_extension(ext).exists()
                || static_root
                    .join(rel)
                    .with_extension(ext.to_uppercase())
                    .exists()
        });
        if !live && std::fs::remove_file(entry.path()).is_ok() {
            pruned += 1;
        }
    }
    pruned
}

/// A derivative is fresh when it exists and is no older than its source.
fn is_fresh(src: &Path, out: &Path) -> bool {
    let (Ok(s), Ok(o)) = (mtime(src), mtime(out)) else {
        return false;
    };
    o >= s
}

fn mtime(p: &Path) -> Result<SystemTime, ()> {
    std::fs::metadata(p).and_then(|m| m.modified()).map_err(|_| ())
}

fn file_len(p: &Path) -> u64 {
    std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)
}

fn tool_available(name: &str) -> bool {
    Command::new(name)
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn has_ext(p: &Path, ext: &str) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case(ext))
        .unwrap_or(false)
}

fn walk_images(root: &Path) -> Vec<PathBuf> {
    walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .filter(|p| RASTER_EXTS.iter().any(|ext| has_ext(p, ext)))
        .collect()
}

/// `img/foo/bar.png` → `/img/foo/bar.png`, with platform separators
/// normalized to URL separators.
fn to_url(rel: &Path) -> String {
    let mut s = String::from("/");
    s.push_str(&rel.to_string_lossy().replace(std::path::MAIN_SEPARATOR, "/"));
    s
}

// ── HTML rewriting ──────────────────────────────────────────────────────────

/// Matches the tags we need to track: anchors (to know whether an image is
/// already inside a link) and images themselves.
static TAG_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<a\b[^>]*>|</a\s*>|<img\b[^>]*>").unwrap());

static SRC_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"(?is)\bsrc\s*=\s*("([^"]*)"|'([^']*)')"#).unwrap());

/// Rewrite every `<img>` in `html` to use its optimized derivative.
///
/// For images in the manifest: `src` points at the WebP, intrinsic
/// `width`/`height` are added (only when the author didn't set their own
/// display size), and the tag is wrapped in a link to the full-resolution
/// original — unless it already sits inside an anchor.
///
/// Images not in the manifest (external hosts, unknown paths) still get
/// `loading`/`decoding` hints but are otherwise left alone.
///
/// The first image on the page never gets `loading="lazy"`: it is the
/// likeliest LCP element, and deferring it would make the metric worse.
pub fn rewrite_images(html: &str, manifest: &ImageManifest, site_url: &str) -> String {
    let mut out = String::with_capacity(html.len() + 256);
    let mut last = 0usize;
    let mut link_depth = 0usize;
    let mut seen_image = false;

    for m in TAG_RE.find_iter(html) {
        out.push_str(&html[last..m.start()]);
        last = m.end();
        let tag = m.as_str();

        if tag.starts_with("</") {
            link_depth = link_depth.saturating_sub(1);
            out.push_str(tag);
        } else if tag.len() > 3 && tag[..3].eq_ignore_ascii_case("<a ")
            || tag.eq_ignore_ascii_case("<a>")
        {
            link_depth += 1;
            out.push_str(tag);
        } else {
            let is_first = !seen_image;
            seen_image = true;
            out.push_str(&rewrite_one(
                tag,
                manifest,
                site_url,
                link_depth > 0,
                is_first,
            ));
        }
    }
    out.push_str(&html[last..]);
    out
}

fn rewrite_one(
    tag: &str,
    manifest: &ImageManifest,
    site_url: &str,
    inside_link: bool,
    is_first: bool,
) -> String {
    let Some(caps) = SRC_RE.captures(tag) else {
        return tag.to_string();
    };
    let src = caps
        .get(2)
        .or_else(|| caps.get(3))
        .map(|m| m.as_str())
        .unwrap_or("");

    let entry = lookup(manifest, src, site_url);

    let mut tag = tag.to_string();
    if let Some(entry) = entry {
        // Preserve any fragment/query the author wrote (legacy Hugo posts use
        // `#center`), so nothing that depended on it silently changes.
        let suffix = src
            .find(['#', '?'])
            .map(|i| &src[i..])
            .unwrap_or("");
        let new_src = format!("{}{}", entry.optimized_url, suffix);
        tag = replace_src(&tag, &new_src);
        // Only supply intrinsic dimensions when the author didn't set an
        // explicit display size — overriding theirs would change the layout.
        if !has_attr(&tag, "width") && !has_attr(&tag, "height") {
            tag = append_attrs(&tag, &format!(r#" width="{}" height="{}""#, entry.width, entry.height));
        }
    }

    if !has_attr(&tag, "decoding") {
        tag = append_attrs(&tag, r#" decoding="async""#);
    }
    if !is_first && !has_attr(&tag, "loading") {
        tag = append_attrs(&tag, r#" loading="lazy""#);
    }

    match entry {
        Some(entry) if !inside_link => format!(
            r#"<a class="img-original" href="{}" title="Open full-resolution image">{}</a>"#,
            entry.original_url, tag
        ),
        _ => tag,
    }
}

/// Resolve an `src` to a manifest entry, tolerating fragments, query
/// strings, HTML-escaped ampersands, and absolute URLs pointing back at this
/// same site (legacy posts hard-code `https://alkzar.cl/img/...`).
fn lookup<'a>(manifest: &'a ImageManifest, src: &str, site_url: &str) -> Option<&'a ImageEntry> {
    let mut path = src.split(['#', '?']).next().unwrap_or(src);
    let stripped = site_url.trim_end_matches('/');
    if !stripped.is_empty() {
        if let Some(rest) = path.strip_prefix(stripped) {
            path = rest;
        }
    }
    if !path.starts_with('/') {
        return None;
    }
    manifest
        .get(path)
        .or_else(|| manifest.get(&path.replace("&amp;", "&")))
}

fn replace_src(tag: &str, new_src: &str) -> String {
    SRC_RE
        .replace(tag, format!(r#"src="{new_src}""#).as_str())
        .into_owned()
}

fn has_attr(tag: &str, name: &str) -> bool {
    Regex::new(&format!(r"(?i)\b{}\s*=", regex::escape(name)))
        .map(|re| re.is_match(tag))
        .unwrap_or(false)
}

/// Insert attributes just before the tag's closing delimiter, preserving
/// whether the author wrote `>` or `/>`.
fn append_attrs(tag: &str, attrs: &str) -> String {
    let trimmed = tag.trim_end();
    if let Some(body) = trimmed.strip_suffix("/>") {
        format!("{}{} />", body.trim_end(), attrs)
    } else if let Some(body) = trimmed.strip_suffix('>') {
        format!("{}{}>", body.trim_end(), attrs)
    } else {
        tag.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> ImageManifest {
        let mut m = ImageManifest::default();
        m.insert(
            "/img/a.png",
            ImageEntry {
                optimized_url: "/img/a.webp".into(),
                original_url: "/img/a.png".into(),
                width: 1400,
                height: 700,
            },
        );
        m
    }

    #[test]
    fn rewrites_src_and_adds_dimensions_and_link() {
        let out = rewrite_images(
            r#"<p><img src="/img/a.png" alt="A"></p>"#,
            &manifest(),
            "https://alkzar.cl",
        );
        assert!(out.contains(r#"src="/img/a.webp""#), "{out}");
        assert!(out.contains(r#"width="1400""#), "{out}");
        assert!(out.contains(r#"height="700""#), "{out}");
        assert!(out.contains(r#"<a class="img-original" href="/img/a.png""#), "{out}");
        assert!(out.contains(r#"alt="A""#), "alt must survive: {out}");
    }

    #[test]
    fn first_image_is_not_lazy_but_later_ones_are() {
        let out = rewrite_images(
            r#"<img src="/img/a.png"><img src="/img/a.png">"#,
            &manifest(),
            "https://alkzar.cl",
        );
        let first = &out[..out.len() / 2];
        assert!(!first.contains("loading="), "LCP image must stay eager: {first}");
        assert_eq!(out.matches(r#"loading="lazy""#).count(), 1, "{out}");
        assert_eq!(out.matches(r#"decoding="async""#).count(), 2, "{out}");
    }

    #[test]
    fn does_not_nest_anchors() {
        let out = rewrite_images(
            r#"<a href="https://example.com"><img src="/img/a.png"></a>"#,
            &manifest(),
            "https://alkzar.cl",
        );
        assert!(!out.contains("img-original"), "must not nest links: {out}");
        assert!(out.contains(r#"src="/img/a.webp""#), "{out}");
    }

    #[test]
    fn respects_author_supplied_display_size() {
        let out = rewrite_images(
            r#"<img src="/img/a.png" width="450" height="450"/>"#,
            &manifest(),
            "https://alkzar.cl",
        );
        assert!(out.contains(r#"width="450""#), "{out}");
        assert!(!out.contains(r#"width="1400""#), "{out}");
        assert!(out.contains("/>"), "self-closing form must survive: {out}");
    }

    #[test]
    fn unknown_and_external_images_pass_through_with_hints_only() {
        let out = rewrite_images(
            r#"<img src="/img/a.png"><img src="https://example.com/x.png">"#,
            &manifest(),
            "https://alkzar.cl",
        );
        assert!(out.contains(r#"src="https://example.com/x.png""#), "{out}");
        assert_eq!(out.matches("img-original").count(), 1, "{out}");
    }

    #[test]
    fn resolves_absolute_self_urls_and_preserves_fragments() {
        let out = rewrite_images(
            r#"<img src="https://alkzar.cl/img/a.png#center">"#,
            &manifest(),
            "https://alkzar.cl",
        );
        assert!(out.contains(r#"src="/img/a.webp#center""#), "{out}");
    }

    #[test]
    fn empty_manifest_leaves_markup_essentially_untouched() {
        let out = rewrite_images(
            r#"<img src="/img/a.png" alt="A">"#,
            &ImageManifest::default(),
            "https://alkzar.cl",
        );
        assert!(out.contains(r#"src="/img/a.png""#), "{out}");
        assert!(!out.contains("img-original"), "{out}");
    }

    #[test]
    fn to_url_is_root_relative() {
        assert_eq!(to_url(Path::new("img/a/b.png")), "/img/a/b.png");
    }
}
