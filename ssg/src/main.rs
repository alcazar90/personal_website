//! `ssg` — Rust static site generator for personal_website.
//!
//! `ssg build` is the only subcommand. It loads `content/config.toml`, walks
//! `content/posts/` and `content/pages/`, renders each markdown source
//! through the pulldown-cmark → syntect/pulldown-latex pipeline, wraps the
//! result in minijinja templates with Flexoki-themed CSS inlined, and writes
//! the final HTML to `public/`. `content/static/` is copied verbatim into
//! `public/`.

use anyhow::{anyhow, Context, Result};
use std::fs;
use std::path::Path;
use std::process::ExitCode;
use time::OffsetDateTime;

mod config;
mod content;
mod feed;
mod render;
mod templates;

use crate::config::Config;
use crate::content::Source;
use crate::feed::{FeedEntry, SitemapEntry};
use crate::templates::{
    IndexContext, PageContext, PageView, PostContext, PostListEntry, PostView, Render404Context,
    RenderEnv, Templates,
};

/// Inlined into every page's `<head>`. Single source of truth — no extra
/// stylesheet request, no external CSS file in `public/`.
const MAIN_CSS: &str = include_str!("../../styles/main.css");

fn main() -> ExitCode {
    let cmd = std::env::args().nth(1);
    let result = match cmd.as_deref() {
        Some("build") => cmd_build(),
        Some("new-post") => {
            let title = std::env::args().nth(2);
            match title {
                Some(t) if !t.trim().is_empty() => cmd_new_post(&t),
                _ => {
                    eprintln!("ssg: new-post requires a non-empty title argument");
                    eprintln!("usage: ssg new-post \"<title>\"");
                    return ExitCode::from(2);
                }
            }
        }
        Some(other) => {
            eprintln!("ssg: unknown subcommand '{other}'");
            usage();
            return ExitCode::from(2);
        }
        None => {
            usage();
            return ExitCode::from(2);
        }
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("ssg: error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn usage() {
    eprintln!("ssg v{}", env!("CARGO_PKG_VERSION"));
    eprintln!("usage: ssg <build | new-post \"<title>\">");
    eprintln!();
    eprintln!("  build              Render content/ to public/.");
    eprintln!("  new-post \"<title>\"  Scaffold a new post at content/posts/YYYY-MM-DD-<slug>.md.");
}

fn cmd_build() -> Result<()> {
    let content_root = Path::new("content");
    let out_root = Path::new("public");

    let config_path = content_root.join("config.toml");
    let config = Config::load(&config_path).with_context(|| {
        format!(
            "loading site config from {} (required for templated build)",
            config_path.display()
        )
    })?;

    eprintln!("loaded config for site: {}", config.title);

    let sources = content::walk(content_root)
        .with_context(|| format!("walking content at {}", content_root.display()))?;
    eprintln!("found {} source(s)", sources.len());

    let templates = Templates::new().context("initializing minijinja environment")?;
    let year = current_year();
    let env = RenderEnv {
        site: &config,
        inline_css: MAIN_CSS,
        year,
    };

    // Fresh build: blow away public/ so we don't accumulate stale output.
    if out_root.exists() {
        fs::remove_dir_all(out_root)
            .with_context(|| format!("clearing {}", out_root.display()))?;
    }
    fs::create_dir_all(out_root).with_context(|| format!("creating {}", out_root.display()))?;

    let mut post_count = 0usize;
    let mut page_count = 0usize;
    let mut total_bytes = 0usize;
    // Index entries: lightweight view-models for the post listing.
    let mut index_entries: Vec<PostListEntry> = Vec::new();
    // Atom feed entries: include full rendered HTML body.
    let mut feed_entries: Vec<FeedEntry> = Vec::new();
    // Sitemap rows: one per output URL.
    let mut sitemap_entries: Vec<SitemapEntry> = Vec::new();

    for source in &sources {
        let kind = classify(source);
        match kind {
            SourceKind::Post => {
                let outcome = render_and_write_post(&templates, &env, source, out_root)
                    .with_context(|| format!("emitting post {}", source.slug))?;
                post_count += 1;
                total_bytes += outcome.bytes;
                index_entries.push(PostListEntry {
                    title: outcome.title.clone(),
                    slug: source.slug.clone(),
                    date: outcome.date.clone(),
                    date_display: outcome.date_display.clone(),
                });
                feed_entries.push(FeedEntry {
                    title: outcome.title,
                    slug: source.slug.clone(),
                    date: outcome.date.clone(),
                    html: outcome.body_html,
                });
                sitemap_entries.push(SitemapEntry {
                    path: format!("/posts/{}/", source.slug),
                    lastmod: (!outcome.date.is_empty()).then_some(outcome.date),
                });
            }
            SourceKind::Page => {
                let bytes = render_and_write_page(&templates, &env, source, out_root)
                    .with_context(|| format!("emitting page {}", source.slug))?;
                page_count += 1;
                total_bytes += bytes;
                sitemap_entries.push(SitemapEntry {
                    path: format!("/{}/", source.slug),
                    lastmod: source.frontmatter.date.clone(),
                });
            }
        }
    }

    // Real home page: post listing in reverse-chronological order (sort
    // already done by content::walk).
    let index_ctx = IndexContext {
        env: env.clone_borrowed(),
        posts: &index_entries,
    };
    let index_html = templates
        .render_index(&index_ctx)
        .context("rendering index.html")?;
    let index_path = out_root.join("index.html");
    fs::write(&index_path, &index_html)
        .with_context(|| format!("writing {}", index_path.display()))?;
    total_bytes += index_html.len();

    // Atom feed with full post content.
    let feed_xml = feed::generate_atom(&config, &feed_entries);
    let feed_path = out_root.join("feed.xml");
    fs::write(&feed_path, &feed_xml)
        .with_context(|| format!("writing {}", feed_path.display()))?;

    // Sitemap: home + every post + every page (404 deliberately excluded).
    // Home `lastmod` is the newest post's date, so feed readers and crawlers
    // know when the index changed.
    let home_lastmod = sitemap_entries
        .iter()
        .filter(|e| e.path.starts_with("/posts/"))
        .find_map(|e| e.lastmod.clone());
    let mut all_sitemap = Vec::with_capacity(sitemap_entries.len() + 1);
    all_sitemap.push(SitemapEntry {
        path: "/".to_string(),
        lastmod: home_lastmod,
    });
    all_sitemap.extend(sitemap_entries);
    let sitemap_xml = feed::generate_sitemap(&config, &all_sitemap);
    let sitemap_path = out_root.join("sitemap.xml");
    fs::write(&sitemap_path, &sitemap_xml)
        .with_context(|| format!("writing {}", sitemap_path.display()))?;

    // 404 page — Cloudflare Pages serves this on missing routes when it
    // lives at /404.html.
    let not_found_ctx = Render404Context {
        env: env.clone_borrowed(),
    };
    let not_found_html = templates
        .render_404(&not_found_ctx)
        .context("rendering 404.html")?;
    let not_found_path = out_root.join("404.html");
    fs::write(&not_found_path, &not_found_html)
        .with_context(|| format!("writing {}", not_found_path.display()))?;
    total_bytes += not_found_html.len();

    // Copy static assets verbatim, if any.
    let static_src = content_root.join("static");
    if static_src.exists() {
        copy_dir_recursive(&static_src, out_root)
            .with_context(|| format!("copying {} to {}", static_src.display(), out_root.display()))?;
    }

    eprintln!(
        "built {} post(s), {} page(s); {} bytes of HTML; feed.xml {} bytes; sitemap.xml {} bytes",
        post_count,
        page_count,
        total_bytes,
        feed_xml.len(),
        sitemap_xml.len()
    );
    Ok(())
}

enum SourceKind {
    Post,
    Page,
}

fn classify(source: &Source) -> SourceKind {
    // The path-component check is deliberately lossy: if the file lives
    // under `content/pages/`, it's a page; otherwise it's a post. Same
    // policy as content::walk's two-directory scan.
    for comp in source.path.iter() {
        if comp == "pages" {
            return SourceKind::Page;
        }
    }
    SourceKind::Post
}

/// Side-channel return from `render_and_write_post` so the caller can build
/// the index listing, Atom feed, and sitemap without re-rendering anything.
struct PostOutcome {
    title: String,
    date: String,
    date_display: String,
    body_html: String,
    bytes: usize,
}

fn render_and_write_post(
    templates: &Templates,
    env: &RenderEnv<'_>,
    source: &Source,
    out_root: &Path,
) -> Result<PostOutcome> {
    let rendered = render::render(source)?;
    let title = source
        .frontmatter
        .title
        .clone()
        .unwrap_or_else(|| source.slug.clone());
    let date = source
        .frontmatter
        .date
        .clone()
        .unwrap_or_default();
    let date_display = format_date(&date);
    let description = source
        .frontmatter
        .description
        .clone()
        .unwrap_or_else(|| env.site.description.clone());
    let lang = source
        .frontmatter
        .lang
        .clone()
        .unwrap_or_else(|| "en".to_string());

    // Clone body HTML once for the feed; the templated page consumes the
    // original via PostView.
    let body_html = rendered.html.clone();

    let view = PostView {
        title: title.clone(),
        date: date.clone(),
        date_display: date_display.clone(),
        reading_time: rendered.reading_time_minutes,
        html: rendered.html,
        slug: source.slug.clone(),
        description,
        lang,
        toc_html: rendered.toc_html,
    };
    let ctx = PostContext {
        env: env.clone_borrowed(),
        post: view,
    };
    let html = templates.render_post(&ctx)?;

    let out_dir = out_root.join("posts").join(&source.slug);
    fs::create_dir_all(&out_dir)
        .with_context(|| format!("creating {}", out_dir.display()))?;
    let out_path = out_dir.join("index.html");
    let bytes = html.len();
    fs::write(&out_path, html).with_context(|| format!("writing {}", out_path.display()))?;
    Ok(PostOutcome {
        title,
        date,
        date_display,
        body_html,
        bytes,
    })
}

fn render_and_write_page(
    templates: &Templates,
    env: &RenderEnv<'_>,
    source: &Source,
    out_root: &Path,
) -> Result<usize> {
    let rendered = render::render(source)?;
    let title = source
        .frontmatter
        .title
        .clone()
        .unwrap_or_else(|| source.slug.clone());
    let description = source
        .frontmatter
        .description
        .clone()
        .unwrap_or_else(|| env.site.description.clone());
    let lang = source
        .frontmatter
        .lang
        .clone()
        .unwrap_or_else(|| "en".to_string());

    let view = PageView {
        title,
        html: rendered.html,
        slug: source.slug.clone(),
        description,
        lang,
    };
    let ctx = PageContext {
        env: env.clone_borrowed(),
        page: view,
    };
    let html = templates.render_page(&ctx)?;

    let out_dir = out_root.join(&source.slug);
    fs::create_dir_all(&out_dir)
        .with_context(|| format!("creating {}", out_dir.display()))?;
    let out_path = out_dir.join("index.html");
    let bytes = html.len();
    fs::write(&out_path, html).with_context(|| format!("writing {}", out_path.display()))?;
    Ok(bytes)
}

/// Format a YYYY-MM-DD date as "Mon DD, YYYY". Falls back to the raw input
/// if parsing fails — frontmatter dates in the legacy posts are not always
/// strict ISO 8601, and we'd rather show *something* than crash.
fn format_date(raw: &str) -> String {
    if raw.is_empty() {
        return String::new();
    }
    // Try parsing as a plain calendar date first.
    let trimmed = raw.split('T').next().unwrap_or(raw);
    let parts: Vec<&str> = trimmed.split('-').collect();
    if parts.len() != 3 {
        return raw.to_string();
    }
    let (Ok(y), Ok(m), Ok(d)) = (
        parts[0].parse::<i32>(),
        parts[1].parse::<u8>(),
        parts[2].parse::<u8>(),
    ) else {
        return raw.to_string();
    };
    let month = match m {
        1 => "Jan",
        2 => "Feb",
        3 => "Mar",
        4 => "Apr",
        5 => "May",
        6 => "Jun",
        7 => "Jul",
        8 => "Aug",
        9 => "Sep",
        10 => "Oct",
        11 => "Nov",
        12 => "Dec",
        _ => return raw.to_string(),
    };
    format!("{month} {d}, {y}")
}

fn current_year() -> i32 {
    // OffsetDateTime::now_local can fail in unusual environments; fall back
    // to UTC and finally to a sane default so the build always succeeds.
    OffsetDateTime::now_utc().year()
}

/// Scaffold a new post stub under `content/posts/YYYY-MM-DD-<slug>.md`.
/// Refuses to overwrite existing files — re-running on the same title the
/// same day prints an error so a typo doesn't silently nuke draft work.
fn cmd_new_post(title: &str) -> Result<()> {
    let posts_dir = Path::new("content").join("posts");
    let date = today_iso();
    let created = new_post_at(&posts_dir, title, &date)?;
    println!("created: {}", created.display());
    Ok(())
}

/// Inner helper for `cmd_new_post`. Split out so tests can pass an explicit
/// temp dir + date — no cwd mutation, no system-clock flakes.
fn new_post_at(
    posts_dir: &Path,
    title: &str,
    date: &str,
) -> Result<std::path::PathBuf> {
    let title = title.trim();
    let slug = slugify(title);
    if slug.is_empty() {
        return Err(anyhow!(
            "title {:?} produces an empty slug; pick something with at least one alphanumeric character",
            title
        ));
    }
    fs::create_dir_all(posts_dir)
        .with_context(|| format!("creating {}", posts_dir.display()))?;
    let filename = format!("{date}-{slug}.md");
    let path = posts_dir.join(&filename);
    if path.exists() {
        return Err(anyhow!(
            "refusing to overwrite existing file: {}",
            path.display()
        ));
    }
    let body = format!(
        "---\ntitle: \"{title}\"\ndate: {date}\nslug: {slug}\ntags: []\ndraft: true\n---\n\n\n"
    );
    fs::write(&path, body).with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

/// Today's date in YYYY-MM-DD. Falls back to UTC if local offset is
/// unavailable (matches `current_year`'s posture).
fn today_iso() -> String {
    let d = OffsetDateTime::now_utc().date();
    // time::Date's Display is ISO 8601 (YYYY-MM-DD).
    format!("{:04}-{:02}-{:02}", d.year(), u8::from(d.month()), d.day())
}

/// Turn a human title into a URL slug.
///
/// Rules:
/// - Spanish (and friends) accented letters fold to ASCII (e.g. "Útil" → "util")
/// - lowercase
/// - non-alphanumeric runs collapse to a single `-`
/// - leading/trailing `-` trimmed
///
/// Inline so we don't pull in a `slug`/`unicode-normalization` crate just
/// for this CLI ergonomics.
fn slugify(input: &str) -> String {
    let folded: String = input
        .chars()
        .map(fold_accent)
        .flat_map(|c| c.to_lowercase())
        .collect();
    let mut out = String::with_capacity(folded.len());
    let mut prev_dash = true; // suppress leading dash
    for c in folded.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c);
            prev_dash = false;
        } else if !prev_dash {
            out.push('-');
            prev_dash = true;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    out
}

/// Best-effort accent folding for the languages we actually write in
/// (English + Spanish). Anything outside the table falls through unchanged
/// and is then dropped by the alphanumeric filter in `slugify`.
fn fold_accent(c: char) -> char {
    match c {
        'á' | 'à' | 'â' | 'ä' | 'ã' | 'å' => 'a',
        'Á' | 'À' | 'Â' | 'Ä' | 'Ã' | 'Å' => 'A',
        'é' | 'è' | 'ê' | 'ë' => 'e',
        'É' | 'È' | 'Ê' | 'Ë' => 'E',
        'í' | 'ì' | 'î' | 'ï' => 'i',
        'Í' | 'Ì' | 'Î' | 'Ï' => 'I',
        'ó' | 'ò' | 'ô' | 'ö' | 'õ' => 'o',
        'Ó' | 'Ò' | 'Ô' | 'Ö' | 'Õ' => 'O',
        'ú' | 'ù' | 'û' | 'ü' => 'u',
        'Ú' | 'Ù' | 'Û' | 'Ü' => 'U',
        'ñ' => 'n',
        'Ñ' => 'N',
        'ç' => 'c',
        'Ç' => 'C',
        other => other,
    }
}

/// Recursively copy `src` directory contents into `dst`. Files overwrite,
/// directories are created. Symlinks are dereferenced (read & copied).
fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<()> {
    if !src.is_dir() {
        return Err(anyhow!("source {} is not a directory", src.display()));
    }
    fs::create_dir_all(dst).with_context(|| format!("creating {}", dst.display()))?;
    for entry in fs::read_dir(src).with_context(|| format!("reading {}", src.display()))? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_recursive(&from, &to)?;
        } else {
            fs::copy(&from, &to)
                .with_context(|| format!("copying {} → {}", from.display(), to.display()))?;
        }
    }
    Ok(())
}

impl<'a> RenderEnv<'a> {
    fn clone_borrowed(&self) -> RenderEnv<'a> {
        RenderEnv {
            site: self.site,
            inline_css: self.inline_css,
            year: self.year,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_date_formats_iso_date() {
        assert_eq!(format_date("2024-10-02"), "Oct 2, 2024");
        assert_eq!(format_date("2024-01-15T00:00:00Z"), "Jan 15, 2024");
    }

    #[test]
    fn format_date_falls_back_on_garbage() {
        assert_eq!(format_date(""), "");
        assert_eq!(format_date("not-a-date"), "not-a-date");
        // Out-of-range month falls back to the raw string.
        assert_eq!(format_date("2024-13-40"), "2024-13-40");
    }

    #[test]
    fn slugify_basic_lowercase_and_spaces() {
        assert_eq!(slugify("Hello World"), "hello-world");
        assert_eq!(slugify("HELLO"), "hello");
    }

    #[test]
    fn slugify_collapses_runs_of_non_alnum() {
        assert_eq!(slugify("foo   bar"), "foo-bar");
        assert_eq!(slugify("foo--bar"), "foo-bar");
        assert_eq!(slugify("a / b / c"), "a-b-c");
        assert_eq!(slugify("hello, world!"), "hello-world");
    }

    #[test]
    fn slugify_trims_leading_and_trailing_dashes() {
        assert_eq!(slugify("  hello  "), "hello");
        assert_eq!(slugify("---hello---"), "hello");
        assert_eq!(slugify("!!!Hello!!!"), "hello");
    }

    #[test]
    fn slugify_folds_spanish_accents() {
        assert_eq!(slugify("Útil"), "util");
        assert_eq!(slugify("Año Nuevo"), "ano-nuevo");
        assert_eq!(slugify("Canción de Otoño"), "cancion-de-otono");
        assert_eq!(slugify("Cómo escribir Rust"), "como-escribir-rust");
    }

    #[test]
    fn slugify_drops_unknown_unicode() {
        // Emoji/CJK fall through to the non-alnum branch and are skipped.
        assert_eq!(slugify("Rust 🦀 rocks"), "rust-rocks");
    }

    #[test]
    fn new_post_at_creates_file_with_expected_frontmatter() {
        let tmp = std::env::temp_dir().join(format!(
            "ssg-newpost-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let posts_dir = tmp.join("content").join("posts");

        let created = new_post_at(&posts_dir, "Hello World", "2026-05-23")
            .expect("first invocation should succeed");
        assert_eq!(
            created,
            posts_dir.join("2026-05-23-hello-world.md"),
            "unexpected output path"
        );
        let body = fs::read_to_string(&created).unwrap();
        assert!(body.starts_with("---\n"), "missing frontmatter open: {body}");
        assert!(body.contains("title: \"Hello World\""), "missing title: {body}");
        assert!(body.contains("date: 2026-05-23"), "missing date: {body}");
        assert!(body.contains("slug: hello-world"), "missing slug: {body}");
        assert!(body.contains("tags: []"), "missing tags: {body}");
        assert!(body.contains("draft: true"), "missing draft flag: {body}");
        assert!(body.ends_with("---\n\n\n"), "missing trailing blank lines: {body}");

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn new_post_at_refuses_overwrite() {
        let tmp = std::env::temp_dir().join(format!(
            "ssg-newpost-overwrite-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let posts_dir = tmp.join("content").join("posts");
        new_post_at(&posts_dir, "Hello World", "2026-05-23").unwrap();
        let second = new_post_at(&posts_dir, "Hello World", "2026-05-23");
        assert!(second.is_err(), "second call should refuse to overwrite");
        let msg = format!("{:#}", second.unwrap_err());
        assert!(msg.contains("refusing to overwrite"), "got: {msg}");
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn new_post_at_rejects_empty_slug_titles() {
        let tmp = std::env::temp_dir().join(format!(
            "ssg-newpost-empty-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let posts_dir = tmp.join("content").join("posts");
        let err = new_post_at(&posts_dir, "   !!!   ", "2026-05-23")
            .expect_err("empty-slug title should be rejected");
        assert!(format!("{err:#}").contains("empty slug"));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn today_iso_is_well_formed() {
        let s = today_iso();
        let parts: Vec<&str> = s.split('-').collect();
        assert_eq!(parts.len(), 3, "expected YYYY-MM-DD, got {s}");
        assert_eq!(parts[0].len(), 4);
        assert_eq!(parts[1].len(), 2);
        assert_eq!(parts[2].len(), 2);
        assert!(parts.iter().all(|p| p.chars().all(|c| c.is_ascii_digit())));
    }

    #[test]
    fn copy_dir_recursive_round_trip() {
        let tmp = std::env::temp_dir().join(format!(
            "ssg-copy-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let src = tmp.join("src");
        let dst = tmp.join("dst");
        fs::create_dir_all(src.join("nested")).unwrap();
        fs::write(src.join("a.txt"), b"alpha").unwrap();
        fs::write(src.join("nested/b.txt"), b"beta").unwrap();

        copy_dir_recursive(&src, &dst).unwrap();
        assert_eq!(fs::read(dst.join("a.txt")).unwrap(), b"alpha");
        assert_eq!(fs::read(dst.join("nested/b.txt")).unwrap(), b"beta");

        let _ = fs::remove_dir_all(&tmp);
    }
}
