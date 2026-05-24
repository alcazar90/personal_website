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
mod render;
mod templates;

use crate::config::Config;
use crate::content::Source;
use crate::templates::{PageContext, PageView, PostContext, PostView, RenderEnv, Templates};

/// Inlined into every page's `<head>`. Single source of truth — no extra
/// stylesheet request, no external CSS file in `public/`.
const MAIN_CSS: &str = include_str!("../../styles/main.css");

fn main() -> ExitCode {
    let cmd = std::env::args().nth(1);
    let result = match cmd.as_deref() {
        Some("build") => cmd_build(),
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
    eprintln!("usage: ssg <build>");
    eprintln!();
    eprintln!("  build  Render content/ to public/.");
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

    for source in &sources {
        let kind = classify(source);
        match kind {
            SourceKind::Post => {
                let bytes = render_and_write_post(&templates, &env, source, out_root)
                    .with_context(|| format!("emitting post {}", source.slug))?;
                post_count += 1;
                total_bytes += bytes;
            }
            SourceKind::Page => {
                let bytes = render_and_write_page(&templates, &env, source, out_root)
                    .with_context(|| format!("emitting page {}", source.slug))?;
                page_count += 1;
                total_bytes += bytes;
            }
        }
    }

    // Stub index — real post listing lands in a later PR.
    let index_html = templates
        .render_index(&env)
        .context("rendering index.html")?;
    let index_path = out_root.join("index.html");
    fs::write(&index_path, &index_html)
        .with_context(|| format!("writing {}", index_path.display()))?;
    total_bytes += index_html.len();

    // Copy static assets verbatim, if any.
    let static_src = content_root.join("static");
    if static_src.exists() {
        copy_dir_recursive(&static_src, out_root)
            .with_context(|| format!("copying {} to {}", static_src.display(), out_root.display()))?;
    }

    eprintln!(
        "built {} post(s), {} page(s); {} bytes of HTML",
        post_count, page_count, total_bytes
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

fn render_and_write_post(
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

    let view = PostView {
        title,
        date,
        date_display,
        reading_time: rendered.reading_time_minutes,
        html: rendered.html,
        slug: source.slug.clone(),
        description,
        lang,
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
    Ok(bytes)
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
