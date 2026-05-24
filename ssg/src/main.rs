//! `ssg` — Rust static site generator for personal_website.
//!
//! Currently supports only `ssg build`, which loads the site config, walks
//! `content/posts` and `content/pages`, renders each markdown source through
//! the pulldown_cmark → syntect/pulldown-latex pipeline, and prints the
//! resulting HTML to stdout. Templating, asset copying, and the index page
//! land in follow-up PRs.

use anyhow::{Context, Result};
use std::path::Path;
use std::process::ExitCode;

mod config;
mod content;
mod render;

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
    eprintln!("  build  Render all posts and pages to stdout (no templating yet).");
}

fn cmd_build() -> Result<()> {
    let content_root = Path::new("content");

    // Config is optional for now — the render pipeline doesn't depend on it.
    // Once templates land it becomes required.
    let config_path = content_root.join("config.toml");
    let config = if config_path.exists() {
        Some(config::Config::load(&config_path)?)
    } else {
        eprintln!(
            "warning: {} not found; proceeding without site config",
            config_path.display()
        );
        None
    };

    if let Some(cfg) = &config {
        eprintln!("loaded config for site: {}", cfg.title);
    }

    let sources = content::walk(content_root)
        .with_context(|| format!("walking content at {}", content_root.display()))?;
    eprintln!("found {} source(s)", sources.len());

    for source in &sources {
        let rendered = render::render(source)
            .with_context(|| format!("rendering {}", source.path.display()))?;
        println!("<!-- {} ({} min) -->", source.slug, rendered.reading_time_minutes);
        println!("{}", rendered.html);
    }

    Ok(())
}
