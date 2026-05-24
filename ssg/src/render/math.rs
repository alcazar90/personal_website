//! LaTeX → MathML rendering via `pulldown-latex`.
//!
//! Both `inline` and `display` produce a MathML string with no wrapping
//! element; the caller wraps in `<span class="math inline">` or
//! `<div class="math display">` so the surrounding CSS knows the context.

use anyhow::{anyhow, Result};
use pulldown_latex::config::RenderConfig;
use pulldown_latex::mathml::push_mathml;
use pulldown_latex::{Parser, Storage};

fn render(latex: &str, display_style: bool) -> Result<String> {
    let storage = Storage::new();
    let parser = Parser::new(latex, &storage);
    let mut out = String::new();
    let config = RenderConfig {
        display_mode: if display_style {
            pulldown_latex::config::DisplayMode::Block
        } else {
            pulldown_latex::config::DisplayMode::Inline
        },
        ..RenderConfig::default()
    };
    push_mathml(&mut out, parser, config).map_err(|e| anyhow!("mathml render error: {e}"))?;
    Ok(out)
}

/// Render inline math (typically `$...$`).
pub fn inline(latex: &str) -> Result<String> {
    render(latex, false)
}

/// Render display math (typically `$$...$$`).
pub fn display(latex: &str) -> Result<String> {
    render(latex, true)
}
