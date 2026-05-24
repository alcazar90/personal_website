//! LaTeX → MathML rendering via `pulldown-latex`.
//!
//! Both `inline` and `display` produce a MathML string with no wrapping
//! element; the caller wraps in `<span class="math inline">` or
//! `<div class="math display">` so the surrounding CSS knows the context.
//!
//! Also exports `preprocess_source` — a markdown-level pass that normalizes
//! math delimiters from legacy MathJax conventions (`\(...\)`, `\[...\]`)
//! into the `$/$$` form that pulldown-cmark recognizes, and strips LaTeX
//! environments pulldown-latex doesn't support (`\begin{equation}`, `\label`).

use anyhow::{anyhow, Result};
use pulldown_latex::config::RenderConfig;
use pulldown_latex::mathml::push_mathml;
use pulldown_latex::{Parser, Storage};
use regex::Regex;
use std::sync::OnceLock;

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

/// Preprocess markdown source to make math compatible with our pipeline.
///
/// Two transformations:
/// 1. **Normalize delimiters** from legacy MathJax conventions:
///    - `\(...\)` → `$...$` (inline)
///    - `\[...\]` → `$$...$$` (display)
///    Pulldown-cmark's math extension only recognizes `$/$$`. Posts authored
///    for Hugo+MathJax commonly use the backslash forms (and in raw markdown
///    they appear escaped as `\\(`, `\\[` etc. since `\` itself escapes).
///
/// 2. **Strip unsupported LaTeX environments** that pulldown-latex doesn't
///    parse: `\begin{equation}` / `\end{equation}` (numbering doesn't exist in
///    MathML) and `\label{...}` (cross-references don't either). Common
///    alignment envs (`split`, `align`, `aligned`, `cases`, `matrix`, etc.)
///    are left alone — pulldown-latex handles those.
///
/// Code fence skipping is intentionally NOT implemented: the false-positive
/// risk (a code block legitimately containing `\\(`) is negligible for typical
/// blog content. If it becomes an issue, switch to a line-by-line pass that
/// tracks fenced-code state.
pub fn preprocess_source(body: &str) -> String {
    let s = normalize_delimiters(body);
    strip_unsupported_envs(&s)
}

fn normalize_delimiters(body: &str) -> String {
    // Source uses `\\(...\\)` and `\\[...\\]` — TWO backslashes each. The double
    // backslash is markdown's escape for a literal backslash; the legacy
    // MathJax convention requires the escape because raw `\(` would be CommonMark
    // escape syntax. So the byte pattern is: `\`, `\`, `(` etc.
    //
    // `(?s)` = dot matches newlines (display math often spans lines).
    // Lazy `.*?` so consecutive math blocks don't merge into one giant match.
    static DISPLAY_RE: OnceLock<Regex> = OnceLock::new();
    static INLINE_RE: OnceLock<Regex> = OnceLock::new();
    let display =
        DISPLAY_RE.get_or_init(|| Regex::new(r"(?s)\\\\\[(.*?)\\\\\]").unwrap());
    let inline =
        INLINE_RE.get_or_init(|| Regex::new(r"\\\\\((.*?)\\\\\)").unwrap());

    // For display: collapse internal newlines to spaces. Multi-line `$$...$$`
    // breaks two CommonMark rules at once: lines with 4+ leading spaces become
    // indented code blocks, and pulldown-cmark's math parser doesn't reliably
    // span lines anyway. LaTeX line breaks inside (e.g. `\\` in `split`)
    // survive intact — those are not Markdown newlines, they're LaTeX tokens.
    let body = display.replace_all(body, |caps: &regex::Captures| {
        let single_line: String = caps[1]
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        format!("$$ {} $$", single_line)
    });
    let body = inline.replace_all(&body, |caps: &regex::Captures| {
        format!("${}$", &caps[1])
    });
    body.into_owned()
}

fn strip_unsupported_envs(body: &str) -> String {
    static ENV_RE: OnceLock<Regex> = OnceLock::new();
    static LABEL_RE: OnceLock<Regex> = OnceLock::new();
    let env = ENV_RE
        .get_or_init(|| Regex::new(r"\\(?:begin|end)\{equation\*?\}").unwrap());
    let label = LABEL_RE.get_or_init(|| Regex::new(r"\\label\{[^}]*\}").unwrap());

    let body = env.replace_all(body, "");
    let body = label.replace_all(&body, "");
    body.into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note on test inputs: in Rust source, `\\(` is 2 chars (`\` + `(`). The
    // actual markdown bytes we want to match have THREE chars (`\` `\` `(`),
    // which in a Rust string literal needs `\\\\(` (4 source chars = 3 bytes).
    // Tests below use that 4-source-char form to mirror what the migrator put
    // into content/posts/*.md.

    #[test]
    fn inline_double_backslash_paren_becomes_dollar() {
        let input = "In the vector \\\\(x \\in \\mathbb{R}^n\\\\), we have...";
        let out = preprocess_source(input);
        assert!(out.contains("$x \\in \\mathbb{R}^n$"), "got: {out}");
        assert!(!out.contains("\\\\("), "two-backslash form remained: {out}");
    }

    #[test]
    fn display_double_backslash_bracket_becomes_double_dollar() {
        let input = "Before \\\\[ a^2 + b^2 = c^2 \\\\] after.";
        let out = preprocess_source(input);
        assert!(out.contains("$$ a^2 + b^2 = c^2 $$"), "got: {out}");
    }

    #[test]
    fn equation_env_is_stripped_label_too() {
        let input =
            "\\\\[ \\begin{equation}\\label{eq:foo} x = y \\end{equation} \\\\]";
        let out = preprocess_source(input);
        assert!(out.contains("$$"), "delimiters: {out}");
        assert!(!out.contains("\\begin{equation}"), "begin remained: {out}");
        assert!(!out.contains("\\end{equation}"), "end remained: {out}");
        assert!(!out.contains("\\label"), "label remained: {out}");
        assert!(out.contains("x = y"), "math content lost: {out}");
    }

    #[test]
    fn split_env_is_preserved() {
        let input =
            "\\\\[ \\begin{split} a &= b \\\\ &= c \\end{split} \\\\]";
        let out = preprocess_source(input);
        assert!(out.contains("\\begin{split}"), "split lost: {out}");
        assert!(out.contains("\\end{split}"), "end split lost: {out}");
    }

    #[test]
    fn multiline_display_math_collapses_to_single_line() {
        // Display math spanning multiple lines must collapse so CommonMark
        // doesn't see 4-space-indented lines as a code block.
        let input = "Text\n\n\\\\[\n    x = y\n    + z\n\\\\]\n\nMore text.";
        let out = preprocess_source(input);
        assert!(out.contains("$$ x = y + z $$"), "got: {out}");
        assert!(!out.contains("\n    x"), "indentation leaked: {out}");
    }

    #[test]
    fn existing_dollar_math_passes_through() {
        let input = "Inline $a$ and display $$b$$ together.";
        let out = preprocess_source(input);
        assert_eq!(out, input);
    }

    #[test]
    fn no_math_is_unchanged() {
        let input = "Just regular text with no math at all.";
        let out = preprocess_source(input);
        assert_eq!(out, input);
    }
}
