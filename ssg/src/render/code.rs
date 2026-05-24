//! Syntect-based syntax highlighting for fenced code blocks.
//!
//! We emit CSS classes (not inline colors) so the eventual theme/Flexoki
//! palette can swap looks without rebuilding the site. The `SyntaxSet` is
//! built once and shared via `Lazy`; loading the default newlines variant
//! costs a few ms and we don't want to pay it per fence.

use once_cell::sync::Lazy;
use syntect::html::{ClassStyle, ClassedHTMLGenerator};
use syntect::parsing::SyntaxSet;
use syntect::util::LinesWithEndings;

static SYNTAX_SET: Lazy<SyntaxSet> = Lazy::new(SyntaxSet::load_defaults_newlines);

/// Render a fenced code block to HTML.
///
/// On unknown / empty `lang` or any syntect error, fall back to a plain
/// `<pre><code>…</code></pre>` with HTML-escaped content. Highlighting must
/// never crash the build.
pub fn highlight(code: &str, lang: &str) -> String {
    let trimmed_lang = lang.trim();

    let syntax = if trimmed_lang.is_empty() {
        None
    } else {
        SYNTAX_SET
            .find_syntax_by_token(trimmed_lang)
            .or_else(|| SYNTAX_SET.find_syntax_by_extension(trimmed_lang))
            .or_else(|| SYNTAX_SET.find_syntax_by_name(trimmed_lang))
    };

    let Some(syntax) = syntax else {
        return plain(code, trimmed_lang);
    };

    let mut gen = ClassedHTMLGenerator::new_with_class_style(
        syntax,
        &SYNTAX_SET,
        ClassStyle::Spaced,
    );
    for line in LinesWithEndings::from(code) {
        if gen.parse_html_for_line_which_includes_newline(line).is_err() {
            return plain(code, trimmed_lang);
        }
    }
    let highlighted = gen.finalize();
    let lang_class = if trimmed_lang.is_empty() {
        String::new()
    } else {
        format!(" lang-{}", html_escape_attr(trimmed_lang))
    };
    format!("<pre><code class=\"code{lang_class}\">{highlighted}</code></pre>")
}

fn plain(code: &str, lang: &str) -> String {
    let lang_class = if lang.is_empty() {
        String::new()
    } else {
        format!(" lang-{}", html_escape_attr(lang))
    };
    format!(
        "<pre><code class=\"code{lang_class}\">{}</code></pre>",
        html_escape(code)
    )
}

fn html_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            _ => out.push(c),
        }
    }
    out
}

fn html_escape_attr(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}
