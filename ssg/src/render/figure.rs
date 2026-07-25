//! Figure numbering, cross-references, and math rendering inside `<figure>`
//! blocks.
//!
//! `<figure>` and `<figcaption>` are both CommonMark "type 6" HTML block
//! tags, so pulldown-cmark swallows an entire `<figure>...</figure>` element
//! as one opaque raw-HTML block and never runs inline parsing — including
//! math — on its contents (see `render_math_in_figures`). This module also
//! mirrors the `\label`/`\ref`/`\eqref` equation system in `math.rs` for
//! figures, using their own independent counter: every `<figure id="...">`
//! is auto-numbered in document order, `\figref{id}` resolves to a linked
//! "Figure N", and a "Figure N. " label is inserted at the start of each
//! `<figcaption>`.

use super::{html_escape, math, sanitize_math_escapes};
use regex::Regex;
use std::collections::HashMap;
use std::sync::OnceLock;

/// Scan for `<figure ... id="key" ...>` occurrences in document order and
/// assign sequential figure numbers (1, 2, 3, …), independent of equation
/// numbering (figures and equations are separate LaTeX counters).
pub fn collect_figure_labels(body: &str) -> HashMap<String, u32> {
    static FIGURE_RE: OnceLock<Regex> = OnceLock::new();
    let re = FIGURE_RE
        .get_or_init(|| Regex::new(r#"(?is)<figure\b[^>]*\bid="([^"]+)"[^>]*>"#).unwrap());
    let mut map = HashMap::new();
    let mut n = 1u32;
    for cap in re.captures_iter(body) {
        let key = cap[1].to_string();
        map.entry(key).or_insert_with(|| {
            let v = n;
            n += 1;
            v
        });
    }
    map
}

/// Replace `\figref{key}` with a link to the figure, e.g.
/// `<a href="#key">Figure 3</a>`. Unknown keys render as `[?:key]`,
/// mirroring the `\ref`/`\eqref` convention in `math.rs`, so a broken
/// reference is immediately visible rather than silently disappearing.
pub fn replace_figrefs(body: &str, labels: &HashMap<String, u32>) -> String {
    static FIGREF_RE: OnceLock<Regex> = OnceLock::new();
    let re = FIGREF_RE.get_or_init(|| Regex::new(r"\\figref\{([^}]+)\}").unwrap());
    re.replace_all(body, |caps: &regex::Captures| {
        let key = &caps[1];
        match labels.get(key) {
            Some(n) => format!(r##"<a href="#{key}">Figure {n}</a>"##),
            None => format!("[?:{key}]"),
        }
    })
    .into_owned()
}

/// Insert an auto-numbered "Figure N. " label right after each figure's
/// `<figcaption>` opening tag (matched via the enclosing `<figure id="key">`).
pub fn number_figcaptions(body: &str, labels: &HashMap<String, u32>) -> String {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        Regex::new(r#"(?is)(<figure\b[^>]*\bid="([^"]+)"[^>]*>.*?<figcaption\b[^>]*>)"#).unwrap()
    });
    re.replace_all(body, |caps: &regex::Captures| {
        let prefix = &caps[1];
        let key = &caps[2];
        match labels.get(key) {
            Some(n) => format!("{prefix}Figure {n}. "),
            None => prefix.to_string(),
        }
    })
    .into_owned()
}

/// Render `$...$` / `$$...$$` math spans found inside `<figure>...</figure>`
/// blocks to MathML. Must run after `math::preprocess_source`'s delimiter
/// normalization (it only understands the `$` / `$$` form), and before the
/// markdown parser sees the source — pulldown-cmark treats `<figure>` as raw
/// HTML and would otherwise emit any `$...$` inside it as literal text
/// instead of a math span.
pub fn render_math_in_figures(body: &str) -> String {
    static FIGURE_BLOCK_RE: OnceLock<Regex> = OnceLock::new();
    let re = FIGURE_BLOCK_RE.get_or_init(|| Regex::new(r"(?s)<figure\b.*?</figure>").unwrap());
    re.replace_all(body, |caps: &regex::Captures| render_math_spans(&caps[0]))
        .into_owned()
}

fn render_math_spans(fragment: &str) -> String {
    static DISPLAY_RE: OnceLock<Regex> = OnceLock::new();
    static INLINE_RE: OnceLock<Regex> = OnceLock::new();
    let display_re = DISPLAY_RE.get_or_init(|| Regex::new(r"(?s)\$\$(.+?)\$\$").unwrap());
    let inline_re = INLINE_RE.get_or_init(|| Regex::new(r"\$([^$\n]+?)\$").unwrap());

    let with_display = display_re.replace_all(fragment, |caps: &regex::Captures| {
        let cleaned = sanitize_math_escapes(&caps[1]);
        match math::display(&cleaned) {
            Ok(mathml) => format!(r#"<div class="math display">{mathml}</div>"#),
            Err(_) => format!("<code>$${}$$</code>", html_escape(&caps[1])),
        }
    });

    inline_re
        .replace_all(&with_display, |caps: &regex::Captures| {
            let cleaned = sanitize_math_escapes(&caps[1]);
            match math::inline(&cleaned) {
                Ok(mathml) => format!(r#"<span class="math inline">{mathml}</span>"#),
                Err(_) => format!("<code>${}$</code>", html_escape(&caps[1])),
            }
        })
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collect_labels_numbers_in_document_order() {
        let body = r#"<figure id="fig:b"></figure> text <figure id="fig:a"></figure>"#;
        let labels = collect_figure_labels(body);
        assert_eq!(labels.get("fig:b"), Some(&1));
        assert_eq!(labels.get("fig:a"), Some(&2));
    }

    #[test]
    fn collect_labels_ignores_duplicates() {
        let body = r#"<figure id="fig:a"></figure> <figure id="fig:a"></figure>"#;
        let labels = collect_figure_labels(body);
        assert_eq!(labels.len(), 1);
        assert_eq!(labels.get("fig:a"), Some(&1));
    }

    #[test]
    fn figref_resolves_to_linked_number() {
        let mut labels = HashMap::new();
        labels.insert("fig:a".to_string(), 3u32);
        let out = replace_figrefs(r"See \figref{fig:a}.", &labels);
        assert!(out.contains(r##"<a href="#fig:a">Figure 3</a>"##), "got: {out}");
    }

    #[test]
    fn figref_unknown_key_shows_broken_marker() {
        let out = replace_figrefs(r"See \figref{fig:missing}.", &HashMap::new());
        assert!(out.contains("[?:fig:missing]"), "got: {out}");
    }

    #[test]
    fn number_figcaptions_inserts_prefix_per_figure() {
        let body = concat!(
            r#"<figure id="fig:a"><img src="a.png"><figcaption>Alpha</figcaption></figure>"#,
            r#"<figure id="fig:b"><img src="b.png"><figcaption>Beta</figcaption></figure>"#,
        );
        let labels = collect_figure_labels(body);
        let out = number_figcaptions(body, &labels);
        assert!(out.contains("<figcaption>Figure 1. Alpha"), "got: {out}");
        assert!(out.contains("<figcaption>Figure 2. Beta"), "got: {out}");
    }

    #[test]
    fn render_math_in_figures_converts_inline_math_to_mathml() {
        let body = r#"<figure id="fig:a"><figcaption>a $x + y$ b</figcaption></figure>"#;
        let out = render_math_in_figures(body);
        assert!(out.contains("<math"), "expected MathML, got: {out}");
        assert!(!out.contains('$'), "literal $ leaked through: {out}");
    }

    #[test]
    fn render_math_in_figures_converts_display_math_to_mathml() {
        let body = r#"<figure id="fig:a"><figcaption>$$x = y$$</figcaption></figure>"#;
        let out = render_math_in_figures(body);
        assert!(out.contains(r#"class="math display""#), "got: {out}");
        assert!(out.contains("<math"), "expected MathML, got: {out}");
    }

    #[test]
    fn render_math_in_figures_leaves_math_outside_figures_untouched() {
        let body = "Some $x + y$ text outside a figure.";
        let out = render_math_in_figures(body);
        assert_eq!(out, body);
    }

    #[test]
    fn render_math_in_figures_handles_escaped_underscore() {
        let body = r#"<figure id="fig:a"><figcaption>$\pi\_{\theta}$</figcaption></figure>"#;
        let out = render_math_in_figures(body);
        assert!(out.contains("<math"), "got: {out}");
        assert!(!out.contains('$'), "got: {out}");
    }
}
