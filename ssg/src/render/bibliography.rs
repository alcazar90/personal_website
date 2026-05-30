//! Bibliography loading and citation preprocessing.
//!
//! Reads a per-post `.refs.yaml` sidecar file, replaces `\cite{key}` /
//! `\citep{key}` patterns in the markdown body with numbered superscript links,
//! and generates a `<section class="references">` block appended after the body.
//!
//! File format (`<post-stem>.refs.yaml`):
//! ```yaml
//! Sutton1998:
//!   author: "Sutton, R. S."
//!   title:  "Reinforcement Learning: An Introduction"
//!   year:   2018
//!   url:    "http://incompleteideas.net/book/the-book-2nd.html"
//! ```
//! Every field except `author`, `title`, and `year` is optional.

use anyhow::{Context, Result};
use regex::Regex;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct BibEntry {
    pub author: String,
    pub title: String,
    pub year: u32,
    pub url: Option<String>,
    pub journal: Option<String>,
    pub booktitle: Option<String>,
    pub note: Option<String>,
}

pub type BibMap = HashMap<String, BibEntry>;

pub fn load_bib(path: &Path) -> Result<BibMap> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading bibliography {}", path.display()))?;
    serde_yaml::from_str(&raw)
        .with_context(|| format!("parsing bibliography {}", path.display()))
}

/// Strip the `## References` heading and everything that follows it from the
/// markdown body. The SSG regenerates the section from the `.refs.yaml` data.
pub fn strip_references_section(body: &str) -> &str {
    if let Some(pos) = body.find("\n## References") {
        &body[..pos]
    } else if body.starts_with("## References") {
        ""
    } else {
        body
    }
}

/// Replace `\cite{key}` / `\citep{key}` patterns with numbered citation links.
///
/// Returns `(processed_body, ordered_keys)` where `ordered_keys` lists every
/// cited key in order of first appearance. Unknown keys are still linked (the
/// bibliography renderer emits a visible placeholder for them).
pub fn preprocess_citations(body: &str, bib: &BibMap) -> (String, Vec<String>) {
    if bib.is_empty() {
        return (body.to_string(), Vec::new());
    }

    let re = Regex::new(r"\\citep?\{([^}]+)\}").unwrap();
    let mut ordered: Vec<String> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();

    let result = re.replace_all(body, |caps: &regex::Captures| {
        let key = caps[1].to_string();
        let n = if let Some(&n) = index.get(&key) {
            n
        } else {
            let n = ordered.len() + 1;
            ordered.push(key.clone());
            index.insert(key.clone(), n);
            n
        };
        format!(
            "<sup><a href=\"#ref-{}\" class=\"cite\">[{}]</a></sup>",
            key, n
        )
    });

    (result.into_owned(), ordered)
}

/// Generate a `<section class="references">` block for all cited keys.
pub fn render_bibliography_html(bib: &BibMap, ordered_keys: &[String]) -> String {
    if ordered_keys.is_empty() {
        return String::new();
    }

    let mut html = String::from(
        "<section class=\"references\">\n\
         <h2 id=\"references\">References</h2>\n\
         <ol class=\"reference-list\">\n",
    );

    for (i, key) in ordered_keys.iter().enumerate() {
        html.push_str(&format!("<li id=\"ref-{}\" value=\"{}\">", key, i + 1));
        match bib.get(key) {
            Some(entry) => html.push_str(&format_entry(entry)),
            None => html.push_str(&format!(
                "<span class=\"cite-unknown\">[unknown citation: {}]</span>",
                key
            )),
        }
        html.push_str("</li>\n");
    }

    html.push_str("</ol>\n</section>\n");
    html
}

fn format_entry(e: &BibEntry) -> String {
    let mut s = format!("{} ({}). ", e.author, e.year);

    if let Some(url) = &e.url {
        s.push_str(&format!(
            r#"<a href="{}" target="_blank" rel="noopener">{}</a>"#,
            url, e.title
        ));
    } else {
        s.push_str(&e.title);
    }

    if let Some(j) = &e.journal {
        s.push_str(&format!(". <em>{}</em>", j));
    } else if let Some(bt) = &e.booktitle {
        s.push_str(&format!(". In <em>{}</em>", bt));
    }

    if let Some(note) = &e.note {
        s.push_str(&format!(". {}", note));
    }

    s.push('.');
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_bib() -> BibMap {
        let mut m = BibMap::new();
        m.insert(
            "foo2024".to_string(),
            BibEntry {
                author: "Foo, A.".to_string(),
                title: "A Great Paper".to_string(),
                year: 2024,
                url: Some("https://example.com".to_string()),
                journal: None,
                booktitle: None,
                note: None,
            },
        );
        m.insert(
            "bar2023".to_string(),
            BibEntry {
                author: "Bar, B.".to_string(),
                title: "Another Work".to_string(),
                year: 2023,
                url: None,
                journal: Some("Journal of Things".to_string()),
                booktitle: None,
                note: None,
            },
        );
        m
    }

    #[test]
    fn preprocess_replaces_cite_with_link() {
        let bib = simple_bib();
        let (out, keys) = preprocess_citations(r"See \cite{foo2024} for details.", &bib);
        assert!(out.contains("href=\"#ref-foo2024\""), "got: {out}");
        assert!(out.contains("[1]"), "got: {out}");
        assert_eq!(keys, vec!["foo2024"]);
    }

    #[test]
    fn preprocess_handles_citep() {
        let bib = simple_bib();
        let (out, keys) = preprocess_citations(r"(\citep{bar2023})", &bib);
        assert!(out.contains("href=\"#ref-bar2023\""), "got: {out}");
        assert_eq!(keys, vec!["bar2023"]);
    }

    #[test]
    fn preprocess_numbers_in_first_appearance_order() {
        let bib = simple_bib();
        let body = r"\cite{bar2023} and \cite{foo2024} and \cite{bar2023} again";
        let (out, keys) = preprocess_citations(body, &bib);
        assert_eq!(keys, vec!["bar2023", "foo2024"]);
        // bar2023 is [1], foo2024 is [2]
        assert!(out.contains("[1]"), "got: {out}");
        assert!(out.contains("[2]"), "got: {out}");
    }

    #[test]
    fn preprocess_empty_bib_returns_body_unchanged() {
        let bib = BibMap::new();
        let body = r"No \cite{anything} here.";
        let (out, keys) = preprocess_citations(body, &bib);
        assert_eq!(out, body);
        assert!(keys.is_empty());
    }

    #[test]
    fn render_bibliography_html_formats_entries() {
        let bib = simple_bib();
        let html = render_bibliography_html(&bib, &["foo2024".to_string(), "bar2023".to_string()]);
        assert!(html.contains("id=\"ref-foo2024\""), "got: {html}");
        assert!(html.contains("id=\"ref-bar2023\""), "got: {html}");
        assert!(html.contains("A Great Paper"), "got: {html}");
        assert!(html.contains("Another Work"), "got: {html}");
        assert!(html.contains("<em>Journal of Things</em>"), "got: {html}");
    }

    #[test]
    fn render_bibliography_html_unknown_key_shows_placeholder() {
        let bib = BibMap::new();
        let html = render_bibliography_html(&bib, &["ghost".to_string()]);
        assert!(html.contains("unknown citation: ghost"), "got: {html}");
    }

    #[test]
    fn render_bibliography_html_empty_keys_returns_empty() {
        let bib = simple_bib();
        assert_eq!(render_bibliography_html(&bib, &[]), "");
    }

    #[test]
    fn strip_references_section_trims_at_heading() {
        let body = "intro\n## References\n[1] foo";
        assert_eq!(strip_references_section(body), "intro");
    }

    #[test]
    fn strip_references_section_no_references_unchanged() {
        let body = "just text";
        assert_eq!(strip_references_section(body), "just text");
    }
}
