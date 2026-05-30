//! Markdown → HTML pipeline.
//!
//! Walks the `pulldown_cmark` event stream and intercepts:
//!   - headings → add `id=` anchors; collect TOC entries
//!   - fenced code blocks → `render::code::highlight` (syntect, class-based)
//!   - inline / display math → `render::math::inline|display` (pulldown-latex → MathML)
//!   - image tags with relative URLs → rewrite to `/posts/<slug>/<filename>`
//!
//! Bibliography: if a `<post-stem>.refs.yaml` sidecar exists, `\cite{key}` /
//! `\citep{key}` patterns are replaced with numbered superscript links, and a
//! `<section class="references">` is appended after the rendered body.
//!
//! Everything else falls through to the default HTML emitter. Errors during
//! math conversion fall back to a `<code>` block with the raw source rather
//! than panicking — a broken equation must not break the build.

pub mod bibliography;
pub mod code;
pub mod math;

use crate::content::Source;
use anyhow::Result;
use pulldown_cmark::{CodeBlockKind, CowStr, Event, Options, Parser, Tag, TagEnd};
use std::collections::HashMap;

/// One entry in the auto-generated Table of Contents.
#[derive(Debug, Clone)]
pub struct TocEntry {
    pub level: u8,
    pub text: String,
    pub id: String,
}

#[derive(Debug, Clone)]
pub struct RenderedPost {
    pub html: String,
    pub reading_time_minutes: u32,
    /// Pre-rendered `<nav class="toc">` block, or empty if the post has fewer
    /// than two headings.
    pub toc_html: String,
}

/// Render a `Source` to HTML, returning body HTML, reading time, and a
/// pre-rendered TOC nav block. No templating happens here.
pub fn render(source: &Source) -> Result<RenderedPost> {
    // --- bibliography ---
    let bib_path = source.path.with_extension("refs.yaml");
    let bib = if bib_path.exists() {
        bibliography::load_bib(&bib_path)?
    } else {
        bibliography::BibMap::new()
    };

    let body_stripped = bibliography::strip_references_section(&source.body);
    let (body_with_cites, ordered_keys) = bibliography::preprocess_citations(body_stripped, &bib);

    // --- markdown → HTML ---
    let mut options = Options::empty();
    options.insert(Options::ENABLE_TABLES);
    options.insert(Options::ENABLE_FOOTNOTES);
    options.insert(Options::ENABLE_STRIKETHROUGH);
    options.insert(Options::ENABLE_TASKLISTS);
    options.insert(Options::ENABLE_MATH);

    // Preprocess math: normalize legacy MathJax delimiters (\(...\), \[...\])
    // into $/$$ form, and strip unsupported LaTeX envs (equation, label).
    let preprocessed = math::preprocess_source(&body_with_cites);
    let parser = Parser::new_ext(&preprocessed, options);
    let mut toc: Vec<TocEntry> = Vec::new();
    let events = transform_events(parser, &source.slug, &mut toc);

    let mut html = String::new();
    pulldown_cmark::html::push_html(&mut html, events.into_iter());

    // --- append bibliography section ---
    let bib_html = bibliography::render_bibliography_html(&bib, &ordered_keys);
    if !bib_html.is_empty() {
        // Add a synthetic TOC entry so References appears in the nav.
        toc.push(TocEntry {
            level: 2,
            text: "References".to_string(),
            id: "references".to_string(),
        });
        html.push_str(&bib_html);
    }

    // --- build TOC nav ---
    let toc_html = if toc.len() >= 2 {
        build_toc_html(&toc)
    } else {
        String::new()
    };

    Ok(RenderedPost {
        html,
        reading_time_minutes: reading_time(&source.body),
        toc_html,
    })
}

fn transform_events<'a>(
    parser: Parser<'a>,
    slug: &str,
    toc: &mut Vec<TocEntry>,
) -> Vec<Event<'a>> {
    let mut out: Vec<Event<'a>> = Vec::new();
    let mut iter = parser;
    // Track used heading IDs to deduplicate (e.g. two "Overview" headings).
    let mut heading_ids: HashMap<String, usize> = HashMap::new();

    while let Some(ev) = iter.next() {
        match ev {
            // ── Headings ────────────────────────────────────────────────────
            Event::Start(Tag::Heading { level, id, .. }) => {
                // Drain inner events, collecting plain text for the anchor.
                let mut inner: Vec<Event<'a>> = Vec::new();
                let mut plain = String::new();
                for e in iter.by_ref() {
                    if matches!(e, Event::End(TagEnd::Heading(_))) {
                        break;
                    }
                    if let Event::Text(ref t) = e {
                        plain.push_str(t);
                    }
                    inner.push(e);
                }

                let base_id = if let Some(eid) = id.filter(|s| !s.is_empty()) {
                    eid.to_string()
                } else {
                    heading_slug(&plain)
                };
                let anchor = deduplicate_id(&base_id, &mut heading_ids);

                let lvl = level as u8;
                toc.push(TocEntry {
                    level: lvl,
                    text: plain,
                    id: anchor.clone(),
                });

                out.push(Event::Html(CowStr::from(format!(
                    "<h{lvl} id=\"{anchor}\">"
                ))));
                out.extend(inner);
                out.push(Event::Html(CowStr::from(format!("</h{lvl}>"))));
            }

            // ── Code blocks ─────────────────────────────────────────────────
            Event::Start(Tag::CodeBlock(CodeBlockKind::Fenced(lang))) => {
                let mut buf = String::new();
                for inner in iter.by_ref() {
                    match inner {
                        Event::Text(t) => buf.push_str(&t),
                        Event::End(TagEnd::CodeBlock) => break,
                        _ => {}
                    }
                }
                let html = code::highlight(&buf, &lang);
                out.push(Event::Html(CowStr::from(html)));
            }
            Event::Start(Tag::CodeBlock(CodeBlockKind::Indented)) => {
                let mut buf = String::new();
                for inner in iter.by_ref() {
                    match inner {
                        Event::Text(t) => buf.push_str(&t),
                        Event::End(TagEnd::CodeBlock) => break,
                        _ => {}
                    }
                }
                let html = code::highlight(&buf, "");
                out.push(Event::Html(CowStr::from(html)));
            }

            // ── Math ─────────────────────────────────────────────────────────
            Event::InlineMath(latex) => {
                // `\_` and `\*` in source were markdown escapes (to avoid italic
                // interpretation); LaTeX wants the bare token (subscript / *).
                let cleaned = sanitize_math_escapes(&latex);
                let html = match math::inline(&cleaned) {
                    Ok(mathml) => format!("<span class=\"math inline\">{mathml}</span>"),
                    Err(_) => format!("<code>${}$</code>", html_escape(&latex)),
                };
                out.push(Event::Html(CowStr::from(html)));
            }
            Event::DisplayMath(latex) => {
                let cleaned = sanitize_math_escapes(&latex);
                let html = match math::display(&cleaned) {
                    Ok(mathml) => format!("<div class=\"math display\">{mathml}</div>"),
                    Err(_) => format!("<code>$${}$$</code>", html_escape(&latex)),
                };
                out.push(Event::Html(CowStr::from(html)));
            }

            // ── Images ───────────────────────────────────────────────────────
            Event::Start(Tag::Image {
                link_type,
                dest_url,
                title,
                id,
            }) => {
                let rewritten = rewrite_image_url(&dest_url, slug);
                out.push(Event::Start(Tag::Image {
                    link_type,
                    dest_url: CowStr::from(rewritten),
                    title,
                    id,
                }));
            }

            other => out.push(other),
        }
    }

    out
}

// ── TOC helpers ──────────────────────────────────────────────────────────────

/// Build a `<nav class="toc">` block from collected heading entries.
fn build_toc_html(toc: &[TocEntry]) -> String {
    let mut html =
        String::from("<nav class=\"toc\">\n<details>\n<summary>Table of Contents</summary>\n");

    // Stack tracks the heading level of each open <ul>/<li> layer. The
    // sentinel 0 means "root" and never corresponds to a real heading.
    let mut stack: Vec<u8> = vec![0];

    for entry in toc {
        let level = entry.level;
        let text = html_escape(&entry.text);

        // Close any open layers that are deeper than the current heading.
        while stack.last().copied().map_or(false, |top| level < top) {
            html.push_str("</li>\n</ul>\n");
            stack.pop();
        }

        if stack.last().copied() == Some(level) {
            // Same nesting level: close the previous <li> (if not the root).
            if stack.len() > 1 {
                html.push_str("</li>\n");
            }
        } else {
            // Going deeper: open a new nested list.
            html.push_str("<ul>\n");
            stack.push(level);
        }

        html.push_str(&format!(
            "<li><a href=\"#{id}\">{text}</a>",
            id = entry.id,
            text = text
        ));
    }

    // Close all remaining open layers.
    while stack.len() > 1 {
        html.push_str("</li>\n</ul>\n");
        stack.pop();
    }

    html.push_str("</details>\n</nav>\n");
    html
}

/// Slug a heading's plain text into an HTML `id`-safe anchor string.
///
/// Rules: lowercase, collapse non-alphanumeric runs to `-`, trim leading/
/// trailing `-`. Unicode letters and digits are preserved.
fn heading_slug(text: &str) -> String {
    let lower = text.to_lowercase();
    let mut out = String::with_capacity(lower.len());
    let mut prev_sep = true; // suppress a leading `-`
    for c in lower.chars() {
        if c.is_alphanumeric() {
            out.push(c);
            prev_sep = false;
        } else if !prev_sep {
            out.push('-');
            prev_sep = true;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    out
}

/// If `base_id` already exists in `used`, append `-2`, `-3`, … until unique.
fn deduplicate_id(base_id: &str, used: &mut HashMap<String, usize>) -> String {
    let count = used.entry(base_id.to_string()).or_insert(0);
    *count += 1;
    if *count == 1 {
        base_id.to_string()
    } else {
        format!("{base_id}-{count}")
    }
}

// ── Misc helpers ─────────────────────────────────────────────────────────────

/// Rewrite a relative image path to its deployed location at
/// `/posts/<slug>/<filename>`. Absolute URLs, schemed URLs, and root-
/// relative paths are returned unchanged.
fn rewrite_image_url(dest_url: &str, slug: &str) -> String {
    if dest_url.is_empty() {
        return dest_url.to_string();
    }
    if dest_url.starts_with('/')
        || dest_url.starts_with("http://")
        || dest_url.starts_with("https://")
        || dest_url.starts_with("data:")
        || dest_url.starts_with("mailto:")
        || dest_url.contains("://")
    {
        return dest_url.to_string();
    }
    let trimmed = dest_url.strip_prefix("./").unwrap_or(dest_url);
    format!("/posts/{slug}/{trimmed}")
}

/// Estimate reading time at 220 wpm, rounded to the nearest minute (min 1).
fn reading_time(body: &str) -> u32 {
    let words = body.split_whitespace().count() as f32;
    if words == 0.0 {
        return 1;
    }
    let minutes = (words / 220.0).round() as u32;
    minutes.max(1)
}

/// Strip markdown escapes that don't belong in LaTeX math. The original
/// posts used `\_` to prevent markdown from interpreting underscores as
/// italics — but the content past pulldown-cmark is meant for LaTeX, where
/// `\_` means literal underscore (not subscript). Same story for `\*`.
fn sanitize_math_escapes(latex: &str) -> String {
    latex.replace("\\_", "_").replace("\\*", "*")
}

pub fn html_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(c),
        }
    }
    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content::Frontmatter;
    use std::path::PathBuf;

    fn make_source(body: &str, slug: &str) -> Source {
        Source {
            path: PathBuf::from("test.md"),
            slug: slug.to_string(),
            frontmatter: Frontmatter::default(),
            body: body.to_string(),
        }
    }

    #[test]
    fn renders_python_code_block_with_syntect_classes() {
        let md = "```python\ndef greet(name):\n    return f\"hi, {name}\"\n```\n";
        let out = render(&make_source(md, "post")).unwrap();
        assert!(out.html.contains("<pre>"), "html: {}", out.html);
        assert!(
            out.html.contains("<span class=\""),
            "expected highlighted spans; got: {}",
            out.html
        );
    }

    #[test]
    fn renders_inline_math_to_mathml() {
        let md = "Here is $a + b$ inline.\n";
        let out = render(&make_source(md, "post")).unwrap();
        assert!(
            out.html.contains("<math"),
            "expected <math> element; got: {}",
            out.html
        );
        assert!(
            out.html.contains("math inline"),
            "expected inline wrapper; got: {}",
            out.html
        );
    }

    #[test]
    fn renders_display_math_to_mathml() {
        let md = "Behold:\n\n$$\\sum x$$\n";
        let out = render(&make_source(md, "post")).unwrap();
        assert!(
            out.html.contains("<math"),
            "expected <math> element; got: {}",
            out.html
        );
        assert!(
            out.html.contains("math display"),
            "expected display wrapper; got: {}",
            out.html
        );
    }

    #[test]
    fn code_fence_without_language_does_not_crash() {
        let md = "```\njust some text\n```\n";
        let out = render(&make_source(md, "post")).unwrap();
        assert!(out.html.contains("<pre>"));
        assert!(out.html.contains("just some text"));
    }

    #[test]
    fn unknown_language_falls_back_cleanly() {
        let md = "```not-a-real-language\nfoo bar\n```\n";
        let out = render(&make_source(md, "post")).unwrap();
        assert!(out.html.contains("<pre>"));
        assert!(out.html.contains("foo bar"));
    }

    #[test]
    fn relative_image_url_is_rewritten() {
        let md = "![alt](thumb.png)\n";
        let out = render(&make_source(md, "my-post")).unwrap();
        assert!(
            out.html.contains("/posts/my-post/thumb.png"),
            "expected rewritten image src; got: {}",
            out.html
        );
    }

    #[test]
    fn absolute_image_url_is_preserved() {
        let md = "![alt](https://example.com/img.png)\n";
        let out = render(&make_source(md, "post")).unwrap();
        assert!(out.html.contains("https://example.com/img.png"));
        assert!(!out.html.contains("/posts/post/https"));
    }

    #[test]
    fn reading_time_minimum_is_one() {
        let md = "tiny.";
        let out = render(&make_source(md, "post")).unwrap();
        assert_eq!(out.reading_time_minutes, 1);
    }

    #[test]
    fn reading_time_scales_with_words() {
        let words: Vec<&str> = std::iter::repeat("lorem").take(660).collect();
        let body = words.join(" ");
        let out = render(&make_source(&body, "post")).unwrap();
        assert_eq!(out.reading_time_minutes, 3);
    }

    #[test]
    fn malformed_math_falls_back_to_code_block() {
        let md = "Bad: $\\frac{1$ here.\n";
        let out = render(&make_source(md, "post")).unwrap();
        assert!(!out.html.is_empty());
    }

    #[test]
    fn headings_get_id_anchors() {
        let md = "## Hello World\n\nSome text.\n\n### Sub Section\n\nMore text.\n";
        let out = render(&make_source(md, "post")).unwrap();
        assert!(
            out.html.contains(r#"id="hello-world""#),
            "got: {}",
            out.html
        );
        assert!(
            out.html.contains(r#"id="sub-section""#),
            "got: {}",
            out.html
        );
    }

    #[test]
    fn toc_generated_for_multi_heading_post() {
        let md = "## Alpha\n\ntext.\n\n## Beta\n\nmore.\n";
        let out = render(&make_source(md, "post")).unwrap();
        assert!(
            out.toc_html.contains("class=\"toc\""),
            "expected toc nav; got: {}",
            out.toc_html
        );
        assert!(out.toc_html.contains("#alpha"), "got: {}", out.toc_html);
        assert!(out.toc_html.contains("#beta"), "got: {}", out.toc_html);
    }

    #[test]
    fn toc_empty_for_single_heading() {
        let md = "## Only One\n\ntext.\n";
        let out = render(&make_source(md, "post")).unwrap();
        assert!(
            out.toc_html.is_empty(),
            "expected empty toc for single heading; got: {}",
            out.toc_html
        );
    }

    #[test]
    fn heading_slug_lowercases_and_replaces_separators() {
        assert_eq!(heading_slug("Hello World"), "hello-world");
        assert_eq!(
            heading_slug("Vanilla Policy Gradient, aka REINFORCE"),
            "vanilla-policy-gradient-aka-reinforce"
        );
        assert_eq!(heading_slug("  leading spaces  "), "leading-spaces");
    }

    #[test]
    fn duplicate_heading_ids_get_suffix() {
        let mut used = HashMap::new();
        assert_eq!(deduplicate_id("intro", &mut used), "intro");
        assert_eq!(deduplicate_id("intro", &mut used), "intro-2");
        assert_eq!(deduplicate_id("intro", &mut used), "intro-3");
    }
}
