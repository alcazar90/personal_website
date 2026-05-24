//! Markdown → HTML pipeline.
//!
//! Walks the `pulldown_cmark` event stream and intercepts:
//!   - fenced code blocks → `render::code::highlight` (syntect, class-based)
//!   - inline / display math → `render::math::inline|display` (pulldown-latex → MathML)
//!   - image tags with relative URLs → rewrite to `/posts/<slug>/<filename>`
//!
//! Everything else falls through to the default HTML emitter. Errors during
//! math conversion fall back to a `<code>` block with the raw source rather
//! than panicking — a broken equation must not break the build.

pub mod code;
pub mod math;

use crate::content::Source;
use anyhow::Result;
use pulldown_cmark::{CodeBlockKind, CowStr, Event, Options, Parser, Tag, TagEnd};

#[derive(Debug, Clone)]
pub struct RenderedPost {
    pub html: String,
    pub reading_time_minutes: u32,
}

/// Render a `Source` to HTML, returning the body HTML and an estimated
/// reading time. No templating happens here — that's a downstream concern.
pub fn render(source: &Source) -> Result<RenderedPost> {
    let mut options = Options::empty();
    options.insert(Options::ENABLE_TABLES);
    options.insert(Options::ENABLE_FOOTNOTES);
    options.insert(Options::ENABLE_STRIKETHROUGH);
    options.insert(Options::ENABLE_TASKLISTS);
    options.insert(Options::ENABLE_MATH);

    // Preprocess math: normalize legacy MathJax delimiters (\(...\), \[...\])
    // into $/$$ form, and strip unsupported LaTeX envs (equation, label).
    let preprocessed = math::preprocess_source(&source.body);
    let parser = Parser::new_ext(&preprocessed, options);
    let events = transform_events(parser, &source.slug);

    let mut html = String::new();
    pulldown_cmark::html::push_html(&mut html, events.into_iter());

    let reading_time_minutes = reading_time(&source.body);

    Ok(RenderedPost {
        html,
        reading_time_minutes,
    })
}

fn transform_events<'a>(parser: Parser<'a>, slug: &str) -> Vec<Event<'a>> {
    let mut out: Vec<Event<'a>> = Vec::new();
    let mut iter = parser;

    while let Some(ev) = iter.next() {
        match ev {
            Event::Start(Tag::CodeBlock(CodeBlockKind::Fenced(lang))) => {
                // Collect text until the matching End. Nested code blocks
                // aren't possible in CommonMark, so a flat buffer is fine.
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
                // Treat indented code blocks identically to plain fenced blocks
                // (no language metadata available).
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
    // Strip any leading `./` for tidiness.
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
        // syntect with ClassedHTMLGenerator emits <span class="..."> tokens.
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
        // 660 / 220 = 3
        assert_eq!(out.reading_time_minutes, 3);
    }

    #[test]
    fn malformed_math_falls_back_to_code_block() {
        // Unmatched brace — pulldown-latex should error and we should
        // emit a <code> fallback rather than panic.
        let md = "Bad: $\\frac{1$ here.\n";
        let out = render(&make_source(md, "post")).unwrap();
        // Either rendered as mathml (lenient parser) or fell back to code —
        // both are acceptable; what's NOT acceptable is a panic / build crash.
        assert!(!out.html.is_empty());
    }
}
