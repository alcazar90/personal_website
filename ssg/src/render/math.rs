//! LaTeX → MathML rendering via `pulldown-latex`.
//!
//! Both `inline` and `display` produce a MathML string with no wrapping
//! element; the caller wraps in `<span class="math inline">` or
//! `<div class="math display">` so the surrounding CSS knows the context.
//!
//! Also exports `preprocess_source` — a markdown-level pass that:
//!   1. Collects `\label{key}` in document order → sequential equation numbers.
//!   2. Replaces `\ref{key}` / `\eqref{key}` in body text with links to that
//!      equation, e.g. `<a href="#key">N</a>` / `<a href="#key">(N)</a>`.
//!   3. Wraps standalone `\begin{equation}…\end{equation}` blocks (those NOT
//!      already inside `\\[…\\]`) in `$$…$$` so they render as display math.
//!   4. Normalizes legacy MathJax delimiters (`\\(…\\)`, `\\[…\\]`) → `$…$` / `$$…$$`.
//!      Inline captures are trimmed so pulldown-cmark's math parser isn't
//!      confused by `$ content $` (leading space invalidates the span).
//!   5. Strips `\begin{equation}` / `\end{equation}` markers, which
//!      pulldown-latex doesn't understand. `\label{…}` is deliberately left
//!      in place — it rides along inside the `$$…$$` text all the way to
//!      `render::transform_events`'s `DisplayMath` handler (see
//!      `extract_label`), which strips it there and uses the key as the
//!      `id` on the rendered `<div class="math display">`, so `\ref`/`\eqref`
//!      links above have somewhere to jump to.
//!   6. Isolates bare `<br>` lines with a trailing blank line so they can't
//!      swallow adjacent `$$…$$` blocks into a raw HTML block (see
//!      `isolate_bare_br_tags`).

use anyhow::{anyhow, Result};
use pulldown_latex::config::RenderConfig;
use pulldown_latex::mathml::push_mathml;
use pulldown_latex::{Parser, Storage};
use regex::Regex;
use std::collections::HashMap;
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
pub fn preprocess_source(body: &str) -> String {
    let body = isolate_bare_br_tags(body);
    let labels = collect_equation_labels(&body);
    let s = replace_refs(&body, &labels);
    let s = wrap_standalone_equations(&s);
    let s = normalize_delimiters(&s);
    let s = fix_display_math_newlines(&s);
    strip_unsupported_envs(&s)
}

// ── Bare `<br>` isolation ─────────────────────────────────────────────────────

/// Ensure a standalone `<br>` (or `<br/>` / `<br />`) line is followed by a
/// blank line.
///
/// CommonMark's HTML-block rule 6 lets a block-level tag like `<br>`
/// interrupt a paragraph and start a raw HTML block; that block then
/// swallows every subsequent non-blank line as literal text until the next
/// blank line — including `$$…$$` math, which never reaches pulldown-cmark's
/// math extension as a result (it just falls through as literal `$$` text,
/// and any lone `~` inside the LaTeX can even get misparsed as GFM
/// strikethrough). Some migrated posts use `<br>` as a bare spacer between
/// display-math blocks with no blank line in between, so we insert one to
/// keep each `<br>` an isolated one-line HTML block.
fn isolate_bare_br_tags(body: &str) -> String {
    static BR_RE: OnceLock<Regex> = OnceLock::new();
    let br_re = BR_RE.get_or_init(|| Regex::new(r"(?i)^\s*<br\s*/?>\s*$").unwrap());

    let had_trailing_newline = body.ends_with('\n');
    let lines: Vec<&str> = body.lines().collect();
    let mut out = String::with_capacity(body.len() + 16);
    for (i, line) in lines.iter().enumerate() {
        out.push_str(line);
        out.push('\n');
        let next_is_blank = lines.get(i + 1).map_or(true, |l| l.trim().is_empty());
        if br_re.is_match(line) && !next_is_blank {
            out.push('\n');
        }
    }
    if !had_trailing_newline {
        out.pop();
    }
    out
}

// ── Delimiter normalisation ───────────────────────────────────────────────────

fn normalize_delimiters(body: &str) -> String {
    // Source uses `\\(...\\)` and `\\[...\\]` — TWO backslashes each.
    static DISPLAY_RE: OnceLock<Regex> = OnceLock::new();
    static INLINE_RE: OnceLock<Regex> = OnceLock::new();
    let display =
        DISPLAY_RE.get_or_init(|| Regex::new(r"(?s)\\\\\[(.*?)\\\\\]").unwrap());
    let inline =
        INLINE_RE.get_or_init(|| Regex::new(r"\\\\\((.*?)\\\\\)").unwrap());

    let body = display.replace_all(body, |caps: &regex::Captures| {
        let single_line: String = caps[1]
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        format!("$$ {} $$", single_line)
    });
    // Trim the captured content: pulldown-cmark's math extension rejects
    // `$ content $` (space immediately after `$` = not an opening delimiter).
    let body = inline.replace_all(&body, |caps: &regex::Captures| {
        format!("${}$", caps[1].trim())
    });
    body.into_owned()
}

// ── Strip unsupported environments ───────────────────────────────────────────

fn strip_unsupported_envs(body: &str) -> String {
    static ENV_RE: OnceLock<Regex> = OnceLock::new();
    let env = ENV_RE
        .get_or_init(|| Regex::new(r"\\(?:begin|end)\{equation\*?\}").unwrap());
    env.replace_all(body, "").into_owned()
}

// ── Per-equation label extraction ────────────────────────────────────────────

/// Extract a `\label{key}` from a single display-math LaTeX fragment (the
/// text captured by `pulldown_cmark::Event::DisplayMath`), returning the
/// cleaned LaTeX (label removed, since pulldown-latex doesn't understand
/// `\label`) and the label key, if one was present. Only the first `\label`
/// is consumed — a display math block should have at most one.
pub fn extract_label(latex: &str) -> (String, Option<String>) {
    static LABEL_RE: OnceLock<Regex> = OnceLock::new();
    let re = LABEL_RE.get_or_init(|| Regex::new(r"\\label\{([^}]+)\}").unwrap());
    let mut key = None;
    let cleaned = re.replace(latex, |c: &regex::Captures| {
        key = Some(c[1].to_string());
        ""
    });
    (cleaned.into_owned(), key)
}

// ── Top-level newline fix for $$ blocks ──────────────────────────────────────

/// Fix `\\` line-break commands inside `$$…$$` display math blocks.
///
/// pulldown-latex rejects `\\` at the top level of display math (outside any
/// `\begin{…}…\end{…}` environment) with "new line command not allowed in
/// current environment". We handle two sub-cases:
///
/// 1. **Trailing `\\`** — strip it; it's cosmetic noise after the last expression.
/// 2. **Mid-block `\\`** — wrap the whole block content in
///    `\begin{aligned}…\end{aligned}`, which does accept `\\`.
fn fix_display_math_newlines(body: &str) -> String {
    static DISPLAY_RE: OnceLock<Regex> = OnceLock::new();
    let re = DISPLAY_RE.get_or_init(|| Regex::new(r"(?s)\$\$(.*?)\$\$").unwrap());

    re.replace_all(body, |caps: &regex::Captures| {
        let inner = &caps[1];

        // Strip trailing \\  (possibly followed only by whitespace).
        let trimmed = inner.trim_end();
        let (s, stripped_newline) = if trimmed.ends_with("\\\\") {
            (trimmed[..trimmed.len() - 2].trim_end(), true)
        } else {
            (trimmed, false)
        };

        if has_top_level_double_backslash(s) {
            // Mid-block \\: wrap in aligned so the newline command is valid.
            format!("$$\\begin{{aligned}}{}\\end{{aligned}}$$", s)
        } else if stripped_newline {
            // Only a trailing \\ was removed; restore surrounding newlines.
            // `s` may still carry the ORIGINAL leading newline (only
            // trim_end was applied above) — trim it here too, otherwise the
            // "\n" we add below stacks with it into "\n\n" (a blank line),
            // which splits the CommonMark paragraph in two and makes
            // pulldown-cmark fail to recognize the $$ pair as math at all.
            format!("$$\n{}\n$$", s.trim())
        } else {
            // Nothing to change — preserve original spacing exactly.
            format!("$${}$$", inner)
        }
    })
    .into_owned()
}

/// Return `true` if `s` contains `\\` (two backslashes) at "top level" —
/// i.e. not nested inside any `\begin{…}…\end{…}` environment.
fn has_top_level_double_backslash(s: &str) -> bool {
    let mut depth: i32 = 0;
    let mut i = 0;
    while i < s.len() {
        if s[i..].starts_with("\\begin{") {
            depth += 1;
            i += 7;
        } else if s[i..].starts_with("\\end{") {
            depth -= 1;
            i += 5;
        } else if depth == 0 && s[i..].starts_with("\\\\") {
            return true;
        } else {
            // Advance by one Unicode scalar so we never land inside a multi-byte char.
            i += s[i..].chars().next().map_or(1, |c| c.len_utf8());
        }
    }
    false
}

// ── Equation labels & cross-references ───────────────────────────────────────

/// Scan `body` for every `\label{key}` in document order and assign sequential
/// numbers (1, 2, 3, …). The first occurrence of each key wins; duplicates are
/// ignored.
///
/// Exposed publicly so `render::mod` can re-derive the same numbering after
/// `preprocess_source` has run (the `\label{…}` markers ride through every
/// preprocessing step untouched — see the module docs) and use it to print a
/// visible "(N)" beside each labelled equation, matching the number that
/// `\ref`/`\eqref` links already point to.
pub fn collect_equation_labels(body: &str) -> HashMap<String, u32> {
    static LABEL_RE: OnceLock<Regex> = OnceLock::new();
    let re = LABEL_RE.get_or_init(|| Regex::new(r"\\label\{([^}]+)\}").unwrap());
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

/// Replace `\eqref{key}` → `<a href="#key">(N)</a>` and `\ref{key}` →
/// `<a href="#key">N</a>` throughout `body`. The link target is the `id`
/// that `render::transform_events` attaches to the equation's rendered
/// `<div class="math display">` (see `extract_label`), so clicking jumps
/// straight to where the equation is defined.
///
/// `\s*` in the `\ref` pattern absorbs the occasional stray space before `{`
/// (e.g. `\ref {eqn:foo}`) that some LaTeX editors emit.
///
/// Unknown keys render as `[?:key]` so broken references are immediately
/// visible in the output rather than silently disappearing.
fn replace_refs(body: &str, labels: &HashMap<String, u32>) -> String {
    static EQREF_RE: OnceLock<Regex> = OnceLock::new();
    static REF_RE: OnceLock<Regex> = OnceLock::new();
    let eqref = EQREF_RE.get_or_init(|| Regex::new(r"\\eqref\{([^}]+)\}").unwrap());
    let rref  = REF_RE.get_or_init(|| Regex::new(r"\\ref\s*\{([^}]+)\}").unwrap());

    let s = eqref.replace_all(body, |c: &regex::Captures| {
        let key = &c[1];
        match labels.get(key) {
            Some(n) => format!(r##"<a href="#{key}">({n})</a>"##),
            None => format!("[?:{key}]"),
        }
    });
    let s = rref.replace_all(&s, |c: &regex::Captures| {
        let key = &c[1];
        match labels.get(key) {
            Some(n) => format!(r##"<a href="#{key}">{n}</a>"##),
            None => format!("[?:{key}]"),
        }
    });
    s.into_owned()
}

// ── Standalone equation wrapping ─────────────────────────────────────────────

/// Wrap standalone `\begin{equation}…\end{equation}` blocks that are NOT
/// already inside `\\[…\\]` in `$$…$$` so pulldown-cmark treats them as
/// display math.
///
/// Uses a placeholder scheme: every `\\[…\\]` block is swapped out for a
/// sentinel before the standalone scan and restored afterwards, preventing
/// double-wrapping of blocks that are already guarded by display delimiters.
fn wrap_standalone_equations(body: &str) -> String {
    static DISPLAY_RE: OnceLock<Regex> = OnceLock::new();
    static STANDALONE_RE: OnceLock<Regex> = OnceLock::new();
    let display_re = DISPLAY_RE
        .get_or_init(|| Regex::new(r"(?s)\\\\\[(.*?)\\\\\]").unwrap());
    let standalone_re = STANDALONE_RE
        .get_or_init(|| Regex::new(r"(?s)\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}").unwrap());

    // Step 1 — hide \\[…\\] blocks behind sentinels.
    let mut saved: Vec<String> = Vec::new();
    let guarded = display_re.replace_all(body, |caps: &regex::Captures| {
        let idx = saved.len();
        saved.push(caps[0].to_string());
        format!("\x00DMATH{idx}\x00")
    });

    // Step 2 — wrap any remaining \begin{equation}…\end{equation}.
    let wrapped = standalone_re.replace_all(&guarded, |caps: &regex::Captures| {
        let inner: String = caps[1]
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        // Keep \begin{equation} / \end{equation} so strip_unsupported_envs
        // can remove them in the next pass, leaving clean $$…$$ content.
        format!("\n\n$$\\begin{{equation}}{inner}\\end{{equation}}$$\n\n")
    });

    // Step 3 — restore hidden blocks.
    let mut out = wrapped.into_owned();
    for (idx, original) in saved.iter().enumerate() {
        out = out.replace(&format!("\x00DMATH{idx}\x00"), original);
    }
    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inline_double_backslash_paren_becomes_dollar() {
        let input = "In the vector \\\\(x \\in \\mathbb{R}^n\\\\), we have...";
        let out = preprocess_source(input);
        assert!(out.contains("$x \\in \\mathbb{R}^n$"), "got: {out}");
        assert!(!out.contains("\\\\("), "two-backslash form remained: {out}");
    }

    #[test]
    fn inline_leading_space_is_trimmed() {
        // pulldown-cmark rejects `$ content $` — the space after `$` breaks
        // the math span. Our normaliser must trim so the opening `$` is
        // immediately followed by the first LaTeX token.
        let input = "See \\\\( \\tau^{(i)} \\\\) here.";
        let out = preprocess_source(input);
        assert!(out.contains("$\\tau^{(i)}$"), "got: {out}");
        assert!(!out.contains("$ \\tau"), "leading space leaked into delimiter: {out}");
    }

    #[test]
    fn display_double_backslash_bracket_becomes_double_dollar() {
        let input = "Before \\\\[ a^2 + b^2 = c^2 \\\\] after.";
        let out = preprocess_source(input);
        assert!(out.contains("$$ a^2 + b^2 = c^2 $$"), "got: {out}");
    }

    #[test]
    fn equation_env_is_stripped_but_label_survives_for_anchor() {
        let input =
            "\\\\[ \\begin{equation}\\label{eq:foo} x = y \\end{equation} \\\\]";
        let out = preprocess_source(input);
        assert!(out.contains("$$"), "delimiters: {out}");
        assert!(!out.contains("\\begin{equation}"), "begin remained: {out}");
        assert!(!out.contains("\\end{equation}"), "end remained: {out}");
        // \label is deliberately kept — `transform_events`'s DisplayMath
        // handler consumes it later to set the equation's anchor `id`.
        assert!(out.contains("\\label{eq:foo}"), "label lost: {out}");
        assert!(out.contains("x = y"), "math content lost: {out}");
    }

    #[test]
    fn standalone_equation_env_becomes_display_math() {
        let input = "Text.\n\n\\begin{equation}\\label{eq:bar}\n    x = y + z\n\\end{equation}\n\nMore.";
        let out = preprocess_source(input);
        assert!(out.contains("$$"), "missing display delimiters: {out}");
        assert!(!out.contains("\\begin{equation}"), "begin remained: {out}");
        assert!(out.contains("\\label{eq:bar}"), "label lost: {out}");
        assert!(out.contains("x = y + z"), "math content lost: {out}");
    }

    #[test]
    fn display_math_inside_brackets_not_double_wrapped() {
        let input = "\\\\[\n    \\begin{equation}\\label{eq:baz} a = b \\end{equation}\n\\\\]";
        let out = preprocess_source(input);
        // Should be a single $$…$$ block, not nested.
        let count = out.matches("$$").count();
        assert_eq!(count, 2, "expected exactly one $$…$$ pair, got: {out}");
        assert!(out.contains("a = b"), "content lost: {out}");
    }

    #[test]
    fn ref_replaced_with_equation_number() {
        let input = "See Equation~(\\ref{eqn:foo}).\n\n\\begin{equation}\\label{eqn:foo}\n    x=1\n\\end{equation}";
        let out = preprocess_source(input);
        assert!(
            out.contains(r##"<a href="#eqn:foo">1</a>"##),
            "expected linked number in ref: {out}"
        );
        assert!(!out.contains("\\ref{"), "\\ref remained: {out}");
    }

    #[test]
    fn eqref_replaced_with_parenthesised_number() {
        let input = "As in \\eqref{eqn:first}.\n\n\\begin{equation}\\label{eqn:first}\n    y=2\n\\end{equation}";
        let out = preprocess_source(input);
        assert!(
            out.contains(r##"<a href="#eqn:first">(1)</a>"##),
            "expected linked (1): {out}"
        );
        assert!(!out.contains("\\eqref{"), "\\eqref remained: {out}");
    }

    #[test]
    fn unknown_ref_renders_as_broken_marker() {
        let input = "See \\ref{eqn:missing} here.";
        let out = preprocess_source(input);
        assert!(out.contains("[?:eqn:missing]"), "expected broken-ref marker: {out}");
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

    #[test]
    fn display_math_trailing_backslash_stripped() {
        // Trailing \\ after \end{bmatrix} is cosmetic noise; must be removed.
        let input = "$$\n\\begin{bmatrix}1\\\\2\\end{bmatrix}~\\in~Z \\\\\n$$";
        let out = preprocess_source(input);
        // The matrix-internal \\ must survive; the outer trailing \\ must not.
        assert!(out.contains("\\begin{bmatrix}"), "bmatrix lost: {out}");
        // Outer trailing \\ should be gone; count top-level $$ pairs.
        assert_eq!(out.matches("$$").count(), 2, "expected one $$…$$ pair: {out}");
        // Ensure the block didn't get needlessly wrapped in aligned.
        assert!(!out.contains("\\begin{aligned}"), "spurious aligned wrap: {out}");
        // Regression: re-wrapping must not introduce a blank line right after
        // the opening `$$` — that splits the block into two CommonMark
        // paragraphs, and pulldown-cmark then fails to recognize the `$$`
        // pair as math at all (see `display_math_survives_paragraph_split`).
        assert!(!out.contains("$$\n\n"), "blank line after opening $$: {out}");
        assert!(!out.contains("\n\n$$"), "blank line before closing $$: {out}");
    }

    #[test]
    fn display_math_survives_paragraph_split_regression() {
        // Reproduces the exact shape found in the "gauss confusion matrix"
        // post: a display block that opens with "$$\n" (content starts on
        // the next line) and ends with a lone trailing `\\` before the
        // closing `$$`. Before the fix, `fix_display_math_newlines` left a
        // stray blank line after the opening `$$`, which splits the block
        // into two CommonMark paragraphs. pulldown-cmark can then no longer
        // match the `$$` pair as a single math span, so the block falls
        // through as literal text — and any lone `~` inside (LaTeX's
        // non-breaking space) gets misparsed as GFM strikethrough.
        let input = "$$\n\\boldsymbol T^{\\top}\\hat{\\boldsymbol T} = \n\\begin{bmatrix}\n1 & 0 & 0\\\\\n1 & 0 & 1\\\\\n0 & 0 & 1\n\\end{bmatrix}~\\in~Z_{0+}^{3~\\times~3} \\\\\n$$";
        let out = preprocess_source(input);
        assert!(!out.contains("$$\n\n"), "blank line reintroduced: {out}");

        let mut options = pulldown_cmark::Options::empty();
        options.insert(pulldown_cmark::Options::ENABLE_MATH);
        options.insert(pulldown_cmark::Options::ENABLE_STRIKETHROUGH);
        let events: Vec<_> = pulldown_cmark::Parser::new_ext(&out, options).collect();
        let has_display_math = events
            .iter()
            .any(|e| matches!(e, pulldown_cmark::Event::DisplayMath(_)));
        assert!(has_display_math, "expected a DisplayMath event, got: {events:?}");
        let has_strikethrough = events.iter().any(|e| {
            matches!(
                e,
                pulldown_cmark::Event::Start(pulldown_cmark::Tag::Strikethrough)
            )
        });
        assert!(
            !has_strikethrough,
            "lone `~` was misparsed as strikethrough: {events:?}"
        );
    }

    #[test]
    fn bare_br_between_display_math_blocks_does_not_swallow_them() {
        // Reproduces the "taylor approximation" post: three single-line $$
        // blocks separated only by bare `<br>` spacers with no blank lines.
        // Without isolation, CommonMark treats the first `<br>` as the start
        // of a raw HTML block that swallows everything up to the next blank
        // line — i.e. all three $$ blocks render as literal text.
        let input = "<br>\n$$y - y_0 = m (x - x_0)$$\n<br>\n$$y = 1 + f'(x_0)x$$\n<br>\n$$y = 1 + x$$\n\nAfter.\n";
        let out = preprocess_source(input);

        let mut options = pulldown_cmark::Options::empty();
        options.insert(pulldown_cmark::Options::ENABLE_MATH);
        let events: Vec<_> = pulldown_cmark::Parser::new_ext(&out, options).collect();
        let display_math_count = events
            .iter()
            .filter(|e| matches!(e, pulldown_cmark::Event::DisplayMath(_)))
            .count();
        assert_eq!(
            display_math_count, 3,
            "expected 3 DisplayMath events, got: {events:?}"
        );
    }

    #[test]
    fn bare_br_immediately_before_display_math_does_not_swallow_it() {
        // Reproduces the "directional derivatives" post: a `<br>` spacer
        // directly followed (no blank line) by a multi-line $$ block.
        let input = "Some text.\n<br>\n$$\n\\nabla f(x) = 0\n$$\n<br>\n\nAfter.\n";
        let out = preprocess_source(input);

        let mut options = pulldown_cmark::Options::empty();
        options.insert(pulldown_cmark::Options::ENABLE_MATH);
        let events: Vec<_> = pulldown_cmark::Parser::new_ext(&out, options).collect();
        let has_display_math = events
            .iter()
            .any(|e| matches!(e, pulldown_cmark::Event::DisplayMath(_)));
        assert!(has_display_math, "expected a DisplayMath event, got: {events:?}");
    }

    #[test]
    fn br_followed_by_blank_line_is_left_untouched() {
        let input = "<br>\n\nSome text.\n";
        assert_eq!(preprocess_source(input), input);
    }

    #[test]
    fn display_math_multiline_wrapped_in_aligned() {
        // Two expressions separated by \\ at the top level must be wrapped.
        let input = "$$\n\\boldsymbol y = [1,0,2,1] \\\\\n\\hat{\\boldsymbol y} = [2,0,2,0]\n$$";
        let out = preprocess_source(input);
        assert!(out.contains("\\begin{aligned}"), "missing aligned wrap: {out}");
        assert!(out.contains("\\end{aligned}"), "missing aligned end: {out}");
        assert!(out.contains("\\boldsymbol y"), "content lost: {out}");
        assert!(out.contains("\\\\"), "line-break removed: {out}");
    }

    #[test]
    fn display_math_matrix_internal_backslash_not_wrapped() {
        // \\ inside \begin{bmatrix}…\end{bmatrix} is depth > 0 — do not wrap.
        let input = "$$\n\\begin{bmatrix}a & b\\\\c & d\\end{bmatrix}\n$$";
        let out = preprocess_source(input);
        assert!(!out.contains("\\begin{aligned}"), "incorrectly wrapped: {out}");
        assert!(out.contains("\\begin{bmatrix}"), "bmatrix lost: {out}");
    }

    #[test]
    fn extract_label_returns_key_and_strips_it() {
        let (cleaned, key) = extract_label("\\label{eqn:foo} x = y");
        assert_eq!(key.as_deref(), Some("eqn:foo"));
        assert_eq!(cleaned.trim(), "x = y");
    }

    #[test]
    fn extract_label_none_when_absent() {
        let (cleaned, key) = extract_label("x = y");
        assert_eq!(key, None);
        assert_eq!(cleaned, "x = y");
    }
}
