//! One-shot Rmd/md migrator. Walks a Hugo-style content directory, strips R
//! code chunks, normalizes frontmatter to YAML, and emits clean CommonMark
//! `.md` files for the Rust SSG to consume. Posts whose value depends on
//! R-executed output (figures, tables, computed values) are classified as
//! `output-dependent` and skipped — left for manual rewrite or removal.

use anyhow::{anyhow, Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Classification {
    Clean,
    DecorativeOnly,
    OutputDependent,
}

#[derive(Debug)]
struct Outcome {
    source: PathBuf,
    classification: Classification,
    written: Option<PathBuf>,
    reason: String,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Frontmatter {
    title: Option<String>,
    date: Option<String>,
    #[serde(deserialize_with = "de_string_lenient")]
    slug: Option<String>,
    #[serde(deserialize_with = "de_string_list_lenient")]
    tags: Option<Vec<String>>,
    #[serde(deserialize_with = "de_string_list_lenient")]
    categories: Option<Vec<String>>,
    description: Option<String>,
    draft: Option<bool>,
    lang: Option<String>,
    #[serde(flatten)]
    _extra: BTreeMap<String, serde_yaml::Value>,
}

/// Accept a string, an empty sequence, null, or anything else — return None
/// for non-string shapes. Hugo posts occasionally have `slug: []`.
fn de_string_lenient<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v = Option::<serde_yaml::Value>::deserialize(deserializer)?;
    Ok(match v {
        Some(serde_yaml::Value::String(s)) if !s.is_empty() => Some(s),
        _ => None,
    })
}

/// Accept a sequence of strings, a single string, or anything else.
/// Returns None for empty or unrecognized shapes.
fn de_string_list_lenient<'de, D>(deserializer: D) -> Result<Option<Vec<String>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v = Option::<serde_yaml::Value>::deserialize(deserializer)?;
    Ok(match v {
        Some(serde_yaml::Value::Sequence(seq)) => {
            let items: Vec<String> = seq
                .into_iter()
                .filter_map(|v| match v {
                    serde_yaml::Value::String(s) if !s.is_empty() => Some(s),
                    _ => None,
                })
                .collect();
            (!items.is_empty()).then_some(items)
        }
        Some(serde_yaml::Value::String(s)) if !s.is_empty() => Some(vec![s]),
        _ => None,
    })
}

#[derive(Debug, Clone, Copy)]
enum FmKind {
    Yaml,
    Toml,
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let source = PathBuf::from(args.next().unwrap_or_else(|| "content".into()));
    let target = PathBuf::from(args.next().unwrap_or_else(|| "posts".into()));

    if !source.exists() {
        return Err(anyhow!(
            "source directory does not exist: {}",
            source.display()
        ));
    }

    fs::create_dir_all(&target).with_context(|| format!("creating {}", target.display()))?;

    let mut outcomes = Vec::new();
    for entry in WalkDir::new(&source).into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase());
        if !matches!(ext.as_deref(), Some("md") | Some("rmd")) {
            continue;
        }
        let result = process(path, &target).unwrap_or_else(|e| Outcome {
            source: path.to_path_buf(),
            classification: Classification::OutputDependent,
            written: None,
            reason: format!("error: {e:#}"),
        });
        outcomes.push(result);
    }

    print_report(&outcomes, &source, &target);
    Ok(())
}

fn process(path: &Path, target_root: &Path) -> Result<Outcome> {
    let raw =
        fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let (front_raw, body, fm_kind) = split_frontmatter(&raw)
        .ok_or_else(|| anyhow!("no frontmatter delimiter found"))?;
    let frontmatter = parse_frontmatter(front_raw, fm_kind)
        .with_context(|| format!("parsing frontmatter in {}", path.display()))?;

    let (cleaned_body, r_total, r_decorative) = strip_r_chunks(body);

    let classification = if r_total == 0 {
        Classification::Clean
    } else if r_decorative == r_total {
        Classification::DecorativeOnly
    } else {
        Classification::OutputDependent
    };

    if classification == Classification::OutputDependent {
        return Ok(Outcome {
            source: path.to_path_buf(),
            classification,
            written: None,
            reason: format!(
                "{} chunk(s), {} produce visible output — manual review",
                r_total,
                r_total - r_decorative
            ),
        });
    }

    let out_name = compute_output_filename(path, &frontmatter);
    let out_path = target_root.join(&out_name);
    let normalized = render_output(&frontmatter, &cleaned_body)?;
    fs::write(&out_path, normalized)
        .with_context(|| format!("writing {}", out_path.display()))?;

    let reason = match classification {
        Classification::Clean => "no chunks".to_string(),
        Classification::DecorativeOnly => {
            format!("{r_total} decorative chunk(s) (eval/include=FALSE)")
        }
        Classification::OutputDependent => unreachable!(),
    };

    Ok(Outcome {
        source: path.to_path_buf(),
        classification,
        written: Some(out_path),
        reason,
    })
}

fn split_frontmatter(raw: &str) -> Option<(&str, &str, FmKind)> {
    let trimmed = raw.trim_start_matches('\u{feff}');
    for (open, kind) in [("---", FmKind::Yaml), ("+++", FmKind::Toml)] {
        if let Some(rest) = trimmed.strip_prefix(&format!("{open}\n")) {
            if let Some((front_len, after_close)) = find_delim(rest, open) {
                let front = &rest[..front_len];
                let body = &rest[after_close..];
                return Some((front, body.trim_start_matches(['\n', '\r']), kind));
            }
        }
        if let Some(rest) = trimmed.strip_prefix(&format!("{open}\r\n")) {
            if let Some((front_len, after_close)) = find_delim(rest, open) {
                let front = &rest[..front_len];
                let body = &rest[after_close..];
                return Some((front, body.trim_start_matches(['\n', '\r']), kind));
            }
        }
    }
    None
}

/// Find the next standalone delimiter line; returns (front_len, byte_after_close_line)
fn find_delim(s: &str, delim: &str) -> Option<(usize, usize)> {
    let mut offset = 0usize;
    for line in s.split_inclusive('\n') {
        let trimmed = line.trim_end_matches(['\n', '\r']);
        if trimmed == delim {
            return Some((offset, offset + line.len()));
        }
        offset += line.len();
    }
    None
}

fn parse_frontmatter(raw: &str, kind: FmKind) -> Result<Frontmatter> {
    match kind {
        FmKind::Yaml => serde_yaml::from_str(raw).map_err(Into::into),
        FmKind::Toml => toml::from_str(raw).map_err(Into::into),
    }
}

/// Strip RMarkdown code chunks. Returns (cleaned_body, total_chunks, decorative_chunks).
///
/// "Decorative" = chunk produces no visible output, either because `eval=FALSE`
/// (code shown, not run) or `include=FALSE` (nothing shown at all). Decorative
/// chunks with code visible (eval=FALSE) are kept as plain ```<lang> fences.
/// Decorative invisible chunks (include=FALSE) are dropped silently.
///
/// Output-producing chunks are also dropped from the body — but their presence
/// flips classification to output-dependent and the post gets skipped upstream,
/// so the discard has no practical effect.
fn strip_r_chunks(body: &str) -> (String, usize, usize) {
    // Match RMarkdown opener like ```{r}, ```{r setup, ...}, ```{python}, etc.
    let opener_re =
        Regex::new(r"^\s*```+\s*\{([A-Za-z][A-Za-z0-9_]*)([\s,}].*)?$").unwrap();
    let eval_false_re = Regex::new(r"\beval\s*=\s*(?:F|FALSE|false)\b").unwrap();
    let include_false_re = Regex::new(r"\binclude\s*=\s*(?:F|FALSE|false)\b").unwrap();
    let closer_re = Regex::new(r"^\s*```+\s*$").unwrap();

    let mut out = String::with_capacity(body.len());
    let mut total = 0usize;
    let mut decorative = 0usize;

    let mut lines = body.lines();
    while let Some(line) = lines.next() {
        if let Some(caps) = opener_re.captures(line) {
            total += 1;
            let lang = caps.get(1).unwrap().as_str().to_lowercase();
            let is_eval_false = eval_false_re.is_match(line);
            let is_include_false = include_false_re.is_match(line);
            let is_decorative = is_eval_false || is_include_false;
            if is_decorative {
                decorative += 1;
            }

            let mut chunk_body = String::new();
            for inner in lines.by_ref() {
                if closer_re.is_match(inner) {
                    break;
                }
                chunk_body.push_str(inner);
                chunk_body.push('\n');
            }

            // Show code only for eval=FALSE (code is intentionally visible).
            // include=FALSE → drop entirely.
            // Output-producing → also drop (post will be classified output-dep).
            if is_eval_false {
                out.push_str(&format!("```{lang}\n"));
                out.push_str(&chunk_body);
                out.push_str("```\n");
            }
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }

    (out, total, decorative)
}

fn compute_output_filename(path: &Path, fm: &Frontmatter) -> String {
    // For posts laid out as <date-slug>/index.rmd, use the parent directory
    // name as the stem; otherwise use the file stem.
    let stem_source = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("untitled");
    let effective_stem: String = if stem_source == "index" {
        path.parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .unwrap_or("untitled")
            .to_string()
    } else {
        stem_source.to_string()
    };

    let date_re = Regex::new(r"^(\d{4}-\d{2}-\d{2})-(.*)$").unwrap();
    let (date_prefix, slug_part) = if let Some(caps) = date_re.captures(&effective_stem) {
        (
            Some(caps.get(1).unwrap().as_str().to_string()),
            caps.get(2).unwrap().as_str().to_string(),
        )
    } else {
        (None, effective_stem.clone())
    };

    let slug = fm
        .slug
        .clone()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| kebab(&slug_part));

    let date = fm
        .date
        .as_deref()
        .and_then(|d| d.get(0..10))
        .map(|d| d.to_string())
        .or(date_prefix);

    match date {
        Some(d) => format!("{d}-{slug}.md"),
        None => format!("{slug}.md"),
    }
}

fn kebab(s: &str) -> String {
    let lower = s.to_lowercase();
    let mut out = String::with_capacity(lower.len());
    let mut prev_dash = false;
    for c in lower.chars() {
        let allowed = c.is_ascii_alphanumeric();
        if allowed {
            out.push(c);
            prev_dash = false;
        } else if !prev_dash && !out.is_empty() {
            out.push('-');
            prev_dash = true;
        }
    }
    out.trim_matches('-').to_string()
}

fn render_output(fm: &Frontmatter, body: &str) -> Result<String> {
    #[derive(Serialize)]
    struct OutFront {
        title: String,
        date: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        slug: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tags: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        draft: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        lang: Option<String>,
    }

    let title = fm.title.clone().unwrap_or_else(|| "Untitled".to_string());
    let date = fm
        .date
        .clone()
        .map(|d| normalize_date(&d))
        .unwrap_or_else(|| "1970-01-01".to_string());

    let out = OutFront {
        title,
        date,
        slug: fm.slug.clone(),
        tags: fm.tags.clone(),
        description: fm.description.clone(),
        draft: fm.draft,
        lang: fm.lang.clone(),
    };

    let yaml = serde_yaml::to_string(&out)?;
    Ok(format!("---\n{yaml}---\n\n{body}"))
}

fn normalize_date(s: &str) -> String {
    if let Some(prefix) = s.get(0..10) {
        let digits = prefix.chars().filter(|c| c.is_ascii_digit()).count();
        if digits >= 8 && prefix.contains('-') {
            return prefix.to_string();
        }
    }
    s.to_string()
}

fn print_report(outcomes: &[Outcome], source: &Path, target: &Path) {
    use Classification::*;

    let total = outcomes.len();
    let emitted = outcomes.iter().filter(|o| o.written.is_some()).count();
    let skipped = total - emitted;
    let cleans = outcomes.iter().filter(|o| o.classification == Clean).count();
    let decoratives = outcomes
        .iter()
        .filter(|o| o.classification == DecorativeOnly)
        .count();
    let output_dep = outcomes
        .iter()
        .filter(|o| o.classification == OutputDependent)
        .count();

    println!();
    println!("=== Migration report ===");
    println!("Source: {}", source.display());
    println!("Target: {}", target.display());
    println!();
    println!("Files seen:        {total}");
    println!("  clean:           {cleans}");
    println!("  decorative-only: {decoratives}");
    println!("  output-dep:      {output_dep}");
    println!();
    println!("Files emitted:     {emitted}");
    println!("Files skipped:     {skipped}");
    println!();
    println!("Per file:");

    let mut sorted: Vec<&Outcome> = outcomes.iter().collect();
    sorted.sort_by(|a, b| a.source.cmp(&b.source));
    for o in sorted {
        let mark = match (o.classification, o.written.is_some()) {
            (Clean, true) => " OK ",
            (DecorativeOnly, true) => "DEC ",
            (_, false) => "SKIP",
            (_, true) => " ?? ",
        };
        let rel = o.source.strip_prefix(source).unwrap_or(&o.source);
        println!("  [{}] {} — {}", mark, rel.display(), o.reason);
    }
    println!();
}
