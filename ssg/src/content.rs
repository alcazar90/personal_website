//! Content discovery and frontmatter parsing.
//!
//! Walks `content/posts/` and `content/pages/`, parses both `---` YAML and
//! `+++` TOML frontmatter, and returns a flat list of `Source` records
//! sorted by date descending. Drafts are skipped.
//!
//! The frontmatter deserializers are intentionally lenient — legacy Hugo
//! posts sometimes have `slug: []` or `tags: "single"`, and we'd rather
//! accept the file than fail the build. The helpers (`de_string_lenient`,
//! `de_string_list_lenient`, `split_frontmatter`, `find_delim`) are copied
//! verbatim from `migrate/src/main.rs` so both binaries agree on input.

use anyhow::{anyhow, Context, Result};
use regex::Regex;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Frontmatter {
    pub title: Option<String>,
    pub date: Option<String>,
    #[serde(deserialize_with = "de_string_lenient")]
    pub slug: Option<String>,
    #[serde(deserialize_with = "de_string_list_lenient")]
    pub tags: Option<Vec<String>>,
    pub description: Option<String>,
    pub draft: Option<bool>,
    pub lang: Option<String>,
    #[serde(flatten)]
    #[allow(dead_code)] // captured for completeness; readers may inspect later.
    pub extra: BTreeMap<String, serde_yaml::Value>,
}

#[derive(Debug, Clone)]
pub struct Source {
    pub path: PathBuf,
    pub slug: String,
    pub frontmatter: Frontmatter,
    pub body: String,
}

/// Accept a string, an empty sequence, null, or anything else — return None
/// for non-string shapes. Hugo posts occasionally have `slug: []`.
pub fn de_string_lenient<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
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
pub fn de_string_list_lenient<'de, D>(deserializer: D) -> Result<Option<Vec<String>>, D::Error>
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
pub enum FmKind {
    Yaml,
    Toml,
}

/// Walk `<root>/posts/` and `<root>/pages/`, parsing every `.md` file and
/// returning a list of sources sorted by date (newest first).
///
/// Files without frontmatter are skipped with a warning rather than failing
/// the whole build — same with parse errors. Drafts are silently dropped.
pub fn walk(root: &Path) -> Result<Vec<Source>> {
    let mut sources = Vec::new();
    for sub in ["posts", "pages"] {
        let dir = root.join(sub);
        if !dir.exists() {
            continue;
        }
        for entry in WalkDir::new(&dir).into_iter().filter_map(|e| e.ok()) {
            if !entry.file_type().is_file() {
                continue;
            }
            let path = entry.path();
            let ext = path
                .extension()
                .and_then(|s| s.to_str())
                .map(|s| s.to_ascii_lowercase());
            if !matches!(ext.as_deref(), Some("md")) {
                continue;
            }

            match parse_file(path) {
                Ok(Some(source)) => sources.push(source),
                Ok(None) => {} // draft, skip silently
                Err(e) => {
                    eprintln!("warning: skipping {}: {e:#}", path.display());
                }
            }
        }
    }

    // Sort by date descending; missing dates sort last.
    sources.sort_by(|a, b| {
        let ad = a.frontmatter.date.as_deref().unwrap_or("");
        let bd = b.frontmatter.date.as_deref().unwrap_or("");
        bd.cmp(ad)
    });

    Ok(sources)
}

fn parse_file(path: &Path) -> Result<Option<Source>> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let (front_raw, body, kind) = split_frontmatter(&raw)
        .ok_or_else(|| anyhow!("no frontmatter delimiter found"))?;
    let frontmatter = parse_frontmatter(front_raw, kind)
        .with_context(|| format!("parsing frontmatter in {}", path.display()))?;

    if frontmatter.draft.unwrap_or(false) {
        return Ok(None);
    }

    let slug = derive_slug(path, &frontmatter);

    Ok(Some(Source {
        path: path.to_path_buf(),
        slug,
        frontmatter,
        body: body.to_string(),
    }))
}

fn derive_slug(path: &Path, fm: &Frontmatter) -> String {
    if let Some(s) = fm.slug.as_deref().filter(|s| !s.is_empty()) {
        return s.to_string();
    }
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("untitled");
    let date_re = Regex::new(r"^\d{4}-\d{2}-\d{2}-(.+)$").unwrap();
    if let Some(caps) = date_re.captures(stem) {
        caps.get(1).unwrap().as_str().to_string()
    } else {
        stem.to_string()
    }
}

/// Split raw text into (frontmatter_body, post_body, kind). Returns None
/// if there's no recognized delimiter at the start of the file.
pub fn split_frontmatter(raw: &str) -> Option<(&str, &str, FmKind)> {
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

/// Find the next standalone delimiter line; returns (front_len, byte_after_close_line).
pub fn find_delim(s: &str, delim: &str) -> Option<(usize, usize)> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TempDir(PathBuf);
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    impl TempDir {
        fn new(label: &str) -> Self {
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let pid = std::process::id();
            let dir = std::env::temp_dir().join(format!("ssg-content-{label}-{pid}-{n}"));
            fs::create_dir_all(dir.join("posts")).unwrap();
            fs::create_dir_all(dir.join("pages")).unwrap();
            Self(dir)
        }
        fn write_post(&self, name: &str, contents: &str) {
            fs::write(self.0.join("posts").join(name), contents).unwrap();
        }
    }

    #[test]
    fn parses_yaml_frontmatter() {
        let dir = TempDir::new("yaml");
        dir.write_post(
            "2024-01-15-hello.md",
            r#"---
title: Hello World
date: 2024-01-15
tags:
  - intro
  - rust
description: A first post.
---

# Hi

Body text.
"#,
        );

        let sources = walk(&dir.0).unwrap();
        assert_eq!(sources.len(), 1);
        let s = &sources[0];
        assert_eq!(s.frontmatter.title.as_deref(), Some("Hello World"));
        assert_eq!(s.frontmatter.date.as_deref(), Some("2024-01-15"));
        assert_eq!(
            s.frontmatter.tags.as_deref(),
            Some(&["intro".to_string(), "rust".to_string()][..])
        );
        assert_eq!(s.slug, "hello");
        assert!(s.body.starts_with("# Hi"));
    }

    #[test]
    fn parses_toml_frontmatter() {
        let dir = TempDir::new("toml");
        dir.write_post(
            "2024-02-01-toml-post.md",
            r#"+++
title = "TOML Post"
date = "2024-02-01"
tags = ["meta"]
+++

Content here.
"#,
        );

        let sources = walk(&dir.0).unwrap();
        assert_eq!(sources.len(), 1);
        let s = &sources[0];
        assert_eq!(s.frontmatter.title.as_deref(), Some("TOML Post"));
        assert_eq!(s.slug, "toml-post");
        assert_eq!(
            s.frontmatter.tags.as_deref(),
            Some(&["meta".to_string()][..])
        );
    }

    #[test]
    fn lenient_handles_slug_empty_sequence_and_tags_single_string() {
        let dir = TempDir::new("lenient");
        dir.write_post(
            "2024-03-01-quirky.md",
            r#"---
title: Quirky
date: 2024-03-01
slug: []
tags: solo
---

Body.
"#,
        );

        let sources = walk(&dir.0).unwrap();
        assert_eq!(sources.len(), 1);
        let s = &sources[0];
        // slug: [] is treated as no slug — fall back to filename-derived slug.
        assert_eq!(s.slug, "quirky");
        assert_eq!(s.frontmatter.slug, None);
        assert_eq!(
            s.frontmatter.tags.as_deref(),
            Some(&["solo".to_string()][..])
        );
    }

    #[test]
    fn drafts_are_skipped() {
        let dir = TempDir::new("drafts");
        dir.write_post(
            "2024-04-01-real.md",
            "---\ntitle: Real\ndate: 2024-04-01\n---\n\nReal.\n",
        );
        dir.write_post(
            "2024-04-02-draft.md",
            "---\ntitle: Draft\ndate: 2024-04-02\ndraft: true\n---\n\nWIP.\n",
        );
        let sources = walk(&dir.0).unwrap();
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].frontmatter.title.as_deref(), Some("Real"));
    }

    #[test]
    fn sources_sort_by_date_descending() {
        let dir = TempDir::new("sort");
        dir.write_post(
            "2024-01-01-old.md",
            "---\ntitle: Old\ndate: 2024-01-01\n---\n\nold.\n",
        );
        dir.write_post(
            "2024-12-01-new.md",
            "---\ntitle: New\ndate: 2024-12-01\n---\n\nnew.\n",
        );
        let sources = walk(&dir.0).unwrap();
        assert_eq!(sources.len(), 2);
        assert_eq!(sources[0].frontmatter.title.as_deref(), Some("New"));
        assert_eq!(sources[1].frontmatter.title.as_deref(), Some("Old"));
    }
}
