//! Site configuration loaded from `content/config.toml`.
//!
//! The schema is intentionally flat — no nested sections — so a missing
//! field is a typo, not a structural mismatch. Add fields here as the
//! generator grows.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MenuItem {
    pub name: String,
    pub url: String,
}

/// Optional giscus comments configuration. When present, post pages render a
/// giscus mount; when absent, post pages render without a comments section.
/// All fields are required when the block is present — `repo_id` and
/// `category_id` are issued by https://giscus.app for a given GitHub repo.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GiscusConfig {
    pub repo: String,
    pub repo_id: String,
    pub category: String,
    pub category_id: String,
    pub mapping: String,
    pub reactions_enabled: String,
    pub input_position: String,
    pub strict: String,
    pub loading: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub title: String,
    pub url: String,
    pub author: String,
    pub description: String,
    #[serde(default)]
    pub menu: Vec<MenuItem>,
    /// Optional. Absent in dev/local configs that haven't claimed a giscus
    /// repo yet — post pages just skip the comments mount in that case.
    #[serde(default)]
    pub giscus: Option<GiscusConfig>,
}

impl Config {
    /// Read and parse a TOML config file. The caller is responsible for
    /// pointing this at `content/config.toml` (or wherever the site root is).
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let raw = fs::read_to_string(path)
            .with_context(|| format!("reading config from {}", path.display()))?;
        let config: Self = toml::from_str(&raw)
            .with_context(|| format!("parsing TOML in {}", path.display()))?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp(contents: &str) -> tempfile_lite::TempPath {
        let mut t = tempfile_lite::NamedTempFile::new("ssg-config", ".toml");
        t.file.write_all(contents.as_bytes()).unwrap();
        t.into_path()
    }

    #[test]
    fn loads_flat_schema() {
        let path = write_temp(
            r#"
title = "My Site"
url = "https://example.com"
author = "Jane"
description = "A blog"

[[menu]]
name = "About"
url = "/about"

[[menu]]
name = "Posts"
url = "/posts"
"#,
        );

        let config = Config::load(&path).unwrap();
        assert_eq!(config.title, "My Site");
        assert_eq!(config.url, "https://example.com");
        assert_eq!(config.author, "Jane");
        assert_eq!(config.description, "A blog");
        assert_eq!(config.menu.len(), 2);
        assert_eq!(config.menu[0].name, "About");
        assert_eq!(config.menu[1].url, "/posts");
    }

    #[test]
    fn empty_menu_is_ok() {
        let path = write_temp(
            r#"
title = "T"
url = "U"
author = "A"
description = "D"
"#,
        );
        let config = Config::load(&path).unwrap();
        assert!(config.menu.is_empty());
    }

    #[test]
    fn giscus_block_is_optional() {
        let path = write_temp(
            r#"
title = "T"
url = "U"
author = "A"
description = "D"
"#,
        );
        let config = Config::load(&path).unwrap();
        assert!(config.giscus.is_none());
    }

    #[test]
    fn giscus_block_parses_when_present() {
        let path = write_temp(
            r#"
title = "T"
url = "U"
author = "A"
description = "D"

[giscus]
repo = "owner/repo"
repo_id = "R_kgABC"
category = "Comments"
category_id = "DIC_kwABC"
mapping = "pathname"
reactions_enabled = "1"
input_position = "bottom"
strict = "0"
loading = "lazy"
"#,
        );
        let config = Config::load(&path).unwrap();
        let g = config.giscus.expect("giscus block should parse");
        assert_eq!(g.repo, "owner/repo");
        assert_eq!(g.repo_id, "R_kgABC");
        assert_eq!(g.category, "Comments");
        assert_eq!(g.category_id, "DIC_kwABC");
        assert_eq!(g.mapping, "pathname");
        assert_eq!(g.reactions_enabled, "1");
        assert_eq!(g.input_position, "bottom");
        assert_eq!(g.strict, "0");
        assert_eq!(g.loading, "lazy");
    }

    #[test]
    fn missing_file_reports_clear_error() {
        let err = Config::load("/nonexistent/path/to/config.toml").unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("reading config"), "got: {msg}");
    }
}

#[cfg(test)]
mod tempfile_lite {
    //! Tiny inline temp-file helper to avoid pulling in a tempfile crate.
    use std::fs::{self, File};
    use std::ops::Deref;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    pub struct NamedTempFile {
        pub path: PathBuf,
        pub file: File,
    }

    pub struct TempPath(PathBuf);

    impl Deref for TempPath {
        type Target = Path;
        fn deref(&self) -> &Path {
            &self.0
        }
    }

    impl AsRef<Path> for TempPath {
        fn as_ref(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempPath {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }

    impl NamedTempFile {
        pub fn new(prefix: &str, suffix: &str) -> Self {
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let pid = std::process::id();
            let path = std::env::temp_dir().join(format!("{prefix}-{pid}-{n}{suffix}"));
            let file = File::create(&path).expect("create tempfile");
            Self { path, file }
        }

        pub fn into_path(self) -> TempPath {
            TempPath(self.path)
        }
    }
}
