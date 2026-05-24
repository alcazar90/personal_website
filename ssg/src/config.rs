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

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub title: String,
    pub url: String,
    pub author: String,
    pub description: String,
    #[serde(default)]
    pub menu: Vec<MenuItem>,
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
