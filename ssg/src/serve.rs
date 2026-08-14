//! Minimal local HTTP server for previewing a build in `public/`.
//!
//! Static-only, single-threaded, no live reload — `ssg serve` builds once
//! then serves the result. Re-run it to pick up new changes.

use anyhow::{anyhow, Result};
use std::fs;
use std::path::{Component, Path, PathBuf};
use tiny_http::{Header, Response, Server};

pub fn run(out_root: &Path, port: u16) -> Result<()> {
    let addr = format!("127.0.0.1:{port}");
    let server = Server::http(&addr).map_err(|e| anyhow!("binding {addr}: {e}"))?;

    eprintln!("serving {} at http://{addr}/  (Ctrl+C to stop)", out_root.display());

    for request in server.incoming_requests() {
        let path_only = request.url().split('?').next().unwrap_or("/").to_string();
        let decoded = percent_decode(&path_only);

        let (body, content_type, status) = match resolve(out_root, &decoded) {
            Some(file_path) => match fs::read(&file_path) {
                Ok(bytes) => {
                    let ct = content_type_for(&file_path);
                    (bytes, ct, 200u16)
                }
                Err(e) => {
                    eprintln!("warning: failed reading {}: {e}", file_path.display());
                    (b"internal server error".to_vec(), "text/plain; charset=utf-8", 500)
                }
            },
            None => {
                let bytes = fs::read(out_root.join("404.html"))
                    .unwrap_or_else(|_| b"404 not found".to_vec());
                (bytes, "text/html; charset=utf-8", 404)
            }
        };

        let header = Header::from_bytes(&b"Content-Type"[..], content_type.as_bytes())
            .expect("static content-type value is a valid header");
        let response = Response::from_data(body)
            .with_status_code(status)
            .with_header(header);
        if let Err(e) = request.respond(response) {
            eprintln!("warning: failed writing response for {decoded}: {e}");
        }
    }

    Ok(())
}

/// Map a request path to a file under `root`, following the same
/// `<dir>/index.html` convention the build emits for every post/page/index.
/// Returns `None` for anything outside `root` (rejects `..` components) or
/// that doesn't resolve to a real file.
fn resolve(root: &Path, req_path: &str) -> Option<PathBuf> {
    let trimmed = req_path.trim_start_matches('/');
    let rel: PathBuf = if trimmed.is_empty() {
        PathBuf::from("index.html")
    } else {
        PathBuf::from(trimmed)
    };
    if rel.components().any(|c| matches!(c, Component::ParentDir)) {
        return None;
    }

    let mut candidate = root.join(&rel);
    if candidate.is_dir() {
        candidate = candidate.join("index.html");
    }
    candidate.is_file().then_some(candidate)
}

fn content_type_for(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("html") => "text/html; charset=utf-8",
        Some("xml") => "application/xml; charset=utf-8",
        Some("json") => "application/json",
        Some("css") => "text/css; charset=utf-8",
        Some("js") => "application/javascript; charset=utf-8",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("ico") => "image/x-icon",
        Some("pdf") => "application/pdf",
        Some("woff") => "font/woff",
        Some("woff2") => "font/woff2",
        Some("txt") => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    }
}

/// Decode `%XX` escapes. Local dev server only — not meant to handle
/// arbitrary untrusted input.
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).ok();
            if let Some(byte) = hex.and_then(|h| u8::from_str_radix(h, 16).ok()) {
                out.push(byte);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(label: &str) -> Self {
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let pid = std::process::id();
            let dir = std::env::temp_dir().join(format!("ssg-serve-test-{label}-{pid}-{n}"));
            fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn write(root: &Path, rel: &str, contents: &str) {
        let path = root.join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, contents).unwrap();
    }

    #[test]
    fn root_resolves_to_index_html() {
        let dir = TempDir::new("root");
        write(&dir.0, "index.html", "home");
        let resolved = resolve(&dir.0, "/").unwrap();
        assert_eq!(resolved, dir.0.join("index.html"));
    }

    #[test]
    fn directory_route_resolves_to_its_index_html() {
        let dir = TempDir::new("dirroute");
        write(&dir.0, "posts/hello/index.html", "post");
        assert_eq!(
            resolve(&dir.0, "/posts/hello/").unwrap(),
            dir.0.join("posts/hello/index.html")
        );
        assert_eq!(
            resolve(&dir.0, "/posts/hello").unwrap(),
            dir.0.join("posts/hello/index.html")
        );
    }

    #[test]
    fn direct_file_route_resolves_as_is() {
        let dir = TempDir::new("file");
        write(&dir.0, "feed.xml", "<feed/>");
        assert_eq!(resolve(&dir.0, "/feed.xml").unwrap(), dir.0.join("feed.xml"));
    }

    #[test]
    fn missing_path_resolves_to_none() {
        let dir = TempDir::new("missing");
        assert!(resolve(&dir.0, "/nope").is_none());
    }

    #[test]
    fn parent_dir_traversal_is_rejected() {
        let dir = TempDir::new("traversal");
        write(&dir.0, "index.html", "home");
        assert!(resolve(&dir.0, "/../index.html").is_none());
        assert!(resolve(&dir.0, "/posts/../../index.html").is_none());
    }

    #[test]
    fn percent_decode_handles_spaces_and_leaves_invalid_escapes_alone() {
        assert_eq!(percent_decode("/hello%20world"), "/hello world");
        assert_eq!(percent_decode("/100%"), "/100%");
        assert_eq!(percent_decode("/a%2Fb"), "/a/b");
    }

    #[test]
    fn content_type_maps_known_extensions() {
        assert_eq!(content_type_for(Path::new("x.html")), "text/html; charset=utf-8");
        assert_eq!(content_type_for(Path::new("x.webp")), "image/webp");
        assert_eq!(content_type_for(Path::new("x.unknownext")), "application/octet-stream");
    }
}
