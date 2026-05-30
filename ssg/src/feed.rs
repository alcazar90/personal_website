//! Atom feed + sitemap.xml emission.
//!
//! Both formats are hand-rolled with `String::push_str` — they're small,
//! the schemas are stable, and an extra crate dependency isn't worth it.
//! Dates from frontmatter come in as `YYYY-MM-DD`; we promote them to
//! RFC3339 by appending `T00:00:00Z` so feed readers don't choke.

use crate::config::Config;

/// One feed entry's worth of data. Fields are owned strings so the caller
/// can free intermediate render structures before emitting.
#[derive(Debug, Clone)]
pub struct FeedEntry {
    pub title: String,
    pub slug: String,
    /// ISO date string as it appears in frontmatter, e.g. `2024-10-02`. May
    /// also be a full RFC3339 timestamp — we'll pass it through as-is in
    /// that case.
    pub date: String,
    /// Full rendered HTML body of the post. Wrapped in CDATA inside the
    /// `<content>` element.
    pub html: String,
}

/// One sitemap row. `lastmod` is optional — pages don't always have a date.
#[derive(Debug, Clone)]
pub struct SitemapEntry {
    /// Absolute URL path, e.g. `/`, `/posts/foo/`, `/about/`.
    pub path: String,
    /// Frontmatter date in `YYYY-MM-DD` form, or empty/None for pages
    /// without one.
    pub lastmod: Option<String>,
}

/// Generate an Atom 1.0 feed string. Entries should already be sorted
/// newest-first by the caller; the feed's `<updated>` is taken from the
/// first entry.
pub fn generate_atom(config: &Config, entries: &[FeedEntry]) -> String {
    let site_url = config.url.trim_end_matches('/');
    let feed_updated = entries
        .first()
        .map(|e| to_rfc3339(&e.date))
        .unwrap_or_else(|| "1970-01-01T00:00:00Z".to_string());

    let mut out = String::with_capacity(4096);
    out.push_str("<?xml version=\"1.0\" encoding=\"utf-8\"?>\n");
    out.push_str("<feed xmlns=\"http://www.w3.org/2005/Atom\">\n");
    out.push_str(&format!("  <title>{}</title>\n", xml_escape(&config.title)));
    out.push_str(&format!(
        "  <link rel=\"self\" href=\"{}/feed.xml\"/>\n",
        site_url
    ));
    out.push_str(&format!("  <link href=\"{}/\"/>\n", site_url));
    out.push_str(&format!("  <updated>{}</updated>\n", feed_updated));
    out.push_str(&format!("  <id>{}/</id>\n", site_url));
    out.push_str(&format!(
        "  <author><name>{}</name></author>\n",
        xml_escape(&config.author)
    ));

    for entry in entries {
        let ts = to_rfc3339(&entry.date);
        out.push_str("  <entry>\n");
        out.push_str(&format!(
            "    <title>{}</title>\n",
            xml_escape(&entry.title)
        ));
        out.push_str(&format!(
            "    <link href=\"{}/posts/{}/\"/>\n",
            site_url, entry.slug
        ));
        out.push_str(&format!(
            "    <id>{}/posts/{}/</id>\n",
            site_url, entry.slug
        ));
        out.push_str(&format!("    <published>{}</published>\n", ts));
        out.push_str(&format!("    <updated>{}</updated>\n", ts));
        out.push_str("    <content type=\"html\"><![CDATA[");
        out.push_str(&cdata_safe(&entry.html));
        out.push_str("]]></content>\n");
        out.push_str("  </entry>\n");
    }

    out.push_str("</feed>\n");
    out
}

/// Generate a standard sitemap.xml string (http://www.sitemaps.org/schemas/sitemap/0.9).
pub fn generate_sitemap(config: &Config, entries: &[SitemapEntry]) -> String {
    let site_url = config.url.trim_end_matches('/');
    let mut out = String::with_capacity(2048);
    out.push_str("<?xml version=\"1.0\" encoding=\"utf-8\"?>\n");
    out.push_str("<urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">\n");
    for entry in entries {
        out.push_str("  <url>\n");
        out.push_str(&format!(
            "    <loc>{}{}</loc>\n",
            site_url,
            xml_escape(&entry.path)
        ));
        if let Some(date) = entry.lastmod.as_deref().filter(|s| !s.is_empty()) {
            // Sitemap accepts plain YYYY-MM-DD; pass the trimmed date through.
            let trimmed = date.split('T').next().unwrap_or(date);
            out.push_str(&format!("    <lastmod>{}</lastmod>\n", trimmed));
        }
        out.push_str("  </url>\n");
    }
    out.push_str("</urlset>\n");
    out
}

/// Promote a YYYY-MM-DD date to an RFC3339 timestamp at midnight UTC. If
/// the input already contains a `T`, assume it's already RFC3339-ish and
/// return as-is. On garbage input, return the original so the feed still
/// validates structurally (a reader may flag the entry but the document
/// itself is well-formed).
fn to_rfc3339(raw: &str) -> String {
    if raw.is_empty() {
        return "1970-01-01T00:00:00Z".to_string();
    }
    if raw.contains('T') {
        return raw.to_string();
    }
    // Plain YYYY-MM-DD.
    format!("{}T00:00:00Z", raw)
}

/// XML-escape the five predefined entities. Sufficient for element text
/// and attribute values.
fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(c),
        }
    }
    out
}

/// Defensively split any `]]>` sequence inside CDATA so it can't terminate
/// the section early. Replaces with `]]]]><![CDATA[>` per the standard
/// trick. Vanishingly rare in real post HTML, but cheap to guard.
fn cdata_safe(s: &str) -> String {
    s.replace("]]>", "]]]]><![CDATA[>")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn fixture_config() -> Config {
        Config {
            title: "Test & Co".to_string(),
            url: "https://example.com".to_string(),
            author: "Tester".to_string(),
            description: "A test site.".to_string(),
            footnote: String::new(),
            social: Default::default(),
            menu: Vec::new(),
            giscus: None,
        }
    }

    fn fixture_entries() -> Vec<FeedEntry> {
        vec![
            FeedEntry {
                title: "New <Post>".to_string(),
                slug: "new-post".to_string(),
                date: "2024-10-02".to_string(),
                html: "<p>Body of the new post.</p>".to_string(),
            },
            FeedEntry {
                title: "Older Thoughts".to_string(),
                slug: "older".to_string(),
                date: "2022-01-01".to_string(),
                html: "<p>Older body.</p>".to_string(),
            },
        ]
    }

    #[test]
    fn atom_feed_is_well_formed_and_contains_entries() {
        let cfg = fixture_config();
        let entries = fixture_entries();
        let xml = generate_atom(&cfg, &entries);

        assert!(xml.starts_with("<?xml version=\"1.0\""));
        assert!(xml.contains("<feed xmlns=\"http://www.w3.org/2005/Atom\">"));
        assert!(xml.trim_end().ends_with("</feed>"));
        // Two entries, opening and closing tags.
        assert_eq!(xml.matches("<entry>").count(), 2);
        assert_eq!(xml.matches("</entry>").count(), 2);
        // Site metadata.
        assert!(xml.contains("<title>Test &amp; Co</title>"));
        assert!(xml.contains("<author><name>Tester</name></author>"));
        assert!(xml.contains("<link rel=\"self\" href=\"https://example.com/feed.xml\"/>"));
        // Feed-level updated comes from the newest entry.
        assert!(xml.contains("<updated>2024-10-02T00:00:00Z</updated>"));
    }

    #[test]
    fn atom_entry_titles_are_xml_escaped() {
        let cfg = fixture_config();
        let entries = fixture_entries();
        let xml = generate_atom(&cfg, &entries);
        // < and > in titles must be escaped.
        assert!(xml.contains("<title>New &lt;Post&gt;</title>"));
        assert!(!xml.contains("<title>New <Post></title>"));
    }

    #[test]
    fn atom_entry_links_use_absolute_urls() {
        let cfg = fixture_config();
        let entries = fixture_entries();
        let xml = generate_atom(&cfg, &entries);
        assert!(xml.contains("<link href=\"https://example.com/posts/new-post/\"/>"));
        assert!(xml.contains("<id>https://example.com/posts/older/</id>"));
    }

    #[test]
    fn atom_entry_body_is_wrapped_in_cdata() {
        let cfg = fixture_config();
        let entries = fixture_entries();
        let xml = generate_atom(&cfg, &entries);
        assert!(xml.contains("<![CDATA[<p>Body of the new post.</p>]]>"));
    }

    #[test]
    fn atom_cdata_close_inside_body_is_escaped() {
        let cfg = fixture_config();
        let entries = vec![FeedEntry {
            title: "Tricky".to_string(),
            slug: "tricky".to_string(),
            date: "2024-01-01".to_string(),
            html: "<p>literal: ]]> embedded</p>".to_string(),
        }];
        let xml = generate_atom(&cfg, &entries);
        // The raw `]]>` sequence must NOT survive intact inside content
        // (would close the CDATA early). The escape splits it across two
        // CDATA sections.
        let content_start = xml.find("<content").unwrap();
        let content_end = xml[content_start..].find("</content>").unwrap();
        let content_slice = &xml[content_start..content_start + content_end];
        // After the opening CDATA marker, the only `]]>` should be the
        // closing one — which lives outside the slice we're looking at.
        // Anything internal is rewritten.
        let inside = content_slice.trim_start_matches(|c: char| c != '[');
        let inside = inside.trim_start_matches("[CDATA[");
        // Within the body region the escape sequence appears.
        assert!(inside.contains("]]]]><![CDATA[>"), "got: {}", inside);
    }

    #[test]
    fn atom_with_no_entries_still_well_formed() {
        let cfg = fixture_config();
        let xml = generate_atom(&cfg, &[]);
        assert!(xml.contains("<feed"));
        assert!(xml.trim_end().ends_with("</feed>"));
        assert!(!xml.contains("<entry>"));
        // Falls back to epoch when there's nothing to report.
        assert!(xml.contains("<updated>1970-01-01T00:00:00Z</updated>"));
    }

    #[test]
    fn sitemap_contains_expected_loc_entries() {
        let cfg = fixture_config();
        let entries = vec![
            SitemapEntry {
                path: "/".to_string(),
                lastmod: Some("2024-10-02".to_string()),
            },
            SitemapEntry {
                path: "/posts/new-post/".to_string(),
                lastmod: Some("2024-10-02".to_string()),
            },
            SitemapEntry {
                path: "/about/".to_string(),
                lastmod: None,
            },
        ];
        let xml = generate_sitemap(&cfg, &entries);
        assert!(xml.contains("<urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">"));
        assert!(xml.contains("<loc>https://example.com/</loc>"));
        assert!(xml.contains("<loc>https://example.com/posts/new-post/</loc>"));
        assert!(xml.contains("<loc>https://example.com/about/</loc>"));
        // Page without lastmod emits a <url> with no <lastmod>.
        assert_eq!(xml.matches("<url>").count(), 3);
        assert_eq!(xml.matches("<lastmod>").count(), 2);
    }

    #[test]
    fn sitemap_strips_time_portion_from_lastmod() {
        let cfg = fixture_config();
        let entries = vec![SitemapEntry {
            path: "/".to_string(),
            lastmod: Some("2024-10-02T12:34:56Z".to_string()),
        }];
        let xml = generate_sitemap(&cfg, &entries);
        assert!(xml.contains("<lastmod>2024-10-02</lastmod>"));
    }

    #[test]
    fn to_rfc3339_pads_plain_date() {
        assert_eq!(to_rfc3339("2024-10-02"), "2024-10-02T00:00:00Z");
        assert_eq!(
            to_rfc3339("2024-10-02T00:00:00Z"),
            "2024-10-02T00:00:00Z"
        );
        assert_eq!(to_rfc3339(""), "1970-01-01T00:00:00Z");
    }
}
