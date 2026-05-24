//! minijinja template environment.
//!
//! All templates are baked into the binary via `include_str!`, so the build
//! never reads templates from disk. The `Templates` struct owns the
//! `Environment` and exposes one render method per page kind.

use crate::config::Config;
use anyhow::{Context, Result};
use minijinja::{context, Environment};
use serde::Serialize;

const BASE_HTML: &str = include_str!("../../templates/base.html");
const POST_HTML: &str = include_str!("../../templates/post.html");
const PAGE_HTML: &str = include_str!("../../templates/page.html");
const INDEX_HTML: &str = include_str!("../../templates/index.html");
const NOT_FOUND_HTML: &str = include_str!("../../templates/404.html");

/// Per-post view-model passed into `post.html`.
#[derive(Debug, Clone, Serialize)]
pub struct PostView {
    pub title: String,
    pub date: String,
    pub date_display: String,
    pub reading_time: u32,
    pub html: String,
    pub slug: String,
    pub description: String,
    pub lang: String,
}

/// Per-page view-model passed into `page.html`.
#[derive(Debug, Clone, Serialize)]
pub struct PageView {
    pub title: String,
    pub html: String,
    pub slug: String,
    pub description: String,
    pub lang: String,
}

/// One row in the home-page post listing. Kept deliberately thin — anything
/// the listing template needs lives here, anything it doesn't is dropped.
#[derive(Debug, Clone, Serialize)]
pub struct PostListEntry {
    pub title: String,
    pub slug: String,
    pub date: String,
    pub date_display: String,
}

/// Render-time inputs that aren't owned by the markdown content itself.
#[derive(Debug, Clone)]
pub struct RenderEnv<'a> {
    pub site: &'a Config,
    pub inline_css: &'a str,
    pub year: i32,
}

/// Full context handed to `render_post`.
#[derive(Debug, Clone)]
pub struct PostContext<'a> {
    pub env: RenderEnv<'a>,
    pub post: PostView,
}

/// Full context handed to `render_page`.
#[derive(Debug, Clone)]
pub struct PageContext<'a> {
    pub env: RenderEnv<'a>,
    pub page: PageView,
}

/// Full context handed to `render_index`. The post list is passed by
/// reference to avoid cloning the per-build vec.
#[derive(Debug, Clone)]
pub struct IndexContext<'a> {
    pub env: RenderEnv<'a>,
    pub posts: &'a [PostListEntry],
}

/// Full context handed to `render_404`. Minimal — the 404 page only needs
/// the site chrome.
#[derive(Debug, Clone)]
pub struct Render404Context<'a> {
    pub env: RenderEnv<'a>,
}

pub struct Templates {
    env: Environment<'static>,
}

impl Templates {
    /// Build a new template environment with all four templates loaded.
    pub fn new() -> Result<Self> {
        let mut env = Environment::new();
        env.add_template("base.html", BASE_HTML)
            .context("loading base.html")?;
        env.add_template("post.html", POST_HTML)
            .context("loading post.html")?;
        env.add_template("page.html", PAGE_HTML)
            .context("loading page.html")?;
        env.add_template("index.html", INDEX_HTML)
            .context("loading index.html")?;
        env.add_template("404.html", NOT_FOUND_HTML)
            .context("loading 404.html")?;
        Ok(Self { env })
    }

    pub fn render_post(&self, ctx: &PostContext<'_>) -> Result<String> {
        let tmpl = self
            .env
            .get_template("post.html")
            .context("looking up post.html")?;
        tmpl.render(context! {
            site => ctx.env.site,
            inline_css => ctx.env.inline_css,
            year => ctx.env.year,
            page_title => format!("{} — {}", ctx.post.title, ctx.env.site.title),
            description => ctx.post.description,
            lang => ctx.post.lang,
            post => ctx.post,
        })
        .context("rendering post.html")
    }

    pub fn render_page(&self, ctx: &PageContext<'_>) -> Result<String> {
        let tmpl = self
            .env
            .get_template("page.html")
            .context("looking up page.html")?;
        tmpl.render(context! {
            site => ctx.env.site,
            inline_css => ctx.env.inline_css,
            year => ctx.env.year,
            page_title => format!("{} — {}", ctx.page.title, ctx.env.site.title),
            description => ctx.page.description,
            lang => ctx.page.lang,
            page => ctx.page,
        })
        .context("rendering page.html")
    }

    /// Render the home page, listing every published post in reverse-
    /// chronological order. The caller is responsible for the sort and for
    /// excluding drafts (both handled by `content::walk`).
    pub fn render_index(&self, ctx: &IndexContext<'_>) -> Result<String> {
        let tmpl = self
            .env
            .get_template("index.html")
            .context("looking up index.html")?;
        tmpl.render(context! {
            site => ctx.env.site,
            inline_css => ctx.env.inline_css,
            year => ctx.env.year,
            page_title => ctx.env.site.title.clone(),
            description => ctx.env.site.description.clone(),
            lang => "en",
            posts => ctx.posts,
        })
        .context("rendering index.html")
    }

    /// Render the 404 page. Served by Cloudflare Pages automatically on
    /// missing routes when written to `public/404.html`.
    pub fn render_404(&self, ctx: &Render404Context<'_>) -> Result<String> {
        let tmpl = self
            .env
            .get_template("404.html")
            .context("looking up 404.html")?;
        tmpl.render(context! {
            site => ctx.env.site,
            inline_css => ctx.env.inline_css,
            year => ctx.env.year,
            page_title => format!("404 — {}", ctx.env.site.title),
            description => "Page not found.",
            lang => "en",
        })
        .context("rendering 404.html")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, GiscusConfig, MenuItem};

    fn fixture_config() -> Config {
        Config {
            title: "Test Site".to_string(),
            url: "https://example.com".to_string(),
            author: "Tester".to_string(),
            description: "A test site.".to_string(),
            menu: vec![
                MenuItem {
                    name: "About".to_string(),
                    url: "/about".to_string(),
                },
                MenuItem {
                    name: "Posts".to_string(),
                    url: "/posts".to_string(),
                },
            ],
            giscus: None,
        }
    }

    fn fixture_post() -> PostView {
        PostView {
            title: "On Reinforcement Learning".to_string(),
            date: "2024-10-02".to_string(),
            date_display: "Oct 2, 2024".to_string(),
            reading_time: 7,
            html: "<p>Body of the post.</p>".to_string(),
            slug: "rl".to_string(),
            description: "An intro.".to_string(),
            lang: "en".to_string(),
        }
    }

    #[test]
    fn post_render_contains_title_and_reading_time() {
        let templates = Templates::new().unwrap();
        let cfg = fixture_config();
        let post = fixture_post();
        let css = ":root { --x: 0; }";
        let ctx = PostContext {
            env: RenderEnv {
                site: &cfg,
                inline_css: css,
                year: 2026,
            },
            post,
        };
        let html = templates.render_post(&ctx).unwrap();
        assert!(!html.is_empty());
        assert!(
            html.contains("On Reinforcement Learning"),
            "missing title in: {html}"
        );
        assert!(html.contains("min read"), "missing 'min read' in: {html}");
        assert!(
            html.contains("Body of the post."),
            "missing rendered body in: {html}"
        );
        assert!(
            html.contains(":root { --x: 0; }"),
            "missing inlined CSS in: {html}"
        );
    }

    #[test]
    fn rendered_post_contains_data_theme_init_script() {
        let templates = Templates::new().unwrap();
        let cfg = fixture_config();
        let ctx = PostContext {
            env: RenderEnv {
                site: &cfg,
                inline_css: "",
                year: 2026,
            },
            post: fixture_post(),
        };
        let html = templates.render_post(&ctx).unwrap();
        assert!(
            html.contains("dataset.theme"),
            "missing theme-init script in: {html}"
        );
        assert!(
            html.contains("prefers-color-scheme"),
            "missing prefers-color-scheme media query in: {html}"
        );
        assert!(
            html.contains("color-scheme"),
            "missing color-scheme meta in: {html}"
        );
    }

    #[test]
    fn page_render_contains_page_title() {
        let templates = Templates::new().unwrap();
        let cfg = fixture_config();
        let page = PageView {
            title: "About Me".to_string(),
            html: "<p>Hello.</p>".to_string(),
            slug: "about".to_string(),
            description: "About page.".to_string(),
            lang: "en".to_string(),
        };
        let ctx = PageContext {
            env: RenderEnv {
                site: &cfg,
                inline_css: "",
                year: 2026,
            },
            page,
        };
        let html = templates.render_page(&ctx).unwrap();
        assert!(html.contains("About Me"), "missing page title in: {html}");
        assert!(html.contains("<p>Hello.</p>"), "missing body in: {html}");
    }

    fn fixture_giscus() -> GiscusConfig {
        GiscusConfig {
            repo: "owner/repo".to_string(),
            repo_id: "R_kgABC".to_string(),
            category: "Comments".to_string(),
            category_id: "DIC_kwABC".to_string(),
            mapping: "pathname".to_string(),
            reactions_enabled: "1".to_string(),
            input_position: "bottom".to_string(),
            strict: "0".to_string(),
            loading: "lazy".to_string(),
        }
    }

    #[test]
    fn post_render_omits_giscus_when_unconfigured() {
        let templates = Templates::new().unwrap();
        let cfg = fixture_config();
        let ctx = PostContext {
            env: RenderEnv {
                site: &cfg,
                inline_css: "",
                year: 2026,
            },
            post: fixture_post(),
        };
        let html = templates.render_post(&ctx).unwrap();
        assert!(
            !html.contains("giscus.app/client.js"),
            "should not include giscus script when giscus is None"
        );
        assert!(
            !html.contains("class=\"comments\""),
            "should not include comments section when giscus is None"
        );
    }

    #[test]
    fn post_render_includes_giscus_when_configured() {
        let templates = Templates::new().unwrap();
        let mut cfg = fixture_config();
        cfg.giscus = Some(fixture_giscus());
        let ctx = PostContext {
            env: RenderEnv {
                site: &cfg,
                inline_css: "",
                year: 2026,
            },
            post: fixture_post(),
        };
        let html = templates.render_post(&ctx).unwrap();
        assert!(
            html.contains("giscus.app/client.js"),
            "should include giscus script when configured"
        );
        // minijinja's HTML autoescape turns `/` into `&#x2f;` in attributes;
        // browsers decode it transparently when giscus reads `script.dataset.repo`.
        assert!(
            html.contains("data-repo=\"owner&#x2f;repo\"")
                || html.contains("data-repo=\"owner/repo\""),
            "should wire repo into data-repo attribute"
        );
        assert!(
            html.contains("data-repo-id=\"R_kgABC\""),
            "should wire repo_id into data-repo-id attribute"
        );
        assert!(
            html.contains("data-category-id=\"DIC_kwABC\""),
            "should wire category_id into data-category-id attribute"
        );
        assert!(
            html.contains("noscript"),
            "should include a noscript fallback link"
        );
        // Theme bridge must ride along on post pages only.
        assert!(
            html.contains("giscus-frame"),
            "post page should include theme bridge JS that targets .giscus-frame"
        );
    }

    #[test]
    fn index_render_lists_every_post_with_title_date_and_slug_link() {
        let templates = Templates::new().unwrap();
        let cfg = fixture_config();
        let env = RenderEnv {
            site: &cfg,
            inline_css: "",
            year: 2026,
        };
        let posts = vec![
            PostListEntry {
                title: "On RL".to_string(),
                slug: "rl".to_string(),
                date: "2024-10-02".to_string(),
                date_display: "Oct 2, 2024".to_string(),
            },
            PostListEntry {
                title: "Older Thoughts".to_string(),
                slug: "older".to_string(),
                date: "2022-01-01".to_string(),
                date_display: "Jan 1, 2022".to_string(),
            },
        ];
        let ctx = IndexContext {
            env,
            posts: &posts,
        };
        let html = templates.render_index(&ctx).unwrap();
        assert!(html.contains("Test Site"), "missing site title in: {html}");
        assert!(html.contains("On RL"), "missing first post title in: {html}");
        assert!(
            html.contains("Older Thoughts"),
            "missing second post title in: {html}"
        );
        assert!(
            html.contains("/posts/rl/"),
            "missing first post link in: {html}"
        );
        assert!(
            html.contains("/posts/older/"),
            "missing second post link in: {html}"
        );
        assert!(html.contains("Oct 2, 2024"), "missing date in: {html}");
        assert!(
            html.contains("post-list"),
            "missing post-list class in: {html}"
        );
    }

    #[test]
    fn index_render_with_no_posts_still_emits_site_chrome() {
        let templates = Templates::new().unwrap();
        let cfg = fixture_config();
        let env = RenderEnv {
            site: &cfg,
            inline_css: "",
            year: 2026,
        };
        let ctx = IndexContext {
            env,
            posts: &[],
        };
        let html = templates.render_index(&ctx).unwrap();
        // Header link still present even when there are no posts.
        assert!(html.contains("Test Site"));
        assert!(html.contains("post-list"));
    }

    #[test]
    fn render_404_contains_404_and_home_link() {
        let templates = Templates::new().unwrap();
        let cfg = fixture_config();
        let env = RenderEnv {
            site: &cfg,
            inline_css: "",
            year: 2026,
        };
        let ctx = Render404Context { env };
        let html = templates.render_404(&ctx).unwrap();
        assert!(html.contains("404"), "missing '404' in: {html}");
        assert!(
            html.contains("href=\"/\""),
            "missing home link in: {html}"
        );
        assert!(html.contains("Test Site"), "missing site chrome in: {html}");
    }
}
