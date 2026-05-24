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

    /// Render the stub index page. Real listing lands in a later PR.
    pub fn render_index(&self, env: &RenderEnv<'_>) -> Result<String> {
        let tmpl = self
            .env
            .get_template("index.html")
            .context("looking up index.html")?;
        tmpl.render(context! {
            site => env.site,
            inline_css => env.inline_css,
            year => env.year,
            page_title => env.site.title.clone(),
            description => env.site.description.clone(),
            lang => "en",
        })
        .context("rendering index.html")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, MenuItem};

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

    #[test]
    fn index_render_includes_site_header_and_stub() {
        let templates = Templates::new().unwrap();
        let cfg = fixture_config();
        let env = RenderEnv {
            site: &cfg,
            inline_css: "",
            year: 2026,
        };
        let html = templates.render_index(&env).unwrap();
        assert!(html.contains("Test Site"));
        assert!(html.contains("Site index coming soon"));
    }
}
