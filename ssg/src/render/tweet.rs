//! Static tweet/X card rendering.
//!
//! `<blockquote class="twitter-tweet">` blocks embedded in post content used
//! to be upgraded client-side by X's `widgets.js`, which meant every visitor
//! saw a flash: plain quoted text first, then — after a script load and a
//! couple of network round-trips — the styled card. This module replaces
//! that at build time instead: it fetches the tweet from X's public (albeit
//! unofficial) syndication endpoint — the same one `widgets.js` calls under
//! the hood — and renders a static card using this site's own Flexoki CSS
//! variables, so it ships already themed in the page HTML with no
//! client-side widget dependency at all.
//!
//! Fetched tweets are cached at `content/tweet-cache.json`, keyed by tweet
//! id, and meant to be committed — once a tweet has been fetched, rebuilds
//! never need the network again. If a tweet can't be fetched (offline build,
//! deleted tweet, endpoint changes) and isn't already cached, the original
//! `<blockquote>` markup passes through untouched, matching the previous
//! plain-text fallback — this must never fail the build.

use super::html_escape;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::OnceLock;

const CACHE_PATH: &str = "content/tweet-cache.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Tweet {
    id_str: String,
    text: String,
    #[serde(default)]
    entities: TweetEntities,
    user: TweetUser,
    created_at: String,
    #[serde(default)]
    favorite_count: u64,
    #[serde(default)]
    conversation_count: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct TweetEntities {
    #[serde(default)]
    urls: Vec<UrlEntity>,
    #[serde(default)]
    hashtags: Vec<IndexedText>,
    #[serde(default)]
    user_mentions: Vec<MentionEntity>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct UrlEntity {
    indices: [usize; 2],
    expanded_url: String,
    display_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IndexedText {
    indices: [usize; 2],
    text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MentionEntity {
    indices: [usize; 2],
    screen_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TweetUser {
    name: String,
    screen_name: String,
    profile_image_url_https: String,
    #[serde(default)]
    is_blue_verified: bool,
    #[serde(default)]
    verified: bool,
}

type Cache = HashMap<String, Tweet>;

/// Replace every `<blockquote class="twitter-tweet">…</blockquote>` block in
/// `body` with a static, themed card. Blocks whose tweet can't be resolved
/// (no cache entry and the fetch fails) are left exactly as they were.
pub fn render_tweets(body: &str) -> String {
    static BLOCK_RE: OnceLock<Regex> = OnceLock::new();
    let re = BLOCK_RE.get_or_init(|| {
        Regex::new(r#"(?is)<blockquote\b[^>]*\bclass="twitter-tweet"[^>]*>.*?</blockquote>"#)
            .unwrap()
    });
    if !re.is_match(body) {
        return body.to_string();
    }

    let mut cache = load_cache();
    let mut dirty = false;
    let out = re
        .replace_all(body, |caps: &regex::Captures| {
            let block = &caps[0];
            let Some(id) = extract_tweet_id(block) else {
                return block.to_string();
            };
            match get_or_fetch(&id, &mut cache, &mut dirty) {
                Some(tweet) => render_card(&tweet),
                None => block.to_string(),
            }
        })
        .into_owned();

    if dirty {
        save_cache(&cache);
    }
    out
}

/// Pull the tweet id out of a `<blockquote class="twitter-tweet">` block by
/// taking the *last* `status/<id>` link — X's standard embed snippet always
/// ends with a permalink of that shape (the "August 27, 2019"-style link).
fn extract_tweet_id(block: &str) -> Option<String> {
    static ID_RE: OnceLock<Regex> = OnceLock::new();
    let re = ID_RE.get_or_init(|| Regex::new(r"status/(\d+)").unwrap());
    re.captures_iter(block).last().map(|c| c[1].to_string())
}

fn get_or_fetch(id: &str, cache: &mut Cache, dirty: &mut bool) -> Option<Tweet> {
    if let Some(tweet) = cache.get(id) {
        return Some(tweet.clone());
    }
    match fetch_tweet(id) {
        Ok(tweet) => {
            cache.insert(id.to_string(), tweet.clone());
            *dirty = true;
            Some(tweet)
        }
        Err(e) => {
            eprintln!(
                "ssg: warning: couldn't fetch tweet {id} ({e}); leaving raw embed markup"
            );
            None
        }
    }
}

fn fetch_tweet(id: &str) -> Result<Tweet, String> {
    let token = syndication_token(id);
    let url =
        format!("https://cdn.syndication.twimg.com/tweet-result?id={id}&lang=en&token={token}");
    let resp = ureq::get(&url)
        .set(
            "User-Agent",
            "Mozilla/5.0 (compatible; alkzar-ssg/1.0; +https://alkzar.cl)",
        )
        .call()
        .map_err(|e| e.to_string())?;
    let body = resp.into_string().map_err(|e| e.to_string())?;
    serde_json::from_str::<Tweet>(&body).map_err(|e| format!("{e} (body: {body})"))
}

/// X's syndication endpoint requires a `token` derived from the tweet id.
/// Reverse-engineered from `widgets.js`:
/// `((Number(id) / 1e15) * Math.PI).toString(36).replace(/(0+|\.)/g, '')`.
fn syndication_token(id: &str) -> String {
    let n: f64 = id.parse().unwrap_or(0.0);
    let value = (n / 1e15) * std::f64::consts::PI;
    to_base36(value)
        .chars()
        .filter(|c| *c != '0' && *c != '.')
        .collect()
}

fn to_base36(value: f64) -> String {
    const DIGITS: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut int_part = value.trunc();
    let mut frac_part = value - int_part;

    let mut int_digits = Vec::new();
    if int_part == 0.0 {
        int_digits.push(b'0');
    } else {
        while int_part > 0.0 {
            let digit = (int_part % 36.0) as usize;
            int_digits.push(DIGITS[digit]);
            int_part = (int_part / 36.0).trunc();
        }
        int_digits.reverse();
    }

    let mut frac_digits = Vec::new();
    for _ in 0..20 {
        frac_part *= 36.0;
        let d = (frac_part.trunc() as usize).min(35);
        frac_digits.push(DIGITS[d]);
        frac_part -= d as f64;
    }

    let mut s = String::from_utf8(int_digits).unwrap();
    s.push('.');
    s.push_str(&String::from_utf8(frac_digits).unwrap());
    s
}

fn load_cache() -> Cache {
    std::fs::read_to_string(CACHE_PATH)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_cache(cache: &Cache) {
    if let Ok(json) = serde_json::to_string_pretty(cache) {
        let _ = std::fs::write(CACHE_PATH, json);
    }
}

/// Splice entity links into `text` at X's UTF-16 code-unit offsets, escaping
/// everything else. Malformed/overlapping indices are skipped rather than
/// risking corrupted output.
fn linkify_text(text: &str, entities: &TweetEntities) -> String {
    let utf16: Vec<u16> = text.encode_utf16().collect();
    let mut spans: Vec<(usize, usize, String)> = Vec::new();

    for u in &entities.urls {
        spans.push((
            u.indices[0],
            u.indices[1],
            format!(
                r#"<a href="{}" target="_blank" rel="noopener noreferrer">{}</a>"#,
                html_escape(&u.expanded_url),
                html_escape(&u.display_url)
            ),
        ));
    }
    for h in &entities.hashtags {
        spans.push((
            h.indices[0],
            h.indices[1],
            format!(
                r#"<a href="https://twitter.com/hashtag/{0}" target="_blank" rel="noopener noreferrer">#{0}</a>"#,
                html_escape(&h.text)
            ),
        ));
    }
    for m in &entities.user_mentions {
        spans.push((
            m.indices[0],
            m.indices[1],
            format!(
                r#"<a href="https://twitter.com/{0}" target="_blank" rel="noopener noreferrer">@{0}</a>"#,
                html_escape(&m.screen_name)
            ),
        ));
    }
    spans.sort_by_key(|s| s.0);

    let mut out = String::new();
    let mut cursor = 0usize;
    for (start, end, html) in spans {
        if start < cursor || end > utf16.len() || start > end {
            continue;
        }
        out.push_str(&html_escape(&String::from_utf16_lossy(
            &utf16[cursor..start],
        )));
        out.push_str(&html);
        cursor = end;
    }
    out.push_str(&html_escape(&String::from_utf16_lossy(&utf16[cursor..])));
    out
}

fn render_card(tweet: &Tweet) -> String {
    let text_html = linkify_text(&tweet.text, &tweet.entities);
    let verified = tweet.user.is_blue_verified || tweet.user.verified;
    let verified_badge = if verified {
        r#"<svg class="tweet-card-badge" viewBox="0 0 22 22" aria-hidden="true"><path fill="currentColor" d="M20.396 11c-.018-.646-.215-1.275-.57-1.816-.354-.54-.852-.972-1.438-1.246.223-.607.27-1.264.14-1.897-.131-.634-.437-1.218-.882-1.687-.47-.445-1.053-.75-1.687-.882-.633-.13-1.29-.083-1.897.14-.273-.587-.704-1.084-1.245-1.439C12.275.215 11.647.017 11 0c-.646.017-1.273.215-1.813.57-.54.354-.972.852-1.246 1.438-.607-.223-1.264-.27-1.897-.14-.634.132-1.218.437-1.687.882-.445.47-.75 1.053-.882 1.687-.13.633-.083 1.29.14 1.897-.587.274-1.084.705-1.439 1.246C.215 8.725.017 9.353 0 10c.017.647.215 1.275.57 1.816.354.54.852.972 1.438 1.246-.223.607-.27 1.264-.14 1.897.132.634.437 1.217.882 1.687.47.444 1.053.75 1.687.882.633.13 1.29.083 1.897-.14.274.586.705 1.084 1.246 1.438.54.354 1.167.552 1.813.569.647-.017 1.275-.215 1.816-.57.54-.354.972-.852 1.246-1.438.607.223 1.264.27 1.897.14.634-.132 1.217-.437 1.687-.882.444-.47.75-1.053.882-1.687.13-.633.083-1.29-.14-1.897.586-.274 1.084-.705 1.438-1.246.354-.54.552-1.169.57-1.816zM9.662 14.85l-3.429-3.428 1.293-1.293 2.136 2.136 4.65-4.65 1.293 1.293-5.943 5.942z"></path></svg>"#
    } else {
        ""
    };
    let date = crate::format_date(&tweet.created_at);
    let handle = html_escape(&tweet.user.screen_name);
    let name = html_escape(&tweet.user.name);
    let avatar = html_escape(&tweet.user.profile_image_url_https);
    let iso = html_escape(&tweet.created_at);
    let tweet_url = format!(
        "https://twitter.com/{}/status/{}",
        tweet.user.screen_name, tweet.id_str
    );

    format!(
        r#"<div class="tweet-card">
<a class="tweet-card-head" href="{tweet_url}" target="_blank" rel="noopener noreferrer">
<img class="tweet-card-avatar" src="{avatar}" alt="" loading="lazy" width="40" height="40">
<span class="tweet-card-who"><span class="tweet-card-name">{name}{verified_badge}</span><span class="tweet-card-handle">@{handle}</span></span>
<svg class="tweet-card-brand" viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M18.9 1.153h3.68l-8.04 9.19L24 22.846h-7.406l-5.8-7.584-6.638 7.584H.474l8.6-9.83L0 1.154h7.594l5.243 6.932ZM17.61 20.65h2.039L6.486 3.24H4.298Z"></path></svg>
</a>
<p class="tweet-card-text">{text_html}</p>
<div class="tweet-card-foot">
<time datetime="{iso}">{date}</time>
<span class="tweet-card-stats"><span>{likes} Likes</span><span>{replies} Replies</span></span>
</div>
<a class="tweet-card-view" href="{tweet_url}" target="_blank" rel="noopener noreferrer">View on X</a>
</div>"#,
        likes = tweet.favorite_count,
        replies = tweet.conversation_count,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tweet() -> Tweet {
        Tweet {
            id_str: "1166347201766780928".to_string(),
            text: "hello world".to_string(),
            entities: TweetEntities::default(),
            user: TweetUser {
                name: "Suzana Ilić".to_string(),
                screen_name: "suzatweet".to_string(),
                profile_image_url_https: "https://pbs.twimg.com/a.jpg".to_string(),
                is_blue_verified: true,
                verified: false,
            },
            created_at: "2019-08-27T13:50:17.000Z".to_string(),
            favorite_count: 182,
            conversation_count: 5,
        }
    }

    #[test]
    fn extract_tweet_id_picks_the_last_status_link() {
        let block = r#"<blockquote class="twitter-tweet"><a href="https://twitter.com/x/status/1">one</a><a href="https://twitter.com/suzatweet/status/1166347201766780928?ref_src=twsrc%5Etfw">August 27, 2019</a></blockquote>"#;
        assert_eq!(
            extract_tweet_id(block).as_deref(),
            Some("1166347201766780928")
        );
    }

    #[test]
    fn extract_tweet_id_none_without_status_link() {
        let block = r#"<blockquote class="twitter-tweet"><p>no link here</p></blockquote>"#;
        assert_eq!(extract_tweet_id(block), None);
    }

    // Tokens verified against the live endpoint for these two ids while
    // building this feature (both returned valid tweet JSON).
    #[test]
    fn syndication_token_matches_known_values() {
        assert_eq!(
            syndication_token("1166347201766780928"),
            "2ts6rewgt6hlb6875bfbt9"
        );
        assert_eq!(
            syndication_token("1326054980134973442"),
            "37pxa9dtow8abcs4brqw7b9"
        );
    }

    #[test]
    fn linkify_text_escapes_plain_text_with_no_entities() {
        let out = linkify_text("<script> & stuff", &TweetEntities::default());
        assert_eq!(out, "&lt;script&gt; &amp; stuff");
    }

    #[test]
    fn linkify_text_splices_entities_at_correct_offsets() {
        let text = "check #rust out";
        let entities = TweetEntities {
            urls: vec![],
            hashtags: vec![IndexedText {
                indices: [6, 11],
                text: "rust".to_string(),
            }],
            user_mentions: vec![],
        };
        let out = linkify_text(text, &entities);
        assert_eq!(
            out,
            r#"check <a href="https://twitter.com/hashtag/rust" target="_blank" rel="noopener noreferrer">#rust</a> out"#
        );
    }

    #[test]
    fn linkify_text_skips_malformed_indices_rather_than_panicking() {
        let text = "short";
        let entities = TweetEntities {
            urls: vec![],
            hashtags: vec![IndexedText {
                indices: [10, 20],
                text: "oob".to_string(),
            }],
            user_mentions: vec![],
        };
        let out = linkify_text(text, &entities);
        assert_eq!(out, "short");
    }

    #[test]
    fn render_card_contains_author_text_and_stats() {
        let html = render_card(&sample_tweet());
        assert!(html.contains("Suzana Ili\u{107}"), "got: {html}");
        assert!(html.contains("@suzatweet"), "got: {html}");
        assert!(html.contains("hello world"), "got: {html}");
        assert!(html.contains("182 Likes"), "got: {html}");
        assert!(html.contains("5 Replies"), "got: {html}");
        assert!(html.contains("tweet-card-badge"), "expected verified badge: {html}");
        assert!(
            html.contains("https://twitter.com/suzatweet/status/1166347201766780928"),
            "got: {html}"
        );
    }

    #[test]
    fn render_card_omits_badge_when_not_verified() {
        let mut tweet = sample_tweet();
        tweet.user.is_blue_verified = false;
        tweet.user.verified = false;
        let html = render_card(&tweet);
        assert!(!html.contains("tweet-card-badge"), "got: {html}");
    }

    #[test]
    fn render_tweets_is_a_no_op_without_embedded_tweets() {
        let body = "Just some regular text with no tweets.";
        assert_eq!(render_tweets(body), body);
    }
}
