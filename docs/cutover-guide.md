# Cutover guide: Hugo on Netlify → Rust SSG on Cloudflare Pages

This is a step-by-step runbook for launching the new Rust-based site at
`alkzar.cl`. It assumes no prior knowledge of Cloudflare, DNS, or web
infrastructure. Every step explains *what* you're doing and *why*, with a
verification check at the end so you know it worked before moving on.

## What this guide gets you to

By the end:
- The new Rust-generated site is live at `https://alkzar.cl`
- DNS is managed at Cloudflare (still registered at NIC Chile)
- The old Hugo site on Netlify is dormant (kept as rollback) for 30 days, then
  archived
- All 20 GitHub issues are auto-closed via the final cutover PR

## Time estimate

Total **active** clicking time: ~1 hour. Total **elapsed** time: 1–2 days
because of DNS propagation. You can do the active work in two sittings:

1. **First sitting (~30 min)**: Cloudflare account + Pages project + first
   preview deploy + preview verification.
2. **Second sitting (~30 min, a day later)**: DNS delegation at NIC Chile +
   custom domain wiring + final merge to master.

## What you'll need

- A web browser
- Terminal with `git` and `gh` (already set up)
- About 30 minutes of focused time per sitting
- Patience with DNS propagation (it's just waiting)

## Where the project stands right now

- `master` branch is the **old Hugo site**, still deployed to Netlify, still
  serving `alkzar.cl`. Untouched by anything you'll do until the final step.
- `rust-ssg` branch is the **new site**, fully built and tested but not yet
  deployed anywhere. This is what gets launched.
- The build (`cargo run --release -p ssg -- build`) produces a `public/`
  directory you can browse locally. Cloudflare Pages will do the same in the
  cloud and serve the result.

Everything below is **reversible** until step 10. If something feels wrong,
stop and ask — nothing happens to the live site until DNS changes hands.

---

# Step 1 — Create a Cloudflare account + Pages project

**Why**: Cloudflare Pages is where the new site will live. It builds and serves
static files from a CDN with edge servers worldwide — including in Santiago and
São Paulo, which is why we picked it over Netlify (more local-to-Chile).

**Time**: ~10 minutes.

### 1a. Sign up (skip if you have a CF account)

1. Go to https://dash.cloudflare.com/sign-up
2. Sign up with your email (recommend the same email associated with your
   GitHub account, but not required)
3. Verify the email link Cloudflare sends you
4. You'll land on the Cloudflare dashboard. There's an "Add a Site" prompt — **don't
   click it yet**. We're not adding the DNS zone here; that happens later.

### 1b. Create the Pages project

1. In the Cloudflare dashboard's left sidebar, look for **Workers & Pages** (or
   just **Pages** depending on layout). Click it.
2. Click **Create** (or **Create application**)
3. Choose the **Pages** tab (not Workers)
4. Choose **Upload assets** (NOT "Connect to Git"). We're using
   GitHub Actions to push builds, not Cloudflare's Git integration. (Why: we
   need Rust support which Cloudflare's built-in build doesn't have first-class.
   Our GitHub Actions workflow already does the Rust build and just uploads
   the result.)
5. Name the project **`alkzar`** (lowercase, no spaces). This matches what the
   workflow references in `.github/workflows/deploy.yml`. If you pick a
   different name you'll have to update the workflow YAML.
6. For the initial upload, just drop in any small empty folder (a placeholder
   `index.html` with "hello" is fine) so the project gets created. The real
   deploys come from GitHub Actions.

### Verify this step worked

You should see the Pages project listed with a temporary URL like
`https://alkzar.pages.dev`. Visiting that URL shows your placeholder. **Don't
try to add a custom domain yet** — that's step 8.

---

# Step 2 — Create the API token + find your Account ID

**Why**: GitHub Actions needs credentials to push builds to Cloudflare. We
generate a scoped API token (limited to your Pages project, can't do anything
else) and read your CF Account ID off the dashboard.

**Time**: ~5 minutes.

### 2a. Create the API token

1. In Cloudflare dashboard, click your **profile icon (top-right)** → **My
   Profile**
2. Click **API Tokens** in the left sidebar
3. Click **Create Token**
4. Scroll to find a template called **"Edit Cloudflare Workers"** or use
   **"Custom token"** — either works. Custom is safer because we can scope
   tighter.
5. Use **Create Custom Token** with these settings:
   - **Token name**: `alkzar-pages-deploy` (anything memorable)
   - **Permissions** — add ONE row:
     - Account → **Cloudflare Pages** → **Edit**
   - **Account Resources**: include the specific account that owns the `alkzar`
     project (usually shown as your email or username)
   - **TTL** (expiration): pick something between 90 days and 1 year. Don't
     pick "no expiration" — rotating tokens occasionally is good hygiene.
6. Click **Continue to summary**, then **Create Token**
7. You'll see the token ONCE — looks like `8a7b6c5d4e3f...` — **copy it
   immediately and paste into a note**. You can't view it again after closing
   this page. (If you lose it, no harm done — delete and create a new one.)

### 2b. Find your Account ID

1. Go back to the main dashboard (click the Cloudflare logo top-left)
2. In the right-hand sidebar of the dashboard, you'll see **API** section with
   **Account ID** — a long hex string like `a1b2c3d4e5f6...`
3. Copy this too. It's not secret-secret (it identifies your account, not
   authenticates it) but you'll need it next.

### Verify this step worked

You have two values copied somewhere:
- API token: ~40-char string starting with letters/numbers
- Account ID: 32-char hex string

If you lost the API token, just regenerate — no harm.

---

# Step 3 — Add the credentials as GitHub secrets

**Why**: GitHub Actions reads these as `${{ secrets.CF_API_TOKEN }}` and
`${{ secrets.CF_ACCOUNT_ID }}` from the deploy workflow. Setting them as
*secrets* (not regular variables) means they're encrypted and never appear in
logs.

**Time**: ~3 minutes.

1. Open https://github.com/alcazar90/personal_website/settings/secrets/actions
   (or: repo page → **Settings** tab → **Secrets and variables** → **Actions**
   in the left sidebar)
2. Click **New repository secret**
3. First secret:
   - **Name**: `CF_API_TOKEN` (exactly this; case-sensitive)
   - **Value**: paste the API token from step 2a
   - Click **Add secret**
4. Click **New repository secret** again
5. Second secret:
   - **Name**: `CF_ACCOUNT_ID`
   - **Value**: paste the Account ID from step 2b
   - Click **Add secret**

### Verify this step worked

The secrets page now lists `CF_API_TOKEN` and `CF_ACCOUNT_ID`. You can't see
their values (that's the point), but you can update them anytime.

---

# Step 4 — Trigger the first preview deploy

**Why**: This proves the GitHub Actions workflow can build the site and push it
to Cloudflare Pages. The result will be visible at `https://alkzar.pages.dev`
(the temporary URL CF gave you in step 1).

**Time**: ~5 minutes (workflow takes ~2 min to run).

1. Open https://github.com/alcazar90/personal_website/actions
2. In the left sidebar, click the workflow named **build-and-deploy**
3. On the right side you'll see a **Run workflow** dropdown — click it
4. Make sure the branch dropdown says `rust-ssg` (NOT `master`)
5. Click the green **Run workflow** button

The workflow starts immediately. It does three things:
1. Compiles the Rust SSG (~30s with cache, ~3 min cold)
2. Runs the `ssg build` command, producing `public/`
3. Uploads `public/` to your Cloudflare Pages project

### Watch the run

- Click into the running workflow to see step-by-step logs
- If it succeeds: green checkmark on every step
- If it fails: usually one of two things:
  - **"401 unauthorized"** or **"403 forbidden"**: the CF API token doesn't
    have permission. Recheck step 2a — the token must have
    `Account > Cloudflare Pages > Edit` permission. Delete and recreate.
  - **"project not found"**: the project name in `.github/workflows/deploy.yml`
    (`alkzar`) doesn't match what you named the Pages project. Either rename
    the Pages project in CF dashboard, OR edit the workflow YAML to match.

### Verify this step worked

In Cloudflare dashboard → Workers & Pages → `alkzar` → **Deployments** tab,
you'll see a deployment timestamped just now. Click it — you get a URL like
`https://abc1234.alkzar.pages.dev` (one-off per deployment) and also the main
`https://alkzar.pages.dev`.

Open `https://alkzar.pages.dev`. **This is your new site.** It should show:
- The post listing (13 posts, newest first)
- Light or dark theme matching your system preference
- A theme toggle button in the header

If you see your placeholder from step 1b instead, wait 30 seconds and refresh —
Cloudflare needs a moment to propagate the new deploy.

---

# Step 5 — Verify the preview thoroughly

**Why**: This is the only chance to catch problems before DNS cutover. After
DNS changes, problems become user-visible.

**Time**: ~20-30 minutes if you're being careful.

The preview URL is `https://alkzar.pages.dev`. Run through this checklist:

### 5a. Smoke tests (5 min)

- [ ] **Home page**: post list shows all 13 posts with dates, newest first
- [ ] **Click 3 different posts**, including:
  - One short text-only post (e.g. *Notes about internalizing vim keybindings*)
  - One with code (e.g. *Fastai Chapter 1*)
  - The math-heavy post (*Reinforcement Learning*)
- [ ] **About page**: navigate via header menu, content displays
- [ ] **Theme toggle**: click the sun/moon button in the header. Page colors
      flip between paper-light and ink-dark. Refresh — your choice persists.
- [ ] **404 page**: visit `https://alkzar.pages.dev/this-does-not-exist` —
      shows the styled 404 page, not a Cloudflare error page

### 5b. Math rendering (10 min) — run the MathML spike

Open `docs/mathml-spike.md` (in this repo or on GitHub) and follow the
verification checklist there:

1. Open the RL post on **macOS Chrome**, **macOS Firefox**, **macOS Safari**,
   and if possible **iOS Safari** (on your phone)
2. Visually inspect the 5 sample math constructs (fractions, integrals,
   matrices, etc.)
3. Fill in the per-browser checklist

**Most likely outcome**: Chrome and Firefox render perfectly. Safari (especially
older versions) may have minor spacing quirks on fractions or matrices but is
generally readable. If Safari is **unreadable** on any sample, the spike doc
has a decision tree pointing to a follow-up KaTeX-fallback issue.

### 5c. Image paths (5 min)

The migrator left several posts with Hugo-style `/img/...` references. These
should resolve correctly since `content/static/img/` copies to `public/img/`,
but verify:

1. Open these posts on the preview URL and check that images load (no broken
   image icons):
   - `crear-entornos-virtuales-en-python...`
   - `berkson-s-paradox`
   - `taylor-approximation-and-jax`
   - `directional-derivatives-and-jax`
   - `fastai-chapter-1`
   - `ukiyo-e-style-postal-generator-app`
2. If any are broken: the path probably has unexpected casing or whitespace.
   Note the post slug and image path; fix can wait until after cutover.

### 5d. Bundle size sanity (2 min)

Open https://radar.cloudflare.com/scan and paste `https://alkzar.pages.dev`. It
runs a quick analysis and reports total page weight. Target: home page and a
typical post ≤ 20 kB HTML+CSS+JS (images budgeted separately).

If a specific post is unusually heavy (>50 kB HTML), it usually means embedded
images or SVG inside the markdown body. Note it for a future audit; not
blocking.

### Verify this step worked

Every checkbox in 5a passes. Math (5b) is readable in at least Chrome and
Firefox (Safari can have minor issues). Images load (5c). Bundle is reasonable
(5d). You're ready for cutover.

If something is genuinely broken, stop here and ping me before continuing — DNS
cutover with a broken preview means you ship broken to users.

---

# Step 6 — Decide the fate of 3 skipped posts (editorial, optional)

The migrator skipped 3 posts because they were heavy on R-rendered figures
(decision trees, ggplot output, k-means iterations):

- `2017-12-26 — What about tree models`
- `2018-01-16 — A data wrangling case with R`
- `2018-05-17 — A brief post about k-means`

**Option A: Delete them**. Simplest. The PR #25 description says these are old
R tutorials whose value lives in the rendered figures — rewriting them is more
work than they're worth as historical archive.

**Option B: Manually rewrite**. Requires re-running the R code, exporting the
figures as PNG, embedding them in clean Markdown, rewriting any narrative that
referenced computed values. Probably half a day per post.

If A: nothing to do — they're already absent from the new site.

If B: open a new GitHub issue *after cutover* called something like "Rewrite
legacy R posts" and tackle one at a time. No rush.

---

# Step 7 — Activate Giscus comments (optional)

**Why**: The post template has a comments mount that auto-shows when you add a
`[giscus]` block to `content/config.toml`. Without that block, post pages don't
ship any comment code. So this step is opt-in.

**Time**: ~15 minutes.

### 7a. Prepare the GitHub side

1. Open https://github.com/alcazar90/personal_website/settings
2. Scroll to **Features** section → find **Discussions** checkbox → enable it
3. Go to https://github.com/alcazar90/personal_website/discussions
4. You'll be prompted to set up categories. Either accept the defaults OR
   create a dedicated category named **"Comments"** (type: *Announcement*)
   for posts. Recommended: create the dedicated category.

### 7b. Install the giscus app on the repo

1. Go to https://github.com/apps/giscus
2. Click **Install**
3. Choose **Only select repositories** → select `personal_website`
4. Click **Install**

### 7c. Get the configuration values

1. Go to https://giscus.app
2. Fill in the form:
   - **Repository**: `alcazar90/personal_website`
   - **Page ↔ Discussions mapping**: pick **"pathname"** (each post URL maps
     to its own discussion)
   - **Discussion Category**: pick **"Comments"** (or whatever you named it)
   - Leave other settings at defaults
3. Scroll down — giscus generates a `<script>` snippet. **You don't paste
   that snippet anywhere.** What you need are two of the `data-*` values from
   inside the snippet:
   - `data-repo-id="..."` — copy the value (looks like `R_kgDOXXXX...`)
   - `data-category-id="..."` — copy the value (looks like `DIC_kwDOXXXX...`)

### 7d. Wire it into your config

Add this block to the end of `content/config.toml` on the `rust-ssg` branch
(or any branch — it'll travel into master at cutover):

```toml
[giscus]
repo = "alcazar90/personal_website"
repo_id = "R_kgDO..."           # paste from giscus.app
category = "Comments"
category_id = "DIC_kwDO..."     # paste from giscus.app
mapping = "pathname"
reactions_enabled = "1"
input_position = "bottom"
strict = "0"
loading = "lazy"
```

Commit + push to `rust-ssg`. The GitHub Actions workflow will run automatically
and re-deploy. Within ~2 minutes, every post page will have a comments section
at the bottom that follows your light/dark theme.

### Verify this step worked

Open any post on `https://alkzar.pages.dev` → scroll down → you see "Comments"
section with a sign-in-with-GitHub prompt. Post a test comment as a check; it
appears both on the page and in GitHub Discussions.

---

# Step 8 — Delegate DNS for `alkzar.cl` to Cloudflare

**Why**: Right now, when someone types `alkzar.cl`, their browser asks NIC
Chile's nameservers "where does this go?" — and NIC Chile says "Netlify".
We want CF to answer that question instead so we can point at Cloudflare Pages.

**Important distinction**: you're **delegating DNS**, not **transferring the
domain**. The domain stays registered at NIC Chile (you still pay them
annually, same as today). You're just changing which servers answer DNS
queries.

**Time**: ~10 minutes active + 1–6 hours for propagation.

### 8a. Add the zone in Cloudflare

1. In Cloudflare dashboard, click **Add a Site** (the prompt from step 1, or
   from the **Websites** sidebar entry)
2. Enter `alkzar.cl` (no `https://`, no `www`)
3. Choose the **Free** plan
4. Cloudflare scans your current DNS records (the ones at NIC Chile) and
   imports them. You'll see a list — usually a few A records, maybe MX records
   if you have email. **Review them and keep what you need** (most likely
   defaults are fine for a personal blog).
5. **Important**: Cloudflare will now show you **two nameservers** with names
   like `lila.ns.cloudflare.com` and `kurt.ns.cloudflare.com` (the names are
   randomly assigned). **Write these down** — you'll need them next.

### 8b. Update nameservers at NIC Chile

NIC Chile's panel is in Spanish. Don't worry, the steps are short.

1. Go to https://www.nic.cl
2. Click **Iniciar sesión** (top-right) → log in with your NIC Chile account
3. In your dashboard, find `alkzar.cl` in your domain list → click it
4. Look for **Cambio de DNS** (Change DNS) or **Servidores de nombres** (Name
   servers) — it's usually one of the main menu options for a domain
5. You'll see your current nameservers (probably Netlify-related). **Replace
   them** with the two Cloudflare nameservers from step 8a:
   - Primer servidor de nombres: `lila.ns.cloudflare.com` (your value)
   - Segundo servidor de nombres: `kurt.ns.cloudflare.com` (your value)
   - Remove any third/fourth ones if NIC Chile asks for them — Cloudflare uses
     only two
6. Save the change. NIC Chile may send a confirmation email — click the
   confirmation link if so.

### 8c. Tell Cloudflare you've done it

1. Back in Cloudflare's dashboard, go to your `alkzar.cl` zone
2. Click **Check nameservers** (or **Re-check now**)
3. Cloudflare polls every few minutes to detect the change

### 8d. Wait for DNS propagation

This is the **slow** part. NIC Chile's nameserver changes typically take
**1–6 hours** to propagate worldwide (gTLDs like .com propagate faster; `.cl`
is slower).

You can monitor propagation:
- Run `dig alkzar.cl NS +short` in terminal — once it returns Cloudflare's
  nameservers (`*.ns.cloudflare.com`), propagation has reached your resolver
- Check https://www.whatsmydns.net/#NS/alkzar.cl — a global map showing which
  resolvers see the new nameservers

You can do everything in step 9 BEFORE propagation completes (CF doesn't need
DNS to be live yet to attach a custom domain), but the domain won't actually
resolve to CF until propagation finishes.

### Verify this step worked

- Cloudflare dashboard shows **"alkzar.cl is now using Cloudflare"** (a
  banner change you can't miss)
- `dig alkzar.cl NS +short` returns the two `*.ns.cloudflare.com` servers
- `dig alkzar.cl` returns an IP address Cloudflare manages

If the wait is taking >12 hours, recheck the NIC Chile panel — sometimes the
change doesn't save the first time. Re-enter the nameservers and save again.

---

# Step 9 — Attach the custom domain to your Pages project

**Why**: Tell Cloudflare Pages "serve the `alkzar` project at the domain
`alkzar.cl`". This automatically sets up an HTTPS certificate (~10 seconds).

**Time**: ~5 minutes.

1. CF dashboard → **Workers & Pages** → `alkzar` project → **Custom domains**
   tab
2. Click **Set up a custom domain**
3. Enter `alkzar.cl` → **Continue**
4. Cloudflare detects that you also manage DNS for this zone, so it offers
   to **add the DNS record automatically**. Accept this.
5. Cloudflare provisions a TLS certificate (10–60 seconds). The status will
   change from **Initializing** to **Active**.

### 9a. (Recommended) Redirect `www.alkzar.cl` to `alkzar.cl`

Most blogs are accessed at the apex (`alkzar.cl`), but some people type
`www.`. Set up a redirect so both work.

1. In CF dashboard → `alkzar.cl` zone → **Rules** → **Redirect Rules**
2. Click **Create rule**
3. Name: `www to apex`
4. Match: **Hostname** equals `www.alkzar.cl`
5. Then: **Static redirect** → URL: `https://alkzar.cl/$1` → Status code:
   **301 (Permanent)**
6. Save

### Verify this step worked

- Open `https://alkzar.cl` — you see the new Rust site
- Open `https://www.alkzar.cl` — browser redirects to `https://alkzar.cl`
- The padlock icon in your browser shows a valid HTTPS certificate

If `https://alkzar.cl` shows the OLD Hugo site instead of the new one, two
possibilities:
1. **DNS propagation hasn't reached your resolver yet** — wait, retry in 30 min
2. **Browser cache** — try in a private/incognito window

---

# Step 10 — Final merge: `rust-ssg` → `master`

**Why**: This is the symbolic cutover. Master is the default branch; merging
here auto-closes all 19 issues that the PRs referenced. It also means future
work happens on master like a normal repo, no special branches.

**Time**: ~2 minutes.

By this point:
- `alkzar.cl` is already serving the new site (from step 9)
- Master is still the OLD Hugo (but no one is looking at it — DNS doesn't
  point there)
- All 20 issues are still open in GitHub (because PRs targeted rust-ssg, not
  master)

### 10a. Open the cutover PR

```bash
cd /Users/cristobalalcazar/Developer/site/personal_website
git fetch origin
gh pr create \
  --repo alcazar90/personal_website \
  --base master \
  --head rust-ssg \
  --title "Cutover: Rust SSG replaces Hugo (closes #20)" \
  --body "Long-lived integration branch lands on master. Site is already live at https://alkzar.cl via Cloudflare Pages. Old Netlify deploy stays dormant for 30 days as rollback insurance.

This PR auto-closes #1, #2, #3, #4, #5, #6, #7, #8, #9, #10, #11, #12, #13, #14, #15, #16, #17, #18, #19 (already implemented on rust-ssg) and #20 (the cutover itself)."
```

### 10b. Merge the PR

```bash
gh pr merge --repo alcazar90/personal_website <PR_NUMBER> --merge
```

Use `--merge` (NOT `--squash`) for this one. We want master's history to
include every individual feature PR commit as a clear historical record. After
this, master is identical to rust-ssg.

### Verify this step worked

- GitHub repo home now shows the new Rust project structure on master
- All 20 issues are **Closed** (https://github.com/alcazar90/personal_website/issues
  shows 0 open)
- `https://alkzar.cl` still works (unaffected — DNS already points to CF)

---

# Step 11 — Wait 30 days, then retire Netlify

**Why**: If something subtle is broken with the new site (a post nobody
reads has a corrupted image, an unusual user-agent gets a weird response),
you'll want the ability to flip DNS back to Netlify within minutes. After 30
days of trouble-free operation, you can confidently kill the old setup.

**Time**: ~5 minutes, 30 days from now.

### 11a. (At day 30 or later) Disable the Netlify site

1. Go to https://app.netlify.com
2. Find the `alkzar` project (or whatever it's named there)
3. Site settings → General → **Stop builds** (so Netlify stops trying to
   redeploy on git push to master — which won't match its old branch anyway)
4. Optionally: **Archive site** (preserves it as a record but stops serving)

You can keep the Netlify free tier account around — it doesn't cost anything
and the historical builds are nice to keep.

### Verify this step worked

- `https://alkzar.cl` still works (DNS still points to Cloudflare; Netlify
  being disabled doesn't affect it)
- Netlify dashboard shows the site as archived/stopped

---

# Troubleshooting common issues

### "I made the API token wrong and broke step 4"

Just delete the token in CF profile → API Tokens → click the broken token →
**Roll** (regenerate). Then update the GitHub secret. The next push or manual
workflow run uses the new token automatically.

### "I lost the new nameservers from step 8a"

CF dashboard → `alkzar.cl` zone → **DNS** tab → **Nameservers** section in
the upper right. They're always shown there.

### "Step 8 said wait 1–6 hours but it's been 24 and nothing"

Two things to check:
1. **NIC Chile**: log in, navigate to `alkzar.cl` → confirm the nameservers
   actually saved. NIC Chile occasionally requires email confirmation that
   gets caught in spam.
2. **Cloudflare**: dashboard → `alkzar.cl` → **Overview** → if it still shows
   "Pending Nameserver Update", click **Check nameservers** manually.

### "Step 9 finished but the site shows old Hugo content"

DNS propagation hasn't reached your specific ISP/resolver yet. Try:
- Private/incognito browsing
- A different network (mobile data instead of WiFi)
- `dig +short alkzar.cl @1.1.1.1` (asks Cloudflare's own DNS — bypasses local
  cache)

Once `dig` returns a Cloudflare IP, your browser will catch up within minutes.

### "Should I rotate any of these secrets later?"

- Cloudflare API token: renew before the TTL you set (the dashboard reminds
  you). Process: create new token, update GitHub secret, delete old token.
- GitHub Discussions / Giscus: no secrets involved (`repo_id` and `category_id`
  are public identifiers, not secrets).

### "Can I roll back to Hugo on Netlify if something goes wrong?"

Yes, anytime in the first 30 days:
1. NIC Chile panel → revert nameservers to Netlify's (your old ones — Netlify
   shows them under Domain settings if you forgot)
2. Wait 1–6 hours for propagation
3. Once `dig alkzar.cl NS` shows Netlify nameservers, you're back on Hugo

Master still has the Hugo source until step 10. After step 10, master is the
new site — but you still have Netlify's last build cached on their CDN as a
safety net.

---

# Appendix: what's where

| Thing | Where |
|---|---|
| Repo source | `github.com/alcazar90/personal_website` |
| Production site (current) | Netlify, deployed from master (Hugo) |
| Production site (after cutover) | Cloudflare Pages, deployed from master (Rust) |
| Preview URL (during testing) | `https://alkzar.pages.dev` |
| DNS registrar | NIC Chile (you pay them annually, don't change this) |
| DNS hosting | NIC Chile (now) → Cloudflare (after step 8) |
| TLS certificate | Cloudflare (auto-renewed, free) |
| Build system | GitHub Actions (`.github/workflows/deploy.yml`) |
| Deploy credentials | GitHub repo secrets (`CF_API_TOKEN`, `CF_ACCOUNT_ID`) |
| Comments | (Optional) GitHub Discussions via giscus |

---

# Order of operations recap

```
First sitting (~30 min, anytime):
  1. Create CF account + Pages project        [10 min]
  2. Generate API token + grab Account ID     [5 min]
  3. Add GitHub secrets                       [3 min]
  4. Trigger first preview deploy             [5 min]
  5. Verify on preview URL                    [15-30 min]

Editorial / optional (any time):
  6. Decide fate of 3 skipped posts
  7. Set up giscus

Second sitting (~30 min, day 2):
  8. DNS delegation at NIC Chile              [10 min active + propagation wait]
  9. Custom domain in CF Pages                [5 min]
  10. Merge rust-ssg → master                 [2 min]

Day 30+:
  11. Retire Netlify                          [5 min]
```

You're done. Welcome to your tiny, fast, owned-from-byte-zero blog.
