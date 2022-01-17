[![Netlify Status](https://api.netlify.com/api/v1/badges/cc461dd3-e527-458f-8da2-8037e8623765/deploy-status)](https://app.netlify.com/sites/alkzar/deploys)

# Personal Website

## About

This repo contain the source files to build my personal wrbsite using
the hugo GOA Theme](https://github.com/shenoybr/hugo-goa) by [@shenoybr](https://github.com/shenoybr).

## Notes

- In the file `config.toml`, there are the following parameters to control
the syntax highlighting:

```toml
highlightjs = true
highlightjslanguages = ["python", "r"]
highlightjsstyle = "atom-one-light"
```

The `highlightjs` is implement in `partials/footer.html` and `partials/heater.html`
respectively.

The following line controls the highlight style specified in `config.toml`.

```html
{{ if .Site.Params.extra.highlightjs }}
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.12.0/styles/{{ .Site.Params.extra.highlightjsstyle | default "default" }}.min.css">
{{ end }}
```

Note: it could be a problem between the highlight.js and bootstrap; the latter can overwrite CSS properties defined by the highlightstyle. Look at this [blog post](https://amber.rbind.io/2017/11/15/syntaxhighlighting/) about this issue.

Mathjax is avaiable via `partials/footer.html`:

```html
<script defer src="//mathjax.rstudio.com/latest/MathJax.js?config=TeX-MML-AM_CHTML"></script>
```

## License

Licensed under the [MIT](https://opensource.org/licenses/MIT) License. See the [LICENSE](https://raw.githubusercontent.com/shenoybr/hugo-goa-demo/master/LICENSE) file for more details.
