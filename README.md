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
### Font-awesome

Adding a customg icon using font-awesome needs to add a new entry in `partials/social.html`:

```html
{{ with .Site.Params.social.cv }}
<a href="{{.}}" aria-label="CV" target="_blank"><i class="fas fa-file-pdf" aria-hidden="true"></i></a>
{{ end }}
```

Note that the `.Site.Params.social.cv` is managed by the `config.toml` file:

```toml
[params.social]
cv = "https://alkzar.cl/cv.pdf"
github = "alcazar90"
...
```

The font-awesome icon is defined in `class="fas fa-file-pdf`, for others icons
you can search on [Font Awesome](https://fontawesome.com/v5.15/icons/file-pdf?style=solid).

It requires to be supported by the font-awesome version specified in the `header.html`.

### Add 

Implement the number of minutes to read, ideas in this [post](https://kodify.net/hugo/strings/reading-time-text/).
Add the following line of code after the data in `partial/content.html` for add the
feature in each post.

```html
       / {{ math.Round (div (countwords .Content) 220.0) }} MIN READ
```
Then add on `partials/li.html` for display on the list of post, again after
the date.

```html
      <p class="meta">
        {{ if not .Date.IsZero }} {{ .Date.Format .Site.Params.dateformat | upper }} {{end}}
        - {{ math.Round (div (countwords .Content) 220.0) }} MIN READ 
```


## License

Licensed under the [MIT](https://opensource.org/licenses/MIT) License. See the [LICENSE](https://raw.githubusercontent.com/shenoybr/hugo-goa-demo/master/LICENSE) file for more details.
