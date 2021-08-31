# in .Rprofile of the website project
if (file.exists("~/.Rprofile")) {
  base::sys.source("~/.Rprofile", envir = environment())
}
# then set options(blogdown.author = 'Your Name') in
# ~/.Rprofile
options(blogdown.author = 'Cristóbal Alcázar', blogdown.subdir = "blog",
        blogdown.ext = ".rmd", servr.daemon = TRUE,
        blogdown.hugo.version = "0.87.0",
        blogdown.server.timeout = 600)


knitr::opts_chunk$set(collapse = TRUE, comments = "#>", message = FALSE,
                      warning = FALSE, fig.align = 'center', out.width = "70%")
