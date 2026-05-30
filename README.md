# alkzar.cl

Source for [alkzar.cl](https://alkzar.cl). Custom Rust static site generator; see `plan.md` for design.

## Build

```
cargo run --release -p ssg -- build
```

Produces a static site in `public/`.

## Migration

`master` is the legacy Hugo site (deployed to Netlify). `rust-ssg` is the new Rust site mid-build. DNS will cut over once parity is reached.
