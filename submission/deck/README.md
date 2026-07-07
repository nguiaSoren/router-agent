# TokenGolf deck + brand assets

8-slide Track-1 pitch (terminal scoreboard aesthetic, 16:9) plus GitHub brand assets.

| file | size / ratio | use |
|---|---|---|
| `out/TokenGolf_deck.pdf` | 1920x1080 x8 | Slide Presentation (upload to lablab) |
| `out/cover.png` | 3840x2160 (16:9) | hackathon Cover Image |
| `out/github_card.png` | 2560x1280 (2:1) | GitHub social preview + README banner |
| `out/logo.png` | 1024x1024 (1:1) | GitHub avatar / repo icon |
| `out/NN_sNN.png` | 1920x1080 | per-slide PNGs |

## Re-render
```
npm i playwright        # chromium is cached
node render_deck.cjs    # slides -> out/*.png + out/TokenGolf_deck.pdf
node render_card.cjs    # -> out/github_card.png
node render_logo.cjs    # -> out/logo.png
```
Edit the matching `*.html` (deck slides are stacked 1920x1080 .slide sections; card 1280x640; logo 512x512). No node_modules committed.
