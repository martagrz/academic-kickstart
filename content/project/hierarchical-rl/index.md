---
title: Hierarchical Reinforcement Learning
summary: Applying hierarchical RL to a layered recommender system. 
tags: 
- Reinforcement Learning
- Recommender Systems
date: "2020-02-18"

# Optional external URL for project (replaces project detail page).
external_link: ""

image:
  caption: https://mc.ai/the-future-directions-of-recommender-systems/
  focal_point: Smart

links:
#- icon: twitter
#  icon_pack: fab
#  name: Follow
#  url: https://twitter.com/georgecushen
#url_code: ""
#url_pdf: ""
#url_slides: ""
#url_video: ""

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
#slides: example
---

Recommender systems often use multiple layers to present the user with items they might like. In Spotify, the carousel of recommended playlists are under a title based on the user's previous behaviour (for example "Because you listened to ...") and each carousel has multiple items. This lends itself to a hierarchical reinforcement learning algorithm for recommenders; first, the title is chosen, then the relevant playlists are chosen based on the title.
