---
title: Simultaneous Q-Learning
summary: Investigating the properties of a non-stationary RL environment.
tags: 
- Reinforcement learning
- Multi-agent
date: "2020-02-18"

# Optional external URL for project (replaces project detail page).
external_link: ""

image:
  caption: Photo by m.prinke on flickr
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

Multiple agents learning at the same time using a value-based approach leads to a non-stationary environment since it is constantly changing based on the opponent's actions. I am investigating this system, including possible conditions on convergence and properties of this seemingly chaotic system.
