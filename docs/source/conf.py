"""Sphinx configuration for the rho prediction model documentation."""

project = "Rho Prediction Model"
copyright = "2026, Kiltefox"
author = "Kiltefox"

release = "0.1"
version = "0.1.0"

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = []
language = "zh_CN"

html_theme = "sphinx_rtd_theme"
html_title = "Rho Prediction Model"
html_static_path = []

epub_show_urls = "footnote"
