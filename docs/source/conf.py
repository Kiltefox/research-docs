"""Sphinx configuration for the rho prediction model documentation."""

project = "学习文档"
copyright = "2026, Kiltefox"
author = "Kiltefox"

release = "0.3.0"
version = "0.3.0"

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.mathjax",
    "sphinxcontrib.mermaid",
]

templates_path = ["_templates"]
exclude_patterns = []
language = "zh_CN"

html_theme = "sphinx_rtd_theme"
html_title = "学习文档"
html_static_path = []

epub_show_urls = "footnote"
