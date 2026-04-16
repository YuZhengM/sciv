# Configuration file for the Sphinx documentation builder.

import sys
from pathlib import Path

HERE = Path(__file__).parent.parent
sys.path[:0] = [str(HERE.parent / "src"), str(HERE.parent), str(HERE / "extensions")]

# -- Project information

project = 'SCIV'
copyright = '2025, Zheng-Min Yu'
author = 'Zheng-Min Yu'
repository_url = "https://github.com/YuZhengM/sciv"

release = '0.0.111b1'
version = '0.0.111b1'

autosummary_generate = True

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
