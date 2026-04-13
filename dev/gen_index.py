# This script generates a docs/index.md on the fly from the README.md
# fixing some of the links
import re

import mkdocs_gen_files

# read README on the root of the repo
with open("README.md") as f:
    content = f.read()

# remove "docs" from gifs and images
content = re.sub(r"\]\(/?docs/", r"](", content)
# remove "docs" from src fields in html
content = re.sub(r"src=\"/?docs/", 'src="', content)
# GitHub and mkdocs slugify headings differently. "Examples & demos" becomes
# "examples--demos" on GitHub (two dashes around the stripped `&`) but
# "examples-demos" under mkdocs' default slugify. Rewrite the anchor so the
# rendered index.md has working in-page links.
content = re.sub(r"#examples--demos\b", "#examples-demos", content)

# write the index
with mkdocs_gen_files.open("index.md", "w") as fd:  #
    print(content, file=fd)
