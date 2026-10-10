# Build the LaTeX deliverables. Requires latexmk (TeX Live / TinyTeX).
# The Word copies (main.docx, for Google Drive) also require pandoc.
DOCS := proposal report

.PHONY: all docx $(DOCS) clean

all: $(DOCS) docx

$(DOCS):
	cd docs/$@ && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

# main.tex -> main.docx in each folder, for Google Drive: Times New Roman,
# black only, justified, numbered sections and citations linked to the
# reference list (IEEE style). Template: docs/tools/reference.docx, built by
# docs/tools/make_reference_docx.py. Table header rows get a thick rule
# (docs/tools/docx_fix_tables.py), which Google Docs would otherwise drop.
docx:
	for d in $(DOCS); do (cd docs/$$d && pandoc main.tex -o main.docx --citeproc \
		--bibliography=../references.bib --csl=../tools/ieee.csl \
		-M link-citations=true -M link-bibliography=true -M reference-section-title=References \
		--reference-doc=../tools/reference.docx --number-sections \
		--lua-filter=../tools/docx-filter.lua && \
		python3 ../tools/docx_fix_bookmarks.py main.docx && \
		python3 ../tools/docx_fix_tables.py main.docx) || exit 1; done

clean:
	for d in $(DOCS); do (cd docs/$$d && latexmk -c main.tex); done
