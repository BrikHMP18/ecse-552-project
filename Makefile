# Build the LaTeX deliverables. Requires latexmk (TeX Live / TinyTeX).
# The Word copies (main.docx, for Google Drive) also require pandoc.
DOCS := proposal report

.PHONY: all docx $(DOCS) clean

all: $(DOCS) docx

$(DOCS):
	cd $@ && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

# main.tex -> main.docx in each folder, for Google Drive: Times New Roman,
# black only, justified, numbered sections and citations linked to the
# reference list (IEEE style). Template: tools/reference.docx, built by
# tools/make_reference_docx.py.
docx:
	for d in $(DOCS); do (cd $$d && pandoc main.tex -o main.docx --citeproc \
		--bibliography=../references.bib --csl=../tools/ieee.csl \
		-M link-citations=true -M link-bibliography=true -M reference-section-title=References \
		--reference-doc=../tools/reference.docx --number-sections \
		--lua-filter=../tools/docx-filter.lua && \
		python3 ../tools/docx_fix_bookmarks.py main.docx) || exit 1; done

clean:
	for d in $(DOCS); do (cd $$d && latexmk -c main.tex); done
