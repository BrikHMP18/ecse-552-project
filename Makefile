# Build the LaTeX deliverables. Requires latexmk (TeX Live / TinyTeX).
DOCS := proposal report

.PHONY: all $(DOCS) clean

all: $(DOCS)

$(DOCS):
	cd $@ && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	for d in $(DOCS); do (cd $$d && latexmk -c main.tex); done
