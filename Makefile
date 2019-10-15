TEXFLAGS = -bibtex -pdf -interaction=nonstopmode -use-make
DRAFT_FLAGS = -bibtex -pdf -interaction=nonstopmode -d	\
-pdflatex="lualatex -synctex=1 -draftmode --shell-escape"
BUILD_DIR = build
MAIN="thesis"
OLD_VERSION="HEAD"
DIFF_TAG= --graphics-markup=1 --disable-citation-markup --flatten 


.PHONY: all clean pdf diff convert

all: pdf

diff:
	cp $(MAIN).tex $(MAIN)_new.tex
	git checkout $(OLD_VERSION) $(MAIN).tex  && cp $(MAIN).tex $(MAIN)_old.tex
	cp $(MAIN)_new.tex $(MAIN).tex
	latexdiff $(DIFF_TAG) $(MAIN)_old.tex $(MAIN)_new.tex > $(MAIN)_change.tex
	latexmk $(TEXFLAGS) -jobname=$(BUILD_DIR)/diff $(MAIN)_change.tex
	rm $(MAIN)_old.tex $(MAIN)_new.tex

pdf:
	latexmk $(TEXFLAGS) -jobname=$(BUILD_DIR)/$(MAIN) -f $(MAIN).tex

draft:
	latexmk -d $(DRAFT_FLAGS) -jobname=$(BUILD_DIR)/$(MAIN) -f $(MAIN).tex

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

convert:
	python scripts/convert_svg.py

clean:
	latexmk $(TEXFLAGS) -jobname=$(BUILD_DIR)/ -C thesis.tex
