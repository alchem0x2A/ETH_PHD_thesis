TEXFLAGS = -bibtex -pdf -interaction=nonstopmode -use-make
BUILD_DIR = build
MAIN="thesis"
OLD_VERSION="HEAD"
DIFF_TAG= --graphics-markup=1 --disable-citation-markup --flatten 


.PHONY: all clean $(BUILD_DIR)/$(MAIN).pdf

all: $(BUILD_DIR)/$(MAIN).pdf

diff:
	cp $(MAIN).tex $(MAIN)_new.tex
	git checkout $(OLD_VERSION) $(MAIN).tex  && cp $(MAIN).tex $(MAIN)_old.tex
	cp $(MAIN)_new.tex $(MAIN).tex
	latexdiff $(DIFF_TAG) $(MAIN)_old.tex $(MAIN)_new.tex > $(MAIN)_change.tex
	latexmk $(TEXFLAGS) -jobname=$(BUILD_DIR)/diff $(MAIN)_change.tex
	rm $(MAIN)_old.tex $(MAIN)_new.tex

$(BUILD_DIR)/thesis.pdf: thesis.tex $(BUILD_DIR)
	latexmk $(TEXFLAGS) -jobname=$(@:.pdf=) -f $<

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	latexmk $(TEXFLAGS) -jobname=$(BUILD_DIR)/ -C thesis.tex
