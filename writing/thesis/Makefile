MAKE = make
PDFLATEX = pdflatex --shell-escape -interaction=nonstopmode 
BIBTEX = bibtex 

FNAME = Masters-Thesis_Filter

.PHONY: all clean clean-all

all:
	${PDFLATEX} ${FNAME}
	${BIBTEX} ${FNAME}
	${PDFLATEX} ${FNAME}
	${PDFLATEX} ${FNAME}

clean:
	rm -f ${FNAME}.aux
	rm -f ${FNAME}.bbl
	rm -f ${FNAME}.blg
	rm -f ${FNAME}.dvi
	rm -f ${FNAME}.log
	rm -f ${FNAME}.toc
	rm -f ${FNAME}.out
	rm -f ${FNAME}.synctex.gz
	${MAKE} -C chapters clean

clean-all: clean
	rm -f ${FNAME}.pdf
	rm -f ${FNAME}.ps
