
texfile := coadds
default: pdf

# just run it
bib: 
	bibtex ${texfile}

pdf:
	pdflatex ${texfile}
	pdflatex ${texfile}

clean:
	rm -f \
	${texfile}.dvi \
	${texfile}.ps \
	${texfile}.pdf \
	${texfile}.aux \
	${texfile}.bbl \
	${texfile}.blg \
	${texfile}.toc \
	${texfile}.log
