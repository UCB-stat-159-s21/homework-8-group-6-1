# Add the output image of the analysis portion to this variable to make conversion to pdf easier once done.
PDFS=analysis-I.pdf analysis-III.pdf analysis-IV.pdf

pdf: $(PDFS)

%.pdf: %.md
	pandoc $< -o $*.pdf

%.pdf: %.inpynb
	jupyter nbconvert $< --to pdf