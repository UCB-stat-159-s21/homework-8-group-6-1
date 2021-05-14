# Add the output image of the analysis portion to this variable to make conversion to pdf easier once done.
PDFS=

pdf: $(PDFS)

%.pdf: %.md
	pandoc $< -o $*.pdf

%.pdf: %.inpynb
	jupyter nbconvert $< --to pdf