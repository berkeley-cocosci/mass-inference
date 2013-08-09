.PHONY: figures poster slides serve clean-figs clean-poster clean-slides clean

all: figures poster slides

figures:
	./makefigs.sh

poster:
	./makeposter.sh man/cogsci2013-poster.graffle

slides:
	./makeslides.sh mathpsych2013-slides
serve:
	./serveslides.sh mathpsych2013-slides

clean: clean-figs clean-poster clean-slides
clean-figs:
	rm -f figures/*.png
	rm -f figures/*.pdf
clean-poster:
	rm -f man/cogsci2013-poster.pdf
clean-slides:
	rm man/mathpsych2013-slides/mathpsych2013-slides.slides.html
