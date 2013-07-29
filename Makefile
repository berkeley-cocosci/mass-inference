.PHONY: figures poster clean-figs clean-poster clean

all: figures poster

figures:
	./makefigs.sh

poster:
	./makeposter.sh man/cogsci2013-poster.graffle

clean: clean-figs clean-poster
clean-figs:
	rm -f figures/*.png
	rm -f figures/*.pdf
clean-poster:
	rm -f man/cogsci2013-poster.pdf

