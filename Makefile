.PHONY: figures clean-figs clean

all: figures pdf


figures:
	./makefigs.sh

clean: clean-figs
clean-figs:
	rm -f figures/*.png
	rm -f figures/*.pdf

