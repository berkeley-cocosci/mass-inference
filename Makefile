.PHONY: figures pdf cogsci2013 clean-figs clean-pdf clean

all: figures pdf


figures:
	./makefigs.sh	

pdf: cogsci2013
cogsci2013:
	./tex2pdf.sh cogsci2013-mass-learning


clean: clean-figs clean-pdf
clean-figs:
	rm -f figures/*.png
	rm -f figures/*.pdf
clean-pdf:
	rm -f man/*.pdf
	rm -rf man/cogsci2013-mass-learning_files/

