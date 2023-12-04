from md2pdf.core import md2pdf
md2pdf('results_stat.pdf', md_content=open('results_stat.md').read())
md2pdf('results.pdf', md_content=open('results.md').read())
