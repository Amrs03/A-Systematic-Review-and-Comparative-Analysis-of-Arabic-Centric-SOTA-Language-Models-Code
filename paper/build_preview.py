"""Build preview_revised_sections.pdf from revised_sections.tex (XeLaTeX; each block under a placement banner)."""
import re, subprocess
s = open('revised_sections.tex', encoding='utf-8').read()
body = s.split('% =====================================================================================', 2)[2]
blocks = re.split(r'%% -{50,}\n', body)
out, i = [], 0
esc = lambda t: (t.replace('\\', '\x00').replace('_', '\\_').replace('&', '\\&').replace('#', '\\#').replace('$', '\\$')
                 .replace('%', '\\%').replace('{', '\\{').replace('}', '\\}').replace('\x00', '\\textbackslash{}'))
while i < len(blocks):
    b = blocks[i]
    if b.strip().startswith('%%'):
        hdr = ' '.join(l.lstrip('% ').strip() for l in b.strip().splitlines())
        content = blocks[i + 1] if i + 1 < len(blocks) else ''
        out.append('\\medskip\\noindent\\colorbox{black!8}{\\parbox{\\dimexpr\\linewidth-2\\fboxsep}{\\small\\sffamily '
                   + esc(hdr) + '}}\\par\\smallskip')
        if content.lstrip().startswith('\\item'):
            content = '\\begin{enumerate}\\setcounter{enumi}{1}\n' + content + '\\end{enumerate}\n'
        out.append(content)
        i += 2
    else:
        i += 1
doc = r'''\documentclass[10pt,twocolumn]{article}
\usepackage[margin=0.7in]{geometry}
\usepackage{graphicx,xcolor,amsmath,fontspec}
\setmainfont{DejaVu Serif}\setsansfont{DejaVu Sans}
\newfontfamily\arabicfont[Script=Arabic]{FreeSerif}
\TeXXeTstate=1
\newcommand{\ar}[1]{{\arabicfont\beginR #1\endR}}
\renewcommand{\cite}[1]{[\,#1\,]}
\title{Revised sections --- Systematic\_revised\_1\\\large drop-in replacements (preview build)}
\date{}
\begin{document}\maketitle
''' + '\n'.join(out) + '\n\\end{document}\n'
open('preview_revised_sections.tex', 'w', encoding='utf-8').write(doc)
for _ in range(2):
    subprocess.run(['xelatex', '-interaction=nonstopmode', 'preview_revised_sections.tex'], capture_output=True)
