"""Compile the standalone editable pipeline diagram; no manuscript required."""
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='qa-pipeline-') as tmp:
    work=Path(tmp)
    shutil.copy2(ROOT/'figures/pipeline.tex',work/'pipeline.tex')
    (work/'diagram.tex').write_text(r'''\documentclass[border=2pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,backgrounds}
\usepackage{fontawesome5}
\usepackage{mathptmx}
\begin{document}
\input{pipeline.tex}
\end{document}
''')
    result=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','diagram.tex'],
                          cwd=work,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    if result.returncode:
        print(result.stdout[-6000:])
        raise SystemExit('Pipeline compilation failed.')
    shutil.copy2(work/'diagram.pdf',ROOT/'figures/pipeline.pdf')
print('Generated figures/pipeline.pdf from editable TikZ source.')
