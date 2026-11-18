# Kerr QNM Python solver

This folder contains the solver and one notebook comparing its prolate and oblate Kerr modes with high-damping predictions. The supplied results reach `a/M = 0.7`.

Open [Kerr_QNM.ipynb](Kerr_QNM.ipynb) in VS Code with Python 3.12. Install the dependencies with `python -m pip install -r requirements.txt`, then **Run All**. This displays the saved comparisons; to calculate new modes, set `COMPUTE = True` in the final cell.

`solver/` contains the calculation, `data/` the Schwarzschild starting frequencies, and `results/` the saved Kerr modes. Units and methods are explained in the notebook.

<!--
We used Claude (Anthropic) and OpenAI tools to help check calculations, search and verify the literature, and edit the text. The authors reviewed and verified all derivations and results, and take full responsibility for this work.
-->
