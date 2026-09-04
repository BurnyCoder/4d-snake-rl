"""snake4d - an N-dimensional (default 4D) snake game and a MaskablePPO agent that fills it.

Global context: the package is organised as one batched numpy physics core (``physics.py``) with two
adapters (``env.py`` for a single Gymnasium env, ``vec_env.py`` for batched training), scripted
Hamiltonian baselines (``hamilton.py``, ``agents.py``), the training/evaluation/report phases, and
``main.py`` as the only orchestrator.  See README.md for the data flow.
"""

__version__ = "0.1.0"
