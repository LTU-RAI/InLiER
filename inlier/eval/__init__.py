"""Evaluation framework: datasets, ground truth, metrics, protocols, artifacts.

Lives inside the installed package (rather than the top-level ``evaluation/``
directory it grew up in) because ``pyproject.toml`` ships only ``inlier``; code
outside it would work from a git checkout and nowhere else.  The heavy
dependencies it needs -- ``open3d``, ``matplotlib``, ``pandas``, ``scipy``,
``tqdm`` -- stay in the ``[eval]`` extra and are imported inside the functions
that use them, so ``import inlier`` never pulls them in.
"""
