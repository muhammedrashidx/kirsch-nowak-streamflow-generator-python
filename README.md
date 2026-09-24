
# Kirsch–Nowak Streamflow Generator — Python

A Python implementation of the stationary Kirsch–Nowak synthetic streamflow generator originally developed in MATLAB by **Matteo Giuliani, Jon Herman, and Julianne Quinn**.

> **AI-assisted development:** The initial MATLAB-to-Python code conversion was performed with assistance from **Google Gemini 3.1 Pro**. The resulting Python implementation was subsequently reviewed, debugged, modified, and validated against the original MATLAB implementation, including reproducibility of the random-number sequence and final generated streamflows.

Original repository:
https://github.com/julianneq/Kirsch-Nowak_Streamflow_Generator

## What this repository provides

* Python implementation of the original MATLAB generation procedure.
* Reproducible random-number generation using **MT19937**, with the Python implementation matched to the MATLAB random-number sequence.
* Support for multiple synthetic realizations and years in a Python environment.

For reproducibility, an explicit random seed (`12345` in the validation tests) was added to the MATLAB reference workflow and reproduced in Python.

## Methodology

The implementation follows the original Kirsch et al. (2013) monthly generation and Nowak et al. (2010) daily disaggregation approach.

## Attribution

This Python implementation is based on the work of:

**Matteo Giuliani, Jon Herman, and Julianne Quinn**

Please cite the original work and repository when using this implementation.

This repository does not claim authorship of the original Kirsch–Nowak methodology.
