#!/bin/bash
set -e
cd "$(dirname "$0")"
python3 src/analyze.py
python3 src/analyze_v2.py
python3 src/ml_lattice.py
python3 src/make_figures.py
(cd src && python3 make_figures_v2.py)
python3 src/symmetrize.py
(cd src && python3 make_figures_sym.py)
python3 src/advanced.py --relattice
(cd src && python3 make_figures_adv.py)
python3 src/make_figures_sic.py
python3 src/make_figures_method.py
python3 src/ablation_fixed_origin.py
python3 src/symmetry_noise_test.py
python3 src/chance_projection_cost.py
