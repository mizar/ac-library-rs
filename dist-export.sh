#!/bin/bash
cargo +1.42.0 clippy --workspace --all-targets -- -A renamed-and-removed-lints -D warnings
cargo +stable clippy --workspace --all-targets -- -A renamed-and-removed-lints -D warnings
mkdir -p .dist
python expand.py -o .dist/convolution.rs convolution
python expand.py -o .dist/dsu.rs dsu
python expand.py -o .dist/fenwicktree.rs fenwicktree
python expand.py -o .dist/lazysegtree.rs lazysegtree
python expand.py -o .dist/math.rs math
python expand.py -o .dist/maxflow.rs maxflow
python expand.py -o .dist/mincostflow.rs mincostflow
python expand.py -o .dist/modint.rs modint
python expand.py -o .dist/scc.rs scc
python expand.py -o .dist/segtree.rs segtree
python expand.py -o .dist/string.rs string
python expand.py -o .dist/twosat.rs twosat
python expand.py -o .dist/all.rs --all
