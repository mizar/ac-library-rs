cargo +1.42.0 clippy --workspace --all-targets -- -A renamed-and-removed-lints -D warnings
cargo +stable clippy --workspace --all-targets -- -A renamed-and-removed-lints -D warnings
New-Item -Type Directory .dist -ErrorAction SilentlyContinue
python.exe expand.py -o .dist/convolution.rs convolution
python.exe expand.py -o .dist/dsu.rs dsu
python.exe expand.py -o .dist/fenwicktree.rs fenwicktree
python.exe expand.py -o .dist/lazysegtree.rs lazysegtree
python.exe expand.py -o .dist/math.rs math
python.exe expand.py -o .dist/maxflow.rs maxflow
python.exe expand.py -o .dist/mincostflow.rs mincostflow
python.exe expand.py -o .dist/modint.rs modint
python.exe expand.py -o .dist/scc.rs scc
python.exe expand.py -o .dist/segtree.rs segtree
python.exe expand.py -o .dist/string.rs string
python.exe expand.py -o .dist/twosat.rs twosat
python.exe expand.py -o .dist/all.rs --all
