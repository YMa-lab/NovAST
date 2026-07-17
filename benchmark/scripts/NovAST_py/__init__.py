# No NovAST algorithm code is vendored here -- NovAST is an installed package
# (see README.md). This folder holds only the benchmark adapter
# (run_benchmark.py) and the installation tutorial.
#
# Kept as a package so main.py can do `from NovAST_py.run_benchmark import
# NovAST_main`. Deliberately NOT named `NovAST`, which would shadow the
# installed package that run_benchmark.py imports.
