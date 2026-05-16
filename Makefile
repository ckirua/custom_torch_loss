# Project-local Python: .venv in this directory (isolated from system Python).
# Run `make develop` once; then `make test` / `make benchmark` use .venv automatically.
# Override: make PYTHON_FOR_VENV=python3 develop

PYTHON_FOR_VENV ?= python3.14
VENV := .venv
PY := $(VENV)/bin/python
PY_ABS := $(CURDIR)/$(VENV)/bin/python
# Always use `python -m pip`: a copied/moved .venv often leaves `pip` with a wrong shebang
# (packages go to another tree while `$(PY)` uses this repo — breaks CMake/torch).
#
# Default PyTorch wheel is CPU-only (PyTorch CPU index). The usual pip/PyPI Linux wheel is CUDA-linked;
# CMake then requires a full local CUDA toolkit (nvcc, etc.) or find_package(Torch) fails with
# "Your installed Caffe2 version uses CUDA". Override: make deps TORCH_INDEX_URL=
TORCH_INDEX_URL ?= https://download.pytorch.org/whl/cpu
TORCH_PIP_OPTS := $(if $(strip $(TORCH_INDEX_URL)),--index-url $(TORCH_INDEX_URL),)

.PHONY: venv deps develop install wheel clean test benchmark configure build-cpp uninstall full help

help:
	@echo "custom_loss — targets"
	@echo "  venv      Create .venv ($(PYTHON_FOR_VENV) -m venv) if missing"
	@echo "  deps      Install torch (CPU wheel by default), pybind11, setuptools, wheel"
	@echo "            Override: make deps TORCH_INDEX_URL=  (empty = PyPI; often CUDA on Linux)"
	@echo "  develop   deps + editable install of this package (no build isolation)"
	@echo "  test / benchmark — run scripts with .venv Python"
	@echo "  configure / build-cpp — CMake demo test_losses (needs deps; CMake finds torch via .venv)"
	@echo "  (Python module custom_loss_cpp: pip/setup.py via develop|install|wheel — not CMake.)"

venv:
	@test -d $(VENV) || $(PYTHON_FOR_VENV) -m venv $(VENV)

deps: venv
	$(PY) -m pip install -U pip "setuptools<82" wheel
	$(PY) -m pip install $(TORCH_PIP_OPTS) torch
	$(PY) -m pip install pybind11

develop: deps
	$(PY) -m pip install -e . --no-build-isolation

install: deps
	$(PY) -m pip install . --no-build-isolation

wheel: deps
	$(PY) setup.py bdist_wheel

clean:
	rm -rf build dist *.egg-info .eggs
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f \( -name '*.so' -o -name '*.pyd' \) -delete 2>/dev/null || true

test: develop
	$(PY) scripts/smoke_test.py

benchmark: develop
	$(PY) scripts/benchmark.py

# C++ demo: CMake auto-uses $(CURDIR)/.venv/bin/python when TORCH_USE_PIP_PYTHON=ON (default).
configure: deps
	mkdir -p build
	cd build && cmake .. \
		-DCMAKE_BUILD_TYPE=Release \
		-DTORCH_PYTHON="$(PY_ABS)"

build-cpp: configure
	cmake --build build --config Release

uninstall:
	$(PY) -m pip uninstall -y custom_loss_cpp 2>/dev/null || true

full: develop test
