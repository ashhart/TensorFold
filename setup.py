"""TensorFold 1.0.0 is a native binary, so a pip install stops here and says where to get it."""
raise SystemExit(
    "TensorFold 1.0.0 and later install as a native binary: brew install ashhart/tensorfold/tensorfold on a Mac, "
    "or the archive for your platform from https://github.com/ashhart/TensorFold/releases. The Python engine stays "
    "at 0.6.6: python -m pip install git+https://github.com/ashhart/TensorFold.git@v0.6.6"
)
