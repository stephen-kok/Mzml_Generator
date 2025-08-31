# Agent Instructions

This document provides special instructions for software agents working in this repository.

## Testing Environment Setup

**IMPORTANT:** The testing environment in this repository is non-standard. The `pytest` command is executed within an isolated `pipx` virtual environment.

If you need to install or update dependencies for testing, a standard `pip install` command will **not** work, as it will target the wrong environment. This will result in `ModuleNotFoundError` when you try to run the tests.

To correctly install dependencies for the testing environment, you **must** use the following command format, which targets the python executable inside the `pipx` virtual environment for `pytest`:

```bash
/home/jules/.local/share/pipx/venvs/pytest/bin/python -m pip install -e .[test]
```

Always use this command to set up the test environment before running `pytest`. This will ensure that all dependencies are installed in the correct location and that the tests can find the required modules.
