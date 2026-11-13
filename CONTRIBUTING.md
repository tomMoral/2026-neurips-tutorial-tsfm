# Contributing to the tutorial notebooks

These conventions keep the notebooks of the tutorial consistent. The notebooks
sit at the root of the repository.
Most of them are checked by `tools/check_notebooks.py`, which runs as a
pre-commit hook and in CI.

## Setup

Create the environment of `environment.yml`, then install the hooks once:

```bash
pre-commit install
```

On each commit, the hooks strip the outputs of the notebooks and check the
conventions below. Run them on the whole repository with
`pre-commit run --all-files`.

## Notebook layout

- Name a notebook `NN-topic.ipynb`, with an optional letter after the number
  (`01b-grow_a_forest.ipynb`).
- The first cell is markdown. Its first line is `# Title`, followed by a blank
  line and the Colab badge of this notebook:

  ```markdown
  # Title

  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tomMoral/2026-neurips-tutorial-tsfm/blob/main/<notebook>.ipynb)
  ```

- Below the badge, give the authors and an outline of the notebook. Long
  notebooks have a table of contents.
- Any other Colab badge in the notebook is the same badge.

## Running on Colab

Every notebook runs on Colab as well as in a local clone.

- The first code cell is the setup cell below.
  On Colab it clones the repository and moves to its root. Elsewhere it
  does nothing.

  ```python
  # Colab: clone the tutorial repository and run from its root
  import os, subprocess, sys

  if "google.colab" in sys.modules:
      repo = "/content/2026-neurips-tutorial-tsfm"
      if not os.path.exists(repo):
          subprocess.run(["git", "clone", "--depth", "1", "https://github.com/tomMoral/2026-neurips-tutorial-tsfm", repo], check=True)
      os.chdir(repo)
  ```

- Read data with a path relative to the repository root, such as
  `pd.read_parquet("data/paris-weather.parquet")`. The data must be tracked in git.
  `.gitignore` ignores CSV and archive files, so add an exception for a
  data file the tutorial ships.
- Local modules at the root, such as `tsfm_utils.py`, import as usual,
  because the setup cell runs from the root.
- Installs are commented out: `# !pip install -U <package>`.
  Students uncomment it on Colab when a package is missing there.

## Exercises and questions

An exercise asks for code. A question asks for a discussion. Both go in a box:

```html
<div class="alert alert-success">
    <b>EXERCISE</b>:
    <ul>
    <li>What to do.</li>
    </ul>
</div>

Solution: `solutions/cv-1_target-distribution.py`
```

- Use `<b>EXERCISE</b>:` or `<b>QUESTION</b>:`, in a
  `<div class="alert alert-success">`. A short title can follow the colon.
- An exercise box is followed by one code cell, the exercise cell. Leave it
  empty, or give starter code with `...` for the parts to fill in.
- If the exercise has a solution, the box ends with one line
  ``Solution: `solutions/<file>.py` ``. Do not load it with `%load`: students
  open the file themselves.
- Name the solution `solutions/<topic>-<n>_<what>.py`: a short topic for the
  notebook, the exercise number, and what it does
  (`bike-2_polynomial-features.py`). Every solution file is named by one
  exercise.
- A solution replaces the whole exercise cell, so it must run on its own
  after the cells above it. It also defines every name the starter code
  defines, because later cells may use them.

## Outputs and dependencies

- Notebooks are committed without outputs. The `strip-outputs` hook does it
  on commit. To strip by hand, run `python tools/strip_outputs.py` (all
  notebooks) or `python tools/strip_outputs.py <notebook.ipynb>`. It clears
  the outputs and execution counts and changes nothing else.
- Every package a notebook imports is listed in `environment.yml`.
- A cell that fails on purpose carries the `raises-exception` cell tag.
  Say in the text that it is expected to fail.
- A notebook that needs a GPU sets `"accelerator": "GPU"` in its notebook
  metadata. Colab then opens it on a GPU runtime, and CI skips it, because
  its runners have no GPU.
- Fix the random seeds (`random_state`, `np.random.default_rng(0)`) so that the
  numbers quoted in the text match a fresh run.

## Continuous integration

The `Notebooks` workflow runs the hooks, then executes the notebooks with
each exercise cell replaced by its solution. A pull request runs the
notebooks it touches. A change to any other file, except in `notes/`, runs
them all. Each notebook is one job, and the run summary lists the errors.

Run every notebook by hand:

```bash
gh workflow run notebooks.yml -f notebooks=all
gh run list -w notebooks.yml
```

To execute a notebook locally as CI does:

```bash
python tools/run_notebook.py <notebook.ipynb> executed/
```
