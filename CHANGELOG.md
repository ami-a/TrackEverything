# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.1] - 2026-09-22

### Fixed

- **`StatMethods` had no members again on Python 3.13.** 2.0.0 wrapped the
  functions in `functools.partial` so that the `Enum` would treat them as
  members rather than methods. Python 3.13 made `functools.partial` a method
  descriptor too, which reinstated the exact bug it was working around:
  `list(StatMethods)` was empty, `StatMethods["CMA"]` raised `KeyError`, and
  importing the module emitted a `FutureWarning`. The functions are now held in
  a small non-descriptor wrapper, which is a member on every supported version.
  `enum.member()` would also work but only exists from 3.11.

### Changed

- CI actions moved to `actions/checkout@v7`, `actions/setup-python@v7` and
  `actions/upload-artifact@v7`, clearing the Node 20 deprecation warnings.

## [2.0.0] - 2026-09-21

A compatibility release. Version 1.7.2 could not be imported at all on Python
3.11 or newer, and its drawing and tracker selection were broken on current
Pillow and OpenCV. Everything below was verified by reproducing the failure
first and pinning it with a regression test.

### Fixed

- **Package could not be imported on Python 3.11+.** `InspectorVars` used
  `Ids(0)` as a dataclass field default. `Ids` was itself a dataclass, so its
  `__hash__` was `None`, and Python 3.11 rejects unhashable defaults:
  `ValueError: mutable default <class 'Ids'> for field trck_id_generator is not
  allowed: use default_factory`. `Ids` is now a plain class and the defaults are
  factories.
- **Drawing raised `AttributeError` on Pillow 10+.** `ImageFont.getsize` was
  removed in Pillow 10, so every `draw_visualization` call failed. Text is now
  measured through a capability-detecting helper that prefers
  `ImageDraw.multiline_textbbox`.
- **Trackers were never usable on OpenCV 4.5.1+.** BOOSTING, TLD, MedianFlow and
  MOSSE moved to the `cv2.legacy` namespace. Lookup now searches both
  namespaces.
- **Nothing was ever tracked on modern OpenCV.** `tracker.init()` returns `None`
  rather than `True` from OpenCV 4.5.1, so `if not _ok` marked every tracker as
  failed the instant it was created, and it was discarded on the same frame.
- **Bounding boxes from the default detection parser were wrong.**
  `get_detection_array` computed width as `xmax - confidence` and height as
  `ymax - xmin` — indices 0 and 1 where it needed 1 and 2 — producing grossly
  oversized boxes for anyone using the built-in parser.
- **Detection without a classification model raised `TypeError`.** The score
  matrix was 1-D, so each `class_score` was a scalar float and `np.argmax`
  followed by indexing failed. It is now shaped `(N, 1)`.
- **Any model that was not binary raised a broadcast `ValueError`.**
  `StatisticalCalculator` hard-coded two classes; the class count now adapts to
  the first real score.
- **State leaked between instances.** `InspectorVars` shared a single `Ids`
  counter and a single `StatisticalCalculator` across every instance, so two
  `Detector`s shared tracker ids and accumulated statistics. `Detector`'s own
  signature defaults were shared objects, `VisualizationVars.colors` handed out
  the module-level palette itself, and `StatParams` was shared as a default
  argument.
- **`StatisticalCalculator.__copy__` discarded what it copied.** It called
  `__init__`, which re-ran `initialize()` and zeroed the statistics.
- **`get_tracker` returned `None` for unrecognised names**, including correct
  names in the wrong case such as `"csrt"`. The failure surfaced much later as
  `TypeError: 'NoneType' object is not callable` inside tracker construction.
  Lookup is now case-insensitive and raises immediately.
- **`box_iou` raised `ZeroDivisionError`** for degenerate zero-area boxes; it
  now returns `0.0`.
- **Bounding boxes are coerced to `int`** before reaching OpenCV, which rejects
  the floats detection models routinely emit with an opaque overload error.
- Two annotations were meaningless: `type(get_detection_array)` evaluates to
  `builtins.function`, and `np.array` is a function rather than a type.

### Changed

- **`trck_type` is now a string**, resolved and validated when `InspectorVars`
  is constructed rather than a function resolved at import time. Importing the
  package no longer touches a `cv2` tracker attribute, so using `box_iou`, the
  NMS helper or the smoothers no longer requires a contrib OpenCV build.
  Callables are still accepted, so `trck_type=get_tracker("kcf")` is unchanged.
- **`class_effect` is now applied.** It was documented, advertised in the README
  and never read by any scoring function. Leaving it unset preserves previous
  behaviour.
- **`StatMethods` is a real enum.** Bare functions in an `Enum` body become
  methods rather than members, so `list(StatMethods)` was empty and the type
  could not be iterated or looked up by name. Members are wrapped in
  `functools.partial` and remain callable.
- **Label placement is corrected.** Rendering was already multi-line while
  measurement was single-line, so labels were positioned using roughly a third
  of their true height.
- **Minimum supported Python is 3.9**; 3.7 and 3.8 are end-of-life.
- Dependency floors raised to where the APIs in use exist: `numpy>=1.20`,
  `scipy>=1.6`, `Pillow>=8.0`, `opencv-contrib-python>=4.5.1`.
- `DetectionVars.load_model` raises `ValueError` rather than bare `Exception`.

### Added

- **A public API.** `__init__.py` was empty; 32 names are now re-exported, so
  `from TrackEverything import Detector, DetectionVars` works. Every 1.7.x
  import path still resolves unchanged.
- `__version__`, and a `py.typed` marker backed by return annotations
  throughout.
- **A test suite** — 135 tests at 94% branch coverage, running in about a
  second with no TensorFlow, GPU, video file or network.
- **Continuous integration** across Python 3.9–3.13 on Linux, plus Windows and
  macOS, with linting and a packaging check.
- `available_trackers()`, reporting the trackers the installed OpenCV provides.
- `pyproject.toml`. The previous `setup.py` was excluded by `.gitignore`, so the
  published repository contained no build metadata and could not be installed
  from a clone.

## [1.7.2] - 2020-10-25

Last release of the original series.

[2.0.1]: https://github.com/ami-a/TrackEverything/releases/tag/v2.0.1
[2.0.0]: https://github.com/ami-a/TrackEverything/releases/tag/v2.0.0
[1.7.2]: https://github.com/ami-a/TrackEverything/releases/tag/v1.7.2
