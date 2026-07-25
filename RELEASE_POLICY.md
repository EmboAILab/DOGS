# Minimal Research-Code Release Policy

Use this policy for this project and similar local research repositories.

## GitHub Contents

Upload only files required for understanding, reviewing, or running the project-specific code:

- project-specific training or evaluation entry points;
- project-specific utility scripts;
- small configs needed by those scripts;
- concise README and usage documentation;
- third-party notices and licenses for copied or adapted source code.

Do not upload:

- full upstream repositories when only a few adapted files are used;
- nested `.git` histories;
- manuscript Word/PDF files and local paper work logs;
- model weights, checkpoints, source-resolution datasets, original recordings, point clouds, USD/USDZ/FBX/GLB assets, or experiment outputs;
- file-level inventories of private media, including source filenames, experiment paths, episode identifiers, byte counts, or source hashes;
- page-ready derivative media unless it is necessary for the public project page and has been downsampled and stripped of embedded metadata;
- generated caches such as `__pycache__`, `.pytest_cache`, build directories, or logs.

## Third-Party Code

When only part of an upstream project is used:

1. Extract the actually used files into a focused folder such as `Translate/`.
2. Include only direct local dependencies.
3. Keep the upstream license for copied/adapted code.
4. Add a `THIRD_PARTY_NOTICES.md` with upstream URL and citation.
5. State in the main README which part was adapted and which upstream package must still be installed separately.

## Large Files

For large assets:

1. Keep them out of Git history.
2. Summarize paper-related asset groups in `Assets/ASSET_MANIFEST.csv`.
3. Copy required assets into a local `CloudDiskPackage/<package-name>/` directory while preserving restore paths.
4. Keep them out of Git history and publish an approved package as a separate GitHub Release asset or an external archival record.
5. Keep only stable, verified access links in the repository documentation.

## Review Before Push

Before pushing, check that no large binary, manuscript, local work log, or unrelated upstream folder is tracked.
