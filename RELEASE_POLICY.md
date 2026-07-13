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
- model weights, checkpoints, datasets, rendered images, videos, point clouds, USD/USDZ/FBX/GLB assets, or experiment outputs;
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

1. Keep them out of GitHub.
2. Summarize paper-related asset groups in `Assets/ASSET_MANIFEST.csv`.
3. Copy required assets into a local `CloudDiskPackage/<package-name>/` directory while preserving restore paths.
4. Publish the package separately from GitHub.
5. Keep access links in release notes or the repository release page, not in source files unless the link is stable.

## Review Before Push

Before pushing, check that no large binary, manuscript, local work log, or unrelated upstream folder is tracked.
