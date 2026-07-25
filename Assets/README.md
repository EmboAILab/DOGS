# Asset Policy

Source-resolution captures, original recordings, model outputs, and generated simulation assets are not committed to this Git repository. A selected, metadata-sanitized self-collected RGB subset is published separately in the verified [data-v1.0.0 GitHub Release](https://github.com/EmboAILab/DOGS/releases/tag/data-v1.0.0). Its scope is defined in `DATA_AVAILABILITY.md`.

`ASSET_MANIFEST.csv` documents category-level local paths used by the project workflow. It is a placement-policy reference rather than an availability or download index, and intentionally omits source filenames, experiment-directory names, episode identifiers, byte counts, and source-media hashes.

`FIGURE_SOURCE_MANIFEST.csv` records only figure-level source classes and provenance. It does not inventory the underlying recordings or raw captures. The repository contains a small set of downsampled, metadata-stripped page derivatives under `static/`; these files are public presentation media rather than source-resolution experimental data.

Do not commit manuscript files, full upstream repositories, local experiment directories, source-resolution captures or recordings, model weights, point clouds, USD/USDZ/FBX/GLB files, or experiment outputs to Git history. Any approved large research-data package must remain a separate release asset with its own scope and integrity manifest.
