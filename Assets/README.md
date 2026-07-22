# Large Asset Package

Source-resolution captures, original recordings, model outputs, and generated simulation assets are not stored in GitHub. They should be distributed through an external package and restored locally when needed.

Recommended package name:

```text
DOGS-paper-assets
```

No cloud-drive URL is committed in this source release. Keep access links in the project release notes or repository release page.

The external package should preserve the category-level restore paths listed in `ASSET_MANIFEST.csv`. The public manifest intentionally omits source filenames, experiment-directory names, episode identifiers, byte counts, and source-media hashes.

`FIGURE_SOURCE_MANIFEST.csv` records only figure-level source classes and provenance. It does not inventory the underlying recordings or raw captures. The repository contains a small set of downsampled, metadata-stripped page derivatives under `static/`; these files are public presentation media rather than source-resolution experimental data.

Do not upload manuscript files, full upstream repositories, local experiment directories, source-resolution captures or recordings, model weights, point clouds, USD/USDZ/FBX/GLB files, or experiment outputs to GitHub.
