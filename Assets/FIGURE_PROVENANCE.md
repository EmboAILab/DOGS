# Manuscript Figure Provenance

The submitted DOGS figures use two source classes:

1. Figures 1-3 are manually authored method diagrams exported as vector PDFs from the project PowerPoint sources.
2. Figures 4-7 are composites of real photographs, reconstructed-scene outputs, or frames selected from recorded experimental and simulation videos.

`FIGURE_SOURCE_MANIFEST.csv` maps each figure or panel to its source path and SHA-256 checksum. `PAPER_MEDIA_MANIFEST.csv` records the available still and video pool with byte counts and checksums. Large source media are kept outside GitHub under the restore policy in `ASSET_MANIFEST.csv`.

The source audit compared embedded media from the PowerPoint drawing archive against the separate historical image-generation directory and found no hash match in the submitted figure set. No generative-AI-created or generative-AI-modified image is used as a submitted manuscript figure.

The figure evidence is bounded as follows:

- visual records support object-level separation, object-removed background availability, object repositioning, real-to-virtual correspondence, and selected USD/USDZ placement checks;
- the records do not establish population-level reconstruction superiority, dynamics accuracy, collision fidelity, or closed-loop task performance;
- a displayed video frame is evidence from its source recording, not an independent trial.
