# Simulation Interface

This folder documents how DOGS assets can be inspected after conversion. It is documentation only and does not provide simulator-side executable code.

The full simulator, platform dependencies, robot assets, and generated USD/USDZ files are intentionally not vendored in this minimal release. Use the external asset package for paper-related converted assets.

## Interface Checks

For each converted object or composed scene, record:

- independent visual asset export status;
- USD/USDZ package validity;
- selected Isaac Sim visibility and placement result when a corresponding record is available;
- expected restore path and scene placement path;
- declared units, world frame, and asset transform;
- whether a mesh or collision-proxy path is present;
- visual correspondence between real observation, virtual scene layout, and robot-view inspection.

The paper directly displays selected USDZ visual-import records in Isaac Sim, including visible placement and robot-scene coexistence. It does not use those images to claim URDF or Isaac Lab compatibility, package hierarchy validation, numerical scale error, or collision-proxy loading. The records also do not validate dynamics modeling, contact stability, mass/inertia calibration, closed-loop control, or robot task performance.

## Paper-Aligned Evidence

The simulation-facing evidence should be reported as asset-interface evidence:

```text
object Gaussian subspace -> visual asset / proxy geometry -> USDZ carrier -> selected Isaac Sim visual-import record
```

Keep screenshots, GIFs, converted USD/USDZ assets, robot assets, and simulator-generated files outside the Git repository. Store paper-related material through the external asset package described in `Assets/ASSET_MANIFEST.csv`.
