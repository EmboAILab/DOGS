# Simulation Interface

This folder documents how DOGS assets are checked after conversion for Isaac Sim / Isaac Lab-style simulation scenes.

The full simulator, platform dependencies, robot assets, and generated USD/USDZ files are intentionally not vendored in this minimal release. Use the external asset package for paper-related converted assets.

## Interface Checks

For each converted object or composed scene, record:

- independent visual asset export status;
- USD/USDZ package validity;
- selected Isaac Sim / Isaac Lab visibility and placement result;
- expected restore path and scene placement path;
- declared units, world frame, and asset transform;
- whether a mesh or collision-proxy path is present;
- visual correspondence between real observation, virtual scene layout, and robot-view inspection.

The submitted evidence contains selected USD/USDZ visual-interface records. Package validity, scene hierarchy, numerical scale error, collision loading, and URDF compatibility must be reported as verified only when a corresponding inspection record exists. These checks do not validate dynamics modeling, contact stability, mass/inertia calibration, closed-loop control, or robot task performance.

## Paper-Aligned Evidence

The simulation-facing evidence should be reported as asset-interface evidence:

```text
object Gaussian subspace -> visual asset / proxy geometry -> USD/USDZ carrier -> selected Isaac Sim / Isaac Lab interface check
```

Keep screenshots, GIFs, converted USD/USDZ assets, robot assets, and simulator-generated files outside the Git repository. Store paper-related material through the external asset package described in `Assets/ASSET_MANIFEST.csv`.
