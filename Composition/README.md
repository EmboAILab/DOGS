# Parameter-Level Scene Composition

`compose_gaussian_scene.py` implements the direct parameter union in the paper.
It concatenates a background Gaussian PLY and one or more extracted object
Gaussian PLY files that already share the same world coordinate system. No
translation, rotation, or scale adjustment is applied.

```bash
python Composition/compose_gaussian_scene.py \
  --background path/to/background.ply \
  --object path/to/object_1.ply \
  --object path/to/object_2.ply \
  --output path/to/composed_scene.ply
```

The inputs must use the same PLY format and ordered scalar vertex-property
schema. The script rejects non-vertex elements, list properties, schema
mismatches, and malformed record counts. It writes a JSON manifest beside the
output with input hashes and Gaussian counts.
