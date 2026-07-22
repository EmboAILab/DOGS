from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from Composition.compose_gaussian_scene import compose, read_gaussian_ply


def write_ascii_ply(path: Path, rows: list[str]) -> None:
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                f"element vertex {len(rows)}",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                *rows,
                "",
            ]
        ),
        encoding="ascii",
    )


class CompositionTests(unittest.TestCase):
    def test_direct_union_preserves_records_and_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            background = root / "background.ply"
            object_one = root / "object_1.ply"
            output = root / "scene.ply"
            write_ascii_ply(background, ["0 0 0", "1 0 0"])
            write_ascii_ply(object_one, ["0 1 0"])

            manifest = compose([background, object_one], output)
            payload = read_gaussian_ply(output)

            self.assertEqual(payload.vertex_count, 3)
            self.assertEqual(payload.body.splitlines(), [b"0 0 0", b"1 0 0", b"0 1 0"])
            self.assertEqual(manifest["coordinate_transform"], "none")
            self.assertEqual(manifest["output"]["gaussian_count"], 3)


if __name__ == "__main__":
    unittest.main()
