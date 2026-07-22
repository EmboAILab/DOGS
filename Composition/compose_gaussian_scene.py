"""Compose aligned DOGS Gaussian PLY files by direct parameter union."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re


SCALAR_BYTES = {
    "char": 1,
    "int8": 1,
    "uchar": 1,
    "uint8": 1,
    "short": 2,
    "int16": 2,
    "ushort": 2,
    "uint16": 2,
    "int": 4,
    "int32": 4,
    "uint": 4,
    "uint32": 4,
    "float": 4,
    "float32": 4,
    "double": 8,
    "float64": 8,
}


@dataclass(frozen=True)
class GaussianPly:
    path: Path
    format_name: str
    line_ending: bytes
    header_lines: tuple[str, ...]
    vertex_count: int
    properties: tuple[tuple[str, str], ...]
    body: bytes


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_header(data: bytes, path: Path) -> tuple[bytes, bytes, bytes]:
    for marker in (b"end_header\r\n", b"end_header\n"):
        index = data.find(marker)
        if index >= 0:
            end = index + len(marker)
            return data[:end], data[end:], b"\r\n" if marker.endswith(b"\r\n") else b"\n"
    raise ValueError(f"{path} does not contain a complete PLY header")


def read_gaussian_ply(path: Path) -> GaussianPly:
    data = path.read_bytes()
    header, body, line_ending = _split_header(data, path)
    lines = tuple(header.decode("ascii").splitlines())
    if not lines or lines[0].strip() != "ply":
        raise ValueError(f"{path} is not a PLY file")

    format_name = ""
    vertex_count = None
    properties = []
    current_element = None
    other_elements = []
    for line in lines:
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "format" and len(fields) >= 2:
            format_name = fields[1]
        elif fields[0] == "element" and len(fields) == 3:
            current_element = fields[1]
            count = int(fields[2])
            if current_element == "vertex":
                vertex_count = count
            else:
                other_elements.append((current_element, count))
        elif fields[0] == "property" and current_element == "vertex":
            if len(fields) != 3 or fields[1] == "list":
                raise ValueError(f"{path} contains an unsupported vertex property: {line}")
            if fields[1] not in SCALAR_BYTES:
                raise ValueError(f"{path} uses an unsupported scalar type: {fields[1]}")
            properties.append((fields[1], fields[2]))

    if format_name not in {"ascii", "binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"{path} uses an unsupported PLY format: {format_name}")
    if vertex_count is None or not properties:
        raise ValueError(f"{path} lacks a vertex element or scalar vertex properties")
    if other_elements:
        raise ValueError(f"{path} contains non-vertex elements: {other_elements}")

    if format_name == "ascii":
        rows = [row for row in body.splitlines() if row.strip()]
        if len(rows) != vertex_count:
            raise ValueError(f"{path} declares {vertex_count} vertices but contains {len(rows)} rows")
        normalized_body = line_ending.join(rows) + (line_ending if rows else b"")
    else:
        record_size = sum(SCALAR_BYTES[type_name] for type_name, _ in properties)
        expected = vertex_count * record_size
        if len(body) != expected:
            raise ValueError(f"{path} declares {vertex_count} vertices ({expected} bytes) but contains {len(body)} bytes")
        normalized_body = body

    return GaussianPly(
        path=path,
        format_name=format_name,
        line_ending=line_ending,
        header_lines=lines,
        vertex_count=vertex_count,
        properties=tuple(properties),
        body=normalized_body,
    )


def compose(inputs: list[Path], output: Path, manifest_path: Path | None = None) -> dict:
    if len(inputs) < 2:
        raise ValueError("composition requires one background and at least one object PLY")
    payloads = [read_gaussian_ply(path) for path in inputs]
    reference = payloads[0]
    for payload in payloads[1:]:
        if payload.format_name != reference.format_name:
            raise ValueError(f"PLY format mismatch: {payload.path}")
        if payload.properties != reference.properties:
            raise ValueError(f"ordered vertex-property schema mismatch: {payload.path}")

    total_vertices = sum(payload.vertex_count for payload in payloads)
    header_lines = list(reference.header_lines)
    replaced = False
    for index, line in enumerate(header_lines):
        if re.fullmatch(r"element\s+vertex\s+\d+", line.strip()):
            header_lines[index] = f"element vertex {total_vertices}"
            replaced = True
            break
    if not replaced:
        raise ValueError("could not update the output vertex count")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    header = reference.line_ending.join(line.encode("ascii") for line in header_lines) + reference.line_ending
    output.write_bytes(header + b"".join(payload.body for payload in payloads))

    manifest = {
        "operation": "parameter_level_union",
        "coordinate_transform": "none",
        "format": reference.format_name,
        "properties": [name for _, name in reference.properties],
        "inputs": [
            {
                "role": "background" if index == 0 else "object",
                "file": payload.path.name,
                "sha256": sha256(payload.path),
                "gaussian_count": payload.vertex_count,
            }
            for index, payload in enumerate(payloads)
        ],
        "output": {
            "file": output.name,
            "sha256": sha256(output),
            "gaussian_count": total_vertices,
        },
    }
    manifest_path = manifest_path or output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--object", dest="objects", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compose([args.background, *args.objects], args.output, args.manifest)
    print(json.dumps(result["output"], indent=2))


if __name__ == "__main__":
    main()
