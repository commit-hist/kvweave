"""Check an installed wheel against project metadata and execute public retrieval."""

import argparse
from importlib.metadata import metadata, requires
from pathlib import Path
import tomllib

from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def _requirement_key(value: str) -> tuple[str, tuple[str, ...], str, str | None, str]:
    requirement = Requirement(value)
    return (
        canonicalize_name(requirement.name),
        tuple(sorted(canonicalize_name(extra) for extra in requirement.extras)),
        str(requirement.specifier),
        requirement.url,
        str(requirement.marker) if requirement.marker else "",
    )


def _extra_requirement(value: str, extra: str) -> str:
    requirement = Requirement(value)
    extra_marker = Marker(f"extra == {extra!r}")
    requirement.marker = (
        Marker(f"({requirement.marker}) and {extra_marker}")
        if requirement.marker is not None
        else extra_marker
    )
    return str(requirement)


def _check_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise AssertionError(f"wheel {label}: actual={actual!r}; expected={expected!r}")


def check_metadata(project_path: Path) -> None:
    project = tomllib.loads(project_path.read_text())["project"]
    installed = metadata("kvweave")
    expected_fields = {
        "Name": project["name"],
        "Version": project["version"],
        "Summary": project["description"],
        "Requires-Python": project["requires-python"],
        "Author": ", ".join(author["name"] for author in project["authors"]),
        "Description-Content-Type": "text/markdown",
    }
    for field, expected in expected_fields.items():
        _check_equal(field, installed[field], expected)
    _check_equal(
        "License",
        installed["License-Expression"] or installed["License"],
        project["license"],
    )
    _check_equal(
        "Classifier",
        set(installed.get_all("Classifier", [])),
        set(project["classifiers"]),
    )
    _check_equal(
        "Project-URL",
        set(installed.get_all("Project-URL", [])),
        {f"{name}, {url}" for name, url in project["urls"].items()},
    )
    _check_equal(
        "Provides-Extra",
        set(installed.get_all("Provides-Extra", [])),
        set(project["optional-dependencies"]),
    )
    _check_equal(
        "Keywords",
        set((installed["Keywords"] or "").replace(",", " ").split()),
        set(project["keywords"]),
    )
    expected_requirements = [
        *project["dependencies"],
        *(
            _extra_requirement(requirement, extra)
            for extra, requirements in project["optional-dependencies"].items()
            for requirement in requirements
        ),
    ]
    _check_equal(
        "Requires-Dist",
        {_requirement_key(value) for value in requires("kvweave") or []},
        {_requirement_key(value) for value in expected_requirements},
    )


def check_retrieval() -> None:
    import torch

    from kvweave import BruteForceIndex, KVCache, PQIndex, QuestIndex, TensorStorage
    from kvweave.metrics.reference import full_attention, selected_attention

    generator = torch.Generator().manual_seed(0)
    keys = torch.randn(1, 2, 16, 4, generator=generator)
    values = torch.randn(1, 2, 16, 4, generator=generator)
    query = torch.randn(1, 2, 4, generator=generator)
    for index in (
        BruteForceIndex(),
        QuestIndex(page_size=4),
        PQIndex(num_subspaces=2, num_centroids=2, max_iterations=2, seed=0),
    ):
        cache = KVCache(index=index, storage=TensorStorage())
        cache.build(keys, values)
        partial = cache.retrieve(query, budget=8)
        _check_equal(
            f"{type(index).__name__} partial retrieval shape",
            tuple(partial.keys.shape),
            (1, 2, 8, 4),
        )
        retrieved = cache.retrieve(query, budget=16)
        torch.testing.assert_close(
            selected_attention(
                query, retrieved.keys, retrieved.values, retrieved.valid_mask
            ),
            full_attention(query, keys, values),
            rtol=1e-4,
            atol=1e-5,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    check_metadata(args.project)
    check_retrieval()
    print("Installed wheel metadata and BruteForce/Quest/PQ retrieval checks passed.")


if __name__ == "__main__":
    main()
