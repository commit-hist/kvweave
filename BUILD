python_requirements(
    name="requirements",
    source="pyproject.toml",
)

resources(
    name="package_metadata",
    sources=[
        "LICENSE",
        "NOTICE",
        "README.md",
    ],
)

python_distribution(
    name="dist",
    dependencies=[
        ":package_metadata",
        "src/kvdb:lib",
    ],
    long_description_path="README.md",
    provides=python_artifact(
        name="kvdb",
        version="0.0.1",
        author="Keyur Gabani",
        description="Experimental indexing and retrieval infrastructure for LLM KV caches",
        license="Apache-2.0",
        license_files=["LICENSE", "NOTICE"],
        long_description_content_type="text/markdown",
        python_requires=">=3.11",
    ),
)
