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
        "src/kvweave:lib",
    ],
    long_description_path="README.md",
    provides=python_artifact(
        name="kvweave",
        version="0.0.1",
        author="Keyur Gabani",
        description="Experimental research infrastructure for shared KV-cache retrieval and indexing in transformer inference.",
        license="Apache-2.0",
        license_files=["LICENSE", "NOTICE"],
        long_description_content_type="text/markdown",
        python_requires=">=3.11",
        # Keep in sync with [project] in pyproject.toml: Pants builds the
        # distribution from these kwargs, not from the [project] table.
        keywords=["attention", "kv-cache", "llm", "retrieval", "transformers"],
        classifiers=[
            "Development Status :: 2 - Pre-Alpha",
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.11",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        project_urls={
            "Homepage": "https://github.com/commit-hist/kvweave",
            "Repository": "https://github.com/commit-hist/kvweave",
            "Issues": "https://github.com/commit-hist/kvweave/issues",
        },
        extras_require={
            "model-experiment": ["transformers==5.15.1"],
            "test": ["pytest>=8,<9", "transformers==5.15.1"],
        },
    ),
)
