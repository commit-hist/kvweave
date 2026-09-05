from packaging.requirements import Requirement
import pytest

from scripts.check_wheel import _extra_requirement, _requirement_key


@pytest.mark.parametrize(
    ("project", "installed"),
    [
        ("Example_Pkg>=1,<3", "example-pkg (<3,>=1)"),
        ("example[Foo_Bar]>=1", "Example[foo-bar]>=1"),
        ('example>=1; python_version < "3.12"', "example>=1; python_version < '3.12'"),
    ],
)
def test_requirement_keys_normalize_backend_formatting(
    project: str, installed: str
) -> None:
    assert _requirement_key(project) == _requirement_key(installed)


@pytest.mark.parametrize(
    "marker",
    ['python_version < "3.12"', 'python_version < "3.12" or sys_platform == "win32"'],
)
def test_optional_requirement_markers_keep_boolean_grouping(marker: str) -> None:
    expected = _extra_requirement(f"example>=1; {marker}", "test")
    installed = f"example>=1; ({marker}) and extra == 'test'"
    assert _requirement_key(expected) == _requirement_key(installed)
    requirement = Requirement(expected)
    assert requirement.marker is not None
    assert requirement.marker.evaluate({"extra": "test", "python_version": "3.11"})
    assert not requirement.marker.evaluate({"extra": "other", "python_version": "3.11"})


def test_requirement_keys_do_not_hide_different_markers_or_extras() -> None:
    assert _requirement_key('example; python_version < "3.12"') != _requirement_key(
        'example; python_version >= "3.12"'
    )
    assert _requirement_key("example[foo]") != _requirement_key("example[bar]")
