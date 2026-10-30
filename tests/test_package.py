"""Packaging invariants: version, bundled web assets, typed marker, console script."""

from __future__ import annotations

import tomllib
from pathlib import Path

import sichuan_speech
from sichuan_speech.service import web_dir

ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_version_matches_pyproject():
    assert sichuan_speech.__version__ == _pyproject()["project"]["version"]


def test_console_script_is_declared():
    scripts = _pyproject()["project"]["scripts"]
    assert scripts["sichuan-speech"] == "sichuan_speech.cli:main"


def test_typed_marker_exists():
    assert (Path(sichuan_speech.__file__).parent / "py.typed").is_file()


def test_web_assets_are_bundled():
    assets = web_dir()
    for name in ("index.html", "app.js", "style.css", "favicon.svg"):
        path = assets / name
        assert path.is_file(), f"missing web asset: {name}"
        assert path.stat().st_size > 200, f"web asset looks empty: {name}"


def test_web_ui_has_no_external_requests():
    index = (web_dir() / "index.html").read_text(encoding="utf-8")
    app = (web_dir() / "app.js").read_text(encoding="utf-8")
    for content in (index, app):
        assert "http://" not in content
        assert "https://" not in content
        assert "cdn." not in content


def test_web_ui_references_local_assets_only():
    index = (web_dir() / "index.html").read_text(encoding="utf-8")
    assert 'href="/static/style.css"' in index
    assert 'src="/static/app.js"' in index
    assert 'href="/static/favicon.svg"' in index


def test_optional_dependency_groups_are_declared():
    extras = _pyproject()["project"]["optional-dependencies"]
    for group in ("serve", "train", "dev"):
        assert group in extras and extras[group]
    assert any(item.startswith("fastapi") for item in extras["serve"])
    assert any(item.startswith("torch") for item in extras["train"])


def test_package_data_includes_web():
    package_data = _pyproject()["tool"]["setuptools"]["package-data"]["sichuan_speech"]
    assert "web/*" in package_data
    assert "py.typed" in package_data


def test_readme_and_license_exist():
    assert (ROOT / "README.md").is_file()
    assert (ROOT / "LICENSE").is_file()
