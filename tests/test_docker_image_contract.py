import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_runtime_image_installs_project_and_verifies_openai_import() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert any(dependency.startswith("openai") for dependency in project["dependencies"])
    assert "--no-compile ." in dockerfile
    assert 'python -c "import openai; import oink_finai"' in dockerfile
