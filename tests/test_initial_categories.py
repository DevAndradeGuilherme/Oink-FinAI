from pathlib import Path
from runpy import run_path

CATEGORIES = run_path(
    Path(__file__).parents[1] / "migrations/versions/20260901_0001_initial_schema.py"
)["CATEGORIES"]

EXPECTED_CATEGORY_NAMES = {
    "Alimentação",
    "Transporte",
    "Moradia",
    "Saúde",
    "Educação",
    "Lazer",
    "Compras",
    "Assinaturas",
    "Contas",
    "Impostos",
    "Trabalho",
    "Viagem",
    "Outros",
}


def test_initial_migration_contains_required_categories() -> None:
    assert {name for name, _slug in CATEGORIES} == EXPECTED_CATEGORY_NAMES
    assert len({slug for _name, slug in CATEGORIES}) == len(CATEGORIES)
