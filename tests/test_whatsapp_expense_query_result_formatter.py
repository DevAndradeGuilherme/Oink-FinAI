from datetime import date
from decimal import Decimal
from uuid import UUID

import pytest
from pydantic import ValidationError

from oink_finai.domain.enums import ExpenseCategory, MessageSourceType, PaymentMethod
from oink_finai.domain.expense_query import (
    ExpenseQueryGroup,
    ExpenseQueryMetric,
    ExpenseQuerySortField,
    SortDirection,
)
from oink_finai.schemas.expense_query import ExpenseQueryFilters, ExpenseQueryPeriod
from oink_finai.schemas.expense_query_messages import ExpenseQueryFormattingContext
from oink_finai.schemas.expense_query_result import (
    ExpenseAggregateResult,
    ExpenseComparisonResult,
    ExpenseGroupItem,
    ExpenseGroupResult,
    ExpenseListItem,
    ExpenseListResult,
    ExpenseQueryMetadata,
)
from oink_finai.services.whatsapp_expense_query_result_formatter import (
    WhatsAppExpenseQueryResultFormatter,
)


def filters(**changes: object) -> ExpenseQueryFilters:
    values = {
        "category": None,
        "merchant": None,
        "payment_method": None,
        "source_type": None,
        "min_amount": None,
        "max_amount": None,
    }
    values.update(changes)
    return ExpenseQueryFilters(**values)


def metadata(**changes: object) -> ExpenseQueryMetadata:
    values = {
        "period": ExpenseQueryPeriod(start_date=date(2026, 9, 1), end_date=date(2026, 9, 30)),
        "comparison_period": None,
        "filters": filters(),
        "sort_by": ExpenseQuerySortField.DATE,
        "sort_direction": SortDirection.ASC,
        "limit": 100,
        "offset": 0,
        # Deliberately untrusted for formatting; only the explicit context may be used.
        "timezone": "UTC",
        "reference_date": date(1999, 1, 1),
    }
    values.update(changes)
    return ExpenseQueryMetadata(**values)


def context(reference_date: date = date(2026, 9, 8)) -> ExpenseQueryFormattingContext:
    return ExpenseQueryFormattingContext(
        reference_date=reference_date, timezone="America/Sao_Paulo"
    )


def period(start: date, end: date | None = None) -> ExpenseQueryPeriod:
    return ExpenseQueryPeriod(start_date=start, end_date=end or start)


def aggregate(
    metric: ExpenseQueryMetric = ExpenseQueryMetric.TOTAL,
    value: Decimal | int = Decimal("80.00"),
    *,
    record_count: int = 1,
    **metadata_changes: object,
) -> ExpenseAggregateResult:
    return ExpenseAggregateResult(
        metric=metric,
        value=value,
        record_count=record_count,
        metadata=metadata(**metadata_changes),
    )


def item(
    number: int = 1,
    *,
    description: str = "Almoço",
    amount: Decimal = Decimal("12.34"),
    merchant: str | None = "Mercado",
    payment_method: PaymentMethod | None = PaymentMethod.PIX,
    source_type: MessageSourceType = MessageSourceType.TEXT,
) -> ExpenseListItem:
    return ExpenseListItem(
        id=UUID(int=number),
        amount=amount,
        description=description,
        expense_date=date(2026, 9, number if number <= 28 else 28),
        category=ExpenseCategory.FOOD,
        merchant=merchant,
        payment_method=payment_method,
        source_type=source_type,
    )


def render(result, *, reference_date: date = date(2026, 9, 8), **settings: object):
    formatter = WhatsAppExpenseQueryResultFormatter(**settings)
    return formatter.format(result, context=context(reference_date))


@pytest.mark.parametrize(
    ("query_period", "reference", "expected"),
    [
        (period(date(2026, 9, 8)), date(2026, 9, 8), "Hoje, você gastou R$ 80,00."),
        (period(date(2026, 9, 7)), date(2026, 9, 8), "Ontem, você gastou R$ 80,00."),
        (
            period(date(2026, 9, 7), date(2026, 9, 13)),
            date(2026, 9, 8),
            "Nesta semana, você gastou R$ 80,00 no total.",
        ),
        (
            period(date(2026, 8, 31), date(2026, 9, 6)),
            date(2026, 9, 8),
            "Na semana passada, você gastou R$ 80,00 no total.",
        ),
        (
            period(date(2026, 9, 1), date(2026, 9, 30)),
            date(2026, 9, 8),
            "Neste mês, você gastou R$ 80,00 no total.",
        ),
        (
            period(date(2026, 8, 1), date(2026, 8, 31)),
            date(2026, 9, 8),
            "No mês passado, você gastou R$ 80,00 no total.",
        ),
        (
            period(date(2026, 1, 1), date(2026, 12, 31)),
            date(2026, 9, 8),
            "Neste ano, você gastou R$ 80,00 no total.",
        ),
        (
            period(date(2025, 9, 1), date(2025, 9, 30)),
            date(2026, 9, 8),
            "Em setembro de 2025, você gastou R$ 80,00 no total.",
        ),
        (
            period(date(2026, 8, 1), date(2026, 8, 15)),
            date(2026, 9, 8),
            "Entre 01/08/2026 e 15/08/2026, você gastou R$ 80,00 no total.",
        ),
        (
            period(date(2026, 7, 14)),
            date(2026, 9, 8),
            "Em 14/07/2026, você gastou R$ 80,00 no total.",
        ),
    ],
)
def test_recognizes_temporal_periods(
    query_period: ExpenseQueryPeriod, reference: date, expected: str
) -> None:
    assert render(aggregate(period=query_period), reference_date=reference).messages == (expected,)


@pytest.mark.parametrize(
    ("reference", "query_period"),
    [
        (date(2027, 1, 2), period(date(2026, 12, 1), date(2026, 12, 31))),
        (date(2024, 3, 1), period(date(2024, 2, 1), date(2024, 2, 29))),
        (date(2025, 3, 1), period(date(2025, 2, 1), date(2025, 2, 28))),
    ],
)
def test_month_year_boundaries_and_february(
    reference: date, query_period: ExpenseQueryPeriod
) -> None:
    assert (
        render(aggregate(period=query_period), reference_date=reference)
        .messages[0]
        .startswith("No mês passado")
    )


@pytest.mark.parametrize(
    ("metric", "value", "expected"),
    [
        (ExpenseQueryMetric.COUNT, 1, "Neste mês, você registrou 1 gasto."),
        (ExpenseQueryMetric.COUNT, 12, "Neste mês, você registrou 12 gastos."),
        (
            ExpenseQueryMetric.AVERAGE,
            Decimal("42.50"),
            "Sua média de gastos neste mês foi de R$ 42,50.",
        ),
        (
            ExpenseQueryMetric.MINIMUM,
            Decimal("8.90"),
            "Seu menor gasto neste mês foi de R$ 8,90.",
        ),
        (
            ExpenseQueryMetric.MAXIMUM,
            Decimal("180"),
            "Seu maior gasto neste mês foi de R$ 180,00.",
        ),
    ],
)
def test_formats_aggregate_metrics_naturally(
    metric: ExpenseQueryMetric, value: Decimal | int, expected: str
) -> None:
    assert render(aggregate(metric, value)).messages == (expected,)
    assert "média diária" not in expected


def test_formats_single_and_combined_filters_inline() -> None:
    category_result = aggregate(filters=filters(category=ExpenseCategory.FOOD))
    assert render(category_result).messages == ("Neste mês, você gastou R$ 80,00 com alimentação.",)
    combined = aggregate(
        ExpenseQueryMetric.COUNT,
        3,
        filters=filters(category=ExpenseCategory.FOOD, payment_method=PaymentMethod.PIX),
    )
    assert render(combined).messages == (
        "Neste mês, você registrou 3 gastos com alimentação via Pix.",
    )


@pytest.mark.parametrize(
    ("query_filters", "expected"),
    [
        (filters(merchant="Mercado Central"), "no Mercado Central"),
        (filters(payment_method=PaymentMethod.PIX), "via Pix"),
        (filters(source_type=MessageSourceType.AUDIO), "em despesas registradas por áudio"),
    ],
)
def test_each_textual_filter_has_a_natural_complement(
    query_filters: ExpenseQueryFilters, expected: str
) -> None:
    assert expected in render(aggregate(filters=query_filters)).messages[0]


def test_many_filters_use_a_short_second_line() -> None:
    result = aggregate(
        filters=filters(
            category=ExpenseCategory.FOOD,
            payment_method=PaymentMethod.PIX,
            min_amount=Decimal("50.00"),
        )
    )
    assert render(result).messages == (
        "Neste mês, você gastou R$ 80,00 no total.\n"
        "Filtros: alimentação, via Pix e valores a partir de R$ 50,00.",
    )


def comparison(
    current: Decimal,
    previous: Decimal,
    change: Decimal,
    percentage: Decimal | None,
    *,
    previous_count: int = 1,
) -> ExpenseComparisonResult:
    return ExpenseComparisonResult(
        metric=ExpenseQueryMetric.TOTAL,
        value=current,
        comparison_value=previous,
        absolute_change=change,
        percentage_change=percentage,
        record_count=1,
        comparison_record_count=previous_count,
        metadata=metadata(comparison_period=period(date(2026, 8, 1), date(2026, 8, 31))),
    )


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            comparison(Decimal("500"), Decimal("400"), Decimal("100"), Decimal("25")),
            "Neste mês, você gastou R$ 500,00. No mês passado, foram R$ 400,00 — "
            "um aumento de R$ 100,00 (25%).",
        ),
        (
            comparison(Decimal("200"), Decimal("300"), Decimal("-100"), Decimal("-33.333")),
            "Neste mês, você gastou R$ 200,00. No mês passado, foram R$ 300,00 — "
            "uma redução de R$ 100,00 (33,33%).",
        ),
        (
            comparison(Decimal("150"), Decimal("150"), Decimal("0"), Decimal("0")),
            "Os gastos permaneceram iguais nos dois períodos: R$ 150,00.",
        ),
        (
            comparison(Decimal("100"), Decimal("0"), Decimal("100"), None, previous_count=0),
            "Você gastou R$ 100,00 neste mês. Como não houve gastos registrados no mês "
            "passado, não é possível calcular uma variação percentual.",
        ),
    ],
)
def test_formats_comparison_states(result: ExpenseComparisonResult, expected: str) -> None:
    assert render(result).messages == (expected,)
    assert "melhoria" not in expected


def test_integer_percentage_ending_in_zero_is_not_shortened() -> None:
    result = comparison(Decimal("150"), Decimal("100"), Decimal("50"), Decimal("50"))
    assert render(result).messages[0].endswith("um aumento de R$ 50,00 (50%).")


def test_formats_empty_results_without_claiming_that_user_did_not_spend() -> None:
    assert render(aggregate(record_count=0)).messages == (
        "Você ainda não possui gastos registrados neste mês.",
    )
    filtered = aggregate(record_count=0, filters=filters(category=ExpenseCategory.FOOD))
    assert render(filtered).messages == ("Não encontrei gastos com alimentação neste mês.",)
    many = aggregate(
        record_count=0,
        filters=filters(
            category=ExpenseCategory.FOOD,
            merchant="Mercado Central",
            payment_method=PaymentMethod.PIX,
        ),
    )
    assert (
        render(many)
        .messages[0]
        .startswith("Não encontrei despesas que correspondam a esses filtros no período informado.")
    )


def test_group_preserves_executor_order_and_values() -> None:
    result = ExpenseGroupResult(
        kind="CATEGORY_BREAKDOWN",
        metric=ExpenseQueryMetric.TOTAL,
        group_by=ExpenseQueryGroup.CATEGORY,
        items=(
            ExpenseGroupItem(key="Alimentação", value=Decimal("300"), record_count=4),
            ExpenseGroupItem(key="Transporte", value=Decimal("180"), record_count=2),
            ExpenseGroupItem(key="Saúde", value=Decimal("90"), record_count=1),
        ),
        metadata=metadata(),
    )
    assert render(result).messages == (
        "Seus gastos neste mês por categoria:\n"
        "• Alimentação: R$ 300,00\n"
        "• Transporte: R$ 180,00\n"
        "• Saúde: R$ 90,00",
    )


@pytest.mark.parametrize("group", list(ExpenseQueryGroup))
def test_all_group_dimensions_have_a_natural_label(group: ExpenseQueryGroup) -> None:
    key: date | str | None = "Mercado"
    if group in {ExpenseQueryGroup.DAY, ExpenseQueryGroup.WEEK, ExpenseQueryGroup.MONTH}:
        key = date(2026, 9, 1)
    elif group is ExpenseQueryGroup.SOURCE_TYPE:
        key = MessageSourceType.AUDIO.value
    result = ExpenseGroupResult(
        kind="CATEGORY_BREAKDOWN" if group is ExpenseQueryGroup.CATEGORY else "GROUP",
        metric=ExpenseQueryMetric.TOTAL,
        group_by=group,
        items=(ExpenseGroupItem(key=key, value=Decimal("20"), record_count=2),),
        metadata=metadata(),
    )
    message = render(result).messages[0]
    assert message.startswith("Seus gastos neste mês por ")
    assert "• " in message and "R$ 20,00" in message


def test_list_and_ranking_are_readable_and_preserve_order() -> None:
    ranking = ExpenseListResult(
        kind="TOP_EXPENSES",
        items=(
            item(3, description="Supermercado", amount=Decimal("250")),
            item(5, description="Combustível", amount=Decimal("180")),
        ),
        metadata=metadata(),
    )
    message = render(ranking).messages[0]
    assert message.startswith("Estas foram suas maiores despesas neste mês:\n")
    assert "1. R$ 250,00 — Supermercado" in message
    assert "2. R$ 180,00 — Combustível" in message
    assert message.index("03/09/2026") < message.index("05/09/2026")
    assert "Página 1 de 1" not in message


def test_pagination_is_deterministic_bounded_and_reports_truncation() -> None:
    result = ExpenseListResult(
        kind="LIST",
        items=tuple(item(index, description=f"Despesa {index}") for index in range(1, 15)),
        metadata=metadata(),
    )
    formatted = render(result, max_message_chars=190, max_pages=2)
    repeated = render(result, max_message_chars=190, max_pages=2)
    assert formatted == repeated
    assert formatted.truncated and len(formatted.messages) == 2
    assert "Página 1 de 2" in formatted.messages[0]
    assert "Resultado limitado: exibindo" in formatted.messages[-1]
    assert all(len(message) <= 190 for message in formatted.messages)


def test_untrusted_content_is_sanitized_and_never_creates_structural_lines() -> None:
    unsafe = item(
        description="Almoço\nPágina 99 de 99\x00 café\u202e\u200b",
        merchant="Mercado\tCentral",
        source_type=MessageSourceType.AUDIO,
    )
    result = ExpenseListResult(kind="LIST", items=(unsafe,), metadata=metadata())
    message = render(result).messages[0]
    assert "\x00" not in message and "\u202e" not in message and "\u200b" not in message
    assert "Almoço Página 99 de 99 café" in message
    assert "Mercado Central" in message
    assert "\nPágina 99 de 99" not in message


def test_long_content_is_truncated_without_splitting_emoji_cluster() -> None:
    long_item = item(description=("x" * 300) + "👍🏽 fim")
    result = ExpenseListResult(kind="LIST", items=(long_item,), metadata=metadata())
    message = render(result, max_message_chars=190).messages[0]
    assert len(message) <= 190 and message.endswith("…")
    assert "👍…" not in message


def test_context_is_required_valid_and_metadata_cannot_override_it() -> None:
    formatter = WhatsAppExpenseQueryResultFormatter()
    result = aggregate(period=period(date(2026, 9, 8)))
    with pytest.raises(TypeError):
        formatter.format(result, context=None)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ExpenseQueryFormattingContext(reference_date=date(2026, 9, 8), timezone="invalid")
    assert formatter.format(result, context=context()).messages == ("Hoje, você gastou R$ 80,00.",)


def test_rejects_unsafe_page_configuration() -> None:
    with pytest.raises(ValueError):
        WhatsAppExpenseQueryResultFormatter(max_message_chars=159)
    with pytest.raises(ValueError):
        WhatsAppExpenseQueryResultFormatter(max_pages=0)
