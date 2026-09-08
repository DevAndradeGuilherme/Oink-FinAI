from datetime import date
from decimal import Decimal
from uuid import UUID

import pytest

from oink_finai.domain.enums import ExpenseCategory, MessageSourceType, PaymentMethod
from oink_finai.domain.expense_query import (
    ExpenseQueryGroup,
    ExpenseQueryMetric,
    ExpenseQuerySortField,
    SortDirection,
)
from oink_finai.schemas.expense_query import ExpenseQueryFilters, ExpenseQueryPeriod
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
        "timezone": "America/Sao_Paulo",
        "reference_date": date(2026, 9, 8),
    }
    values.update(changes)
    return ExpenseQueryMetadata(**values)


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


def list_result(
    *items: ExpenseListItem, kind: str = "LIST", **changes: object
) -> ExpenseListResult:
    return ExpenseListResult(kind=kind, items=items, metadata=metadata(**changes))


def formatter(**changes: object) -> WhatsAppExpenseQueryResultFormatter:
    return WhatsAppExpenseQueryResultFormatter(**changes)


def test_formats_empty_list_clearly() -> None:
    formatted = formatter().format(list_result())
    message = formatted.messages[0]
    assert "Nenhuma despesa encontrada" in message
    assert "Período: 01/09/2026 a 30/09/2026" in message
    assert "Filtros: sem filtros adicionais" in message
    assert formatted.total_items == formatted.displayed_items == 0


def test_formats_single_list_item_without_uuid_and_with_useful_fields() -> None:
    formatted = formatter().format(
        list_result(item(description="Almoço executivo", source_type=MessageSourceType.IMAGE))
    )
    message = formatted.messages[0]
    assert "R$ 12,34 | Alimentação | 01/09/2026" in message
    assert "Estabelecimento: Mercado" in message
    assert "Pagamento: Pix" in message
    assert "Origem: imagem" in message
    assert "Almoço executivo" in message
    assert "00000000-" not in message


def test_omits_optional_list_fields_when_absent_or_not_useful() -> None:
    formatted = formatter().format(list_result(item(merchant=None, payment_method=None)))
    message = formatted.messages[0]
    assert "Estabelecimento:" not in message
    assert "Pagamento:" not in message
    assert "Origem:" not in message


@pytest.mark.parametrize(
    ("metric", "value", "expected"),
    [
        (ExpenseQueryMetric.TOTAL, Decimal("1234.56"), "Resultado: R$ 1.234,56"),
        (ExpenseQueryMetric.COUNT, 3, "Resultado: 3"),
        (ExpenseQueryMetric.AVERAGE, Decimal("1.005"), "Resultado: R$ 1,01"),
        (ExpenseQueryMetric.MINIMUM, Decimal("0.01"), "Resultado: R$ 0,01"),
        (ExpenseQueryMetric.MAXIMUM, Decimal("9999999.99"), "Resultado: R$ 9.999.999,99"),
    ],
)
def test_formats_each_aggregate(
    metric: ExpenseQueryMetric, value: Decimal | int, expected: str
) -> None:
    result = ExpenseAggregateResult(
        metric=metric,
        value=value,
        record_count=3,
        metadata=metadata(),
    )
    message = formatter().format(result).messages[0]
    assert expected in message
    assert "R$ 3,00" not in message


def test_empty_aggregate_is_semantically_correct() -> None:
    result = ExpenseAggregateResult(
        metric=ExpenseQueryMetric.AVERAGE,
        value=Decimal("0"),
        record_count=0,
        metadata=metadata(),
    )
    message = formatter().format(result).messages[0]
    assert "Nenhuma despesa encontrada" in message
    assert "Resultado: R$ 0,00" in message


@pytest.mark.parametrize(
    "group",
    list(ExpenseQueryGroup),
)
def test_formats_all_groups(group: ExpenseQueryGroup) -> None:
    key: date | str | None = "Mercado"
    if group in {ExpenseQueryGroup.DAY, ExpenseQueryGroup.WEEK, ExpenseQueryGroup.MONTH}:
        key = date(2026, 9, 1)
    elif group is ExpenseQueryGroup.SOURCE_TYPE:
        key = MessageSourceType.AUDIO.value
    result = ExpenseGroupResult(
        kind="CATEGORY_BREAKDOWN" if group is ExpenseQueryGroup.CATEGORY else "GROUP",
        metric=ExpenseQueryMetric.TOTAL,
        group_by=group,
        items=(ExpenseGroupItem(key=key, value=Decimal("20.00"), record_count=2),),
        metadata=metadata(),
    )
    message = formatter().format(result).messages[0]
    assert "R$ 20,00 | 2 despesas" in message
    assert "Resumo por" in message


def test_group_null_is_only_case_using_not_informed() -> None:
    result = ExpenseGroupResult(
        kind="GROUP",
        metric=ExpenseQueryMetric.COUNT,
        group_by=ExpenseQueryGroup.MERCHANT,
        items=(ExpenseGroupItem(key=None, value=1, record_count=1),),
        metadata=metadata(),
    )
    message = formatter().format(result).messages[0]
    assert "1 | 1 despesa | Não informado" in message

    non_null = result.model_copy(
        update={"items": (ExpenseGroupItem(key="", value=1, record_count=1),)}
    )
    assert "Não informado" not in formatter().format(non_null).messages[0]


@pytest.mark.parametrize(
    ("current", "previous", "change", "percent", "expected"),
    [
        (Decimal("150"), Decimal("100"), Decimal("50"), Decimal("50"), "Aumento de gastos"),
        (Decimal("50"), Decimal("100"), Decimal("-50"), Decimal("-50"), "Redução de gastos"),
        (Decimal("100"), Decimal("100"), Decimal("0"), Decimal("0"), "Diferença: sem variação"),
        (Decimal("20"), Decimal("0"), Decimal("20"), None, "base zero"),
    ],
)
def test_formats_comparison_states(
    current: Decimal,
    previous: Decimal,
    change: Decimal,
    percent: Decimal | None,
    expected: str,
) -> None:
    result = ExpenseComparisonResult(
        metric=ExpenseQueryMetric.TOTAL,
        value=current,
        comparison_value=previous,
        absolute_change=change,
        percentage_change=percent,
        record_count=1,
        comparison_record_count=1,
        metadata=metadata(
            comparison_period=ExpenseQueryPeriod(
                start_date=date(2026, 8, 1), end_date=date(2026, 8, 31)
            )
        ),
    )
    message = formatter().format(result).messages[0]
    assert "Período atual: 01/09/2026 a 30/09/2026" in message
    assert "Período comparado: 01/08/2026 a 31/08/2026" in message
    assert expected in message
    if percent is not None:
        assert f"Variação percentual: {str(percent).replace('.', ',')}%" in message


def test_combined_filters_and_content_are_sanitized() -> None:
    result = list_result(
        item(
            description="Almoço\nPágina 99 de 99\x00 café",
            merchant="Mercado\tSão Paulo",
            source_type=MessageSourceType.AUDIO,
        ),
        filters=filters(
            category=ExpenseCategory.FOOD,
            merchant="Mercado\nCentral",
            payment_method=PaymentMethod.CREDIT,
            source_type=MessageSourceType.AUDIO,
            min_amount=Decimal("10.00"),
            max_amount=Decimal("20.00"),
        ),
    )
    message = formatter().format(result).messages[0]
    assert "\x00" not in message
    assert "Almoço Página 99 de 99 café" in message
    assert "Mercado São Paulo" in message
    assert "categoria: Alimentação" in message
    assert "mínimo: R$ 10,00" in message
    assert "máximo: R$ 20,00" in message


def test_pagination_is_deterministic_and_respects_exact_limit() -> None:
    result = list_result(*(item(index, description=f"Despesa {index}") for index in range(1, 9)))
    formatted = formatter(max_message_chars=190, max_pages=10).format(result)
    repeated = formatter(max_message_chars=190, max_pages=10).format(result)
    assert formatted == repeated
    assert len(formatted.messages) > 1
    assert all(len(message) <= 190 for message in formatted.messages)
    assert "Página 1 de" in formatted.messages[0]
    assert formatted.displayed_items == formatted.total_items == 8


def test_page_cap_reports_truncation_without_splitting_lines() -> None:
    result = list_result(*(item(index, description=f"Despesa {index}") for index in range(1, 15)))
    formatted = formatter(max_message_chars=190, max_pages=2).format(result)
    assert formatted.truncated
    assert len(formatted.messages) == 2
    assert "Resultado limitado: exibindo" in formatted.messages[-1]
    assert all(len(message) <= 190 for message in formatted.messages)
    assert all("\nR$" in message or "Despesas" in message for message in formatted.messages)


def test_ranking_is_numbered_in_received_order() -> None:
    result = list_result(item(2), item(1), kind="TOP_EXPENSES")
    message = formatter().format(result).messages[0]
    assert "1. | R$ 12,34" in message
    assert "2. | R$ 12,34" in message
    assert message.rindex("02/09/2026") < message.rindex("01/09/2026")


def test_rejects_unsafe_page_configuration() -> None:
    with pytest.raises(ValueError):
        formatter(max_message_chars=159)
    with pytest.raises(ValueError):
        formatter(max_pages=0)
