import unicodedata
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

from oink_finai.domain.enums import MessageSourceType
from oink_finai.domain.expense_query import ExpenseQueryGroup, ExpenseQueryMetric
from oink_finai.schemas.expense_query_messages import ExpenseQueryFormattedMessages
from oink_finai.schemas.expense_query_result import (
    ExpenseAggregateResult,
    ExpenseComparisonResult,
    ExpenseGroupItem,
    ExpenseGroupResult,
    ExpenseListItem,
    ExpenseListResult,
    ExpenseQueryMetadata,
    ExpenseQueryResult,
)
from oink_finai.services.expense_query_result_formatter import ExpenseQueryResultFormatter

_MONEY_QUANTUM = Decimal("0.01")
_PERCENT_QUANTUM = Decimal("0.01")
_MONTH_NAMES = (
    "janeiro",
    "fevereiro",
    "março",
    "abril",
    "maio",
    "junho",
    "julho",
    "agosto",
    "setembro",
    "outubro",
    "novembro",
    "dezembro",
)
_METRIC_LABELS = {
    ExpenseQueryMetric.TOTAL: "Total gasto",
    ExpenseQueryMetric.COUNT: "Quantidade de despesas",
    ExpenseQueryMetric.AVERAGE: "Gasto médio",
    ExpenseQueryMetric.MINIMUM: "Menor gasto",
    ExpenseQueryMetric.MAXIMUM: "Maior gasto",
}
_GROUP_LABELS = {
    ExpenseQueryGroup.DAY: "dia",
    ExpenseQueryGroup.WEEK: "semana",
    ExpenseQueryGroup.MONTH: "mês",
    ExpenseQueryGroup.CATEGORY: "categoria",
    ExpenseQueryGroup.MERCHANT: "estabelecimento",
    ExpenseQueryGroup.PAYMENT_METHOD: "pagamento",
    ExpenseQueryGroup.SOURCE_TYPE: "origem",
}
_SOURCE_LABELS = {
    MessageSourceType.TEXT: "texto",
    MessageSourceType.AUDIO: "áudio",
    MessageSourceType.IMAGE: "imagem",
}


class WhatsAppExpenseQueryResultFormatter(ExpenseQueryResultFormatter):
    """Pure, deterministic pt-BR text formatter with bounded pagination."""

    def __init__(self, *, max_message_chars: int = 3_500, max_pages: int = 10) -> None:
        if max_message_chars < 160:
            raise ValueError("max_message_chars must be at least 160")
        if max_pages < 1:
            raise ValueError("max_pages must be positive")
        self._max_message_chars = max_message_chars
        self._max_pages = max_pages

    def format(self, result: ExpenseQueryResult) -> ExpenseQueryFormattedMessages:
        if isinstance(result, ExpenseListResult):
            return self._format_list(result)
        if isinstance(result, ExpenseAggregateResult):
            return self._format_aggregate(result)
        if isinstance(result, ExpenseGroupResult):
            return self._format_group(result)
        if isinstance(result, ExpenseComparisonResult):
            return self._format_comparison(result)
        raise TypeError("result type is not supported")

    def _format_list(self, result: ExpenseListResult) -> ExpenseQueryFormattedMessages:
        title = "Maiores gastos" if result.kind == "TOP_EXPENSES" else "Despesas"
        if not result.items:
            return self._paginate(
                title,
                result.metadata,
                ("Nenhuma despesa encontrada no período.",),
                total_items=0,
            )
        lines = tuple(
            self._list_line(item, index if result.kind == "TOP_EXPENSES" else None)
            for index, item in enumerate(result.items, start=1)
        )
        return self._paginate(title, result.metadata, lines, total_items=len(lines))

    def _format_aggregate(self, result: ExpenseAggregateResult) -> ExpenseQueryFormattedMessages:
        lines = [f"Métrica: {_METRIC_LABELS[result.metric]}"]
        if result.record_count == 0:
            lines.append("Nenhuma despesa encontrada no período.")
        lines.append(self._metric_line(result.metric, result.value, "Resultado"))
        return self._paginate(
            "Resumo de despesas", result.metadata, tuple(lines), total_items=len(lines)
        )

    def _format_group(self, result: ExpenseGroupResult) -> ExpenseQueryFormattedMessages:
        title = f"Resumo por {_GROUP_LABELS[result.group_by]}"
        if not result.items:
            return self._paginate(
                title,
                result.metadata,
                ("Nenhum agrupamento encontrado no período.",),
                total_items=0,
            )
        lines = tuple(
            self._group_line(item, result.metric, result.group_by) for item in result.items
        )
        return self._paginate(title, result.metadata, lines, total_items=len(lines))

    def _format_comparison(self, result: ExpenseComparisonResult) -> ExpenseQueryFormattedMessages:
        metadata = result.metadata
        lines = [
            f"Período atual: {self._period(metadata.period)}",
            f"Período comparado: {self._period(metadata.comparison_period)}",
            self._metric_line(result.metric, result.value, "Atual"),
            self._metric_line(result.metric, result.comparison_value, "Comparado"),
        ]
        if result.absolute_change == 0:
            lines.append("Diferença: sem variação")
        elif result.absolute_change > 0:
            lines.append(
                f"Aumento de gastos: {self._format_metric(result.metric, result.absolute_change)}"
            )
        else:
            lines.append(
                f"Redução de gastos: {self._format_metric(result.metric, -result.absolute_change)}"
            )
        if result.percentage_change is None:
            lines.append("Variação percentual: não disponível (base zero)")
        else:
            lines.append(f"Variação percentual: {self._percent(result.percentage_change)}")
        if result.record_count == 0 and result.comparison_record_count == 0:
            lines.append("Nenhuma despesa encontrada nos dois períodos.")
        return self._paginate(
            "Comparação de despesas",
            metadata,
            tuple(lines),
            total_items=len(lines),
            include_period=False,
        )

    def _paginate(
        self,
        title: str,
        metadata: ExpenseQueryMetadata,
        lines: tuple[str, ...],
        *,
        total_items: int,
        include_period: bool = True,
    ) -> ExpenseQueryFormattedMessages:
        base_header = self._header(title, metadata, include_period=include_period)
        reserve = len(f"\nPágina {self._max_pages} de {self._max_pages}")
        line_limit = self._max_message_chars - len(base_header) - reserve - 1
        if line_limit < 16:
            raise ValueError("max_message_chars cannot fit safe query metadata")
        safe_lines = tuple(self._fit_line(line, line_limit) for line in lines)
        pages: list[list[str]] = [[]]
        for line in safe_lines:
            current = pages[-1]
            current_size = len(base_header) + reserve + sum(len(item) + 1 for item in current)
            if current and current_size + len(line) + 1 > self._max_message_chars:
                if len(pages) == self._max_pages:
                    break
                pages.append([])
            pages[-1].append(line)

        displayed_items = sum(len(page) for page in pages)
        truncated = displayed_items < len(safe_lines)
        while truncated:
            total_pages = len(pages)
            footer = f"Resultado limitado: exibindo {displayed_items} de {total_items} itens."
            final_page = "\n".join(
                [
                    base_header,
                    *pages[-1],
                    f"Página {total_pages} de {total_pages}",
                    footer,
                ]
            )
            if len(final_page) <= self._max_message_chars:
                break
            if not pages[-1]:
                if len(pages) == 1:
                    raise ValueError("max_message_chars cannot fit truncation notice")
                pages.pop()
                continue
            pages[-1].pop()
            displayed_items -= 1
        total_pages = len(pages)
        messages = tuple(
            self._page_message(
                base_header,
                page,
                page_index=index,
                total_pages=total_pages,
                truncated=truncated and index == total_pages,
                displayed_items=displayed_items,
                total_items=total_items,
            )
            for index, page in enumerate(pages, start=1)
        )
        return ExpenseQueryFormattedMessages(
            messages=messages,
            truncated=truncated,
            total_items=total_items,
            displayed_items=displayed_items if total_items else 0,
        )

    def _page_message(
        self,
        header: str,
        lines: list[str],
        *,
        page_index: int,
        total_pages: int,
        truncated: bool,
        displayed_items: int,
        total_items: int,
    ) -> str:
        parts = [header, *lines, f"Página {page_index} de {total_pages}"]
        if truncated:
            parts.append(f"Resultado limitado: exibindo {displayed_items} de {total_items} itens.")
        message = "\n".join(parts)
        if len(message) > self._max_message_chars:
            raise ValueError("formatted message exceeds configured limit")
        return message

    def _header(
        self,
        title: str,
        metadata: ExpenseQueryMetadata,
        *,
        include_period: bool,
    ) -> str:
        lines = [title]
        if include_period:
            lines.append(f"Período: {self._period(metadata.period)}")
        filter_prefix = "Filtros: "
        page_reserve = len(f"\nPágina {self._max_pages} de {self._max_pages}")
        footer_reserve = len("\nResultado limitado: exibindo 100 de 100 itens.")
        available = (
            self._max_message_chars
            - sum(len(line) + 1 for line in lines)
            - page_reserve
            - footer_reserve
            - len(filter_prefix)
        )
        lines.append(
            filter_prefix
            + self._fit_line(self._filters(metadata), max(1, available - len(filter_prefix)))
        )
        return "\n".join(lines)

    def _list_line(self, item: ExpenseListItem, position: int | None) -> str:
        parts = []
        if position is not None:
            parts.append(f"{position}.")
        parts.extend(
            [
                self._real(item.amount),
                self._clean(item.category.value),
                self._date(item.expense_date),
            ]
        )
        if item.merchant:
            parts.append(f"Estabelecimento: {self._clean(item.merchant)}")
        if item.payment_method:
            parts.append(f"Pagamento: {self._clean(item.payment_method.value)}")
        if item.source_type is not MessageSourceType.TEXT:
            parts.append(f"Origem: {_SOURCE_LABELS[item.source_type]}")
        parts.append(self._clean(item.description))
        return " | ".join(parts)

    def _group_line(
        self,
        item: ExpenseGroupItem,
        metric: ExpenseQueryMetric,
        group_by: ExpenseQueryGroup,
    ) -> str:
        count_label = "despesa" if item.record_count == 1 else "despesas"
        return " | ".join(
            [
                self._format_metric(metric, item.value),
                f"{item.record_count} {count_label}",
                self._group_key(item.key, group_by),
            ]
        )

    @staticmethod
    def _metric_line(metric: ExpenseQueryMetric, value: Decimal | int, label: str) -> str:
        return f"{label}: {WhatsAppExpenseQueryResultFormatter._format_metric(metric, value)}"

    @staticmethod
    def _format_metric(metric: ExpenseQueryMetric, value: Decimal | int) -> str:
        if metric is ExpenseQueryMetric.COUNT:
            return str(int(value))
        if not isinstance(value, Decimal):
            raise TypeError("monetary result must be Decimal")
        return WhatsAppExpenseQueryResultFormatter._real(value)

    @staticmethod
    def _real(value: Decimal) -> str:
        if not value.is_finite():
            raise ValueError("monetary value must be finite")
        rounded = value.quantize(_MONEY_QUANTUM, rounding=ROUND_HALF_UP)
        formatted = f"{rounded:,.2f}"
        return f"R$ {formatted.replace(',', '_').replace('.', ',').replace('_', '.')}"

    @staticmethod
    def _percent(value: Decimal) -> str:
        if not value.is_finite():
            raise ValueError("percentage must be finite")
        rounded = value.quantize(_PERCENT_QUANTUM, rounding=ROUND_HALF_UP)
        text = f"{rounded:f}".rstrip("0").rstrip(".")
        return f"{text.replace('.', ',')}%"

    @staticmethod
    def _date(value: date) -> str:
        return value.strftime("%d/%m/%Y")

    def _period(self, period) -> str:
        if period is None:
            return "todo período disponível"
        if period.start_date == period.end_date:
            return self._date(period.start_date)
        return f"{self._date(period.start_date)} a {self._date(period.end_date)}"

    def _filters(self, metadata: ExpenseQueryMetadata) -> str:
        filters = metadata.filters
        parts: list[str] = []
        if filters.category:
            parts.append(f"categoria: {self._clean(filters.category.value)}")
        if filters.merchant:
            parts.append(f"estabelecimento: {self._clean(filters.merchant)}")
        if filters.payment_method:
            parts.append(f"pagamento: {self._clean(filters.payment_method.value)}")
        if filters.source_type:
            parts.append(f"origem: {_SOURCE_LABELS[filters.source_type]}")
        if filters.min_amount is not None:
            parts.append(f"mínimo: {self._real(filters.min_amount)}")
        if filters.max_amount is not None:
            parts.append(f"máximo: {self._real(filters.max_amount)}")
        return "; ".join(parts) if parts else "sem filtros adicionais"

    def _group_key(self, key: date | str | None, group_by: ExpenseQueryGroup) -> str:
        if key is None:
            return "Não informado"
        if group_by is ExpenseQueryGroup.DAY and isinstance(key, date):
            return self._date(key)
        if group_by is ExpenseQueryGroup.WEEK and isinstance(key, date):
            return f"Semana de {self._date(key)}"
        if group_by is ExpenseQueryGroup.MONTH and isinstance(key, date):
            return f"{_MONTH_NAMES[key.month - 1].capitalize()} de {key.year}"
        if group_by is ExpenseQueryGroup.SOURCE_TYPE:
            try:
                return _SOURCE_LABELS[MessageSourceType(key)]
            except ValueError:
                return self._clean(str(key), empty_value="")
        return self._clean(str(key), empty_value="")

    @staticmethod
    def _clean(value: str, *, empty_value: str = "Não informado") -> str:
        normalized = unicodedata.normalize("NFC", value)
        safe_characters: list[str] = []
        for character in normalized:
            if character.isspace():
                safe_characters.append(" ")
            elif not unicodedata.category(character).startswith("C"):
                safe_characters.append(character)
        safe = "".join(safe_characters)
        return " ".join(safe.split()) or empty_value

    @staticmethod
    def _fit_line(line: str, limit: int) -> str:
        if len(line) <= limit:
            return line
        if limit < 2:
            raise ValueError("configured line limit is too small")
        return f"{line[: limit - 1].rstrip()}…"
