import unicodedata
from calendar import monthrange
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

from oink_finai.domain.enums import MessageSourceType
from oink_finai.domain.expense_query import ExpenseQueryGroup, ExpenseQueryMetric
from oink_finai.schemas.expense_query import ExpenseQueryPeriod
from oink_finai.schemas.expense_query_messages import (
    ExpenseQueryFormattedMessages,
    ExpenseQueryFormattingContext,
)
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
_GROUP_LABELS = {
    ExpenseQueryGroup.DAY: "dia",
    ExpenseQueryGroup.WEEK: "semana",
    ExpenseQueryGroup.MONTH: "mês",
    ExpenseQueryGroup.CATEGORY: "categoria",
    ExpenseQueryGroup.MERCHANT: "estabelecimento",
    ExpenseQueryGroup.PAYMENT_METHOD: "forma de pagamento",
    ExpenseQueryGroup.SOURCE_TYPE: "origem",
}
_SOURCE_LABELS = {
    MessageSourceType.TEXT: "texto",
    MessageSourceType.AUDIO: "áudio",
    MessageSourceType.IMAGE: "imagem",
}


class WhatsAppExpenseQueryResultFormatter(ExpenseQueryResultFormatter):
    """Pure, deterministic pt-BR formatter with bounded, safe pagination."""

    def __init__(self, *, max_message_chars: int = 3_500, max_pages: int = 10) -> None:
        if max_message_chars < 160:
            raise ValueError("max_message_chars must be at least 160")
        if max_pages < 1:
            raise ValueError("max_pages must be positive")
        self._max_message_chars = max_message_chars
        self._max_pages = max_pages

    def format(
        self,
        result: ExpenseQueryResult,
        *,
        context: ExpenseQueryFormattingContext,
    ) -> ExpenseQueryFormattedMessages:
        if not isinstance(context, ExpenseQueryFormattingContext):
            raise TypeError("trusted formatting context is required")
        if isinstance(result, ExpenseListResult):
            return self._format_list(result, context)
        if isinstance(result, ExpenseAggregateResult):
            return self._format_aggregate(result, context)
        if isinstance(result, ExpenseGroupResult):
            return self._format_group(result, context)
        if isinstance(result, ExpenseComparisonResult):
            return self._format_comparison(result, context)
        raise TypeError("result type is not supported")

    def _format_list(
        self, result: ExpenseListResult, context: ExpenseQueryFormattingContext
    ) -> ExpenseQueryFormattedMessages:
        if not result.items:
            return self._single(self._empty_message(result.metadata, context), total_items=0)
        period = self._period_inline(result.metadata.period, context)
        ranking = result.kind == "TOP_EXPENSES"
        intro = (
            f"Estas foram suas maiores despesas {period}:"
            if ranking
            else f"Estas são suas despesas {period}:"
        )
        filters, summary = self._filter_display(result.metadata)
        if filters:
            intro = f"{intro[:-1]}{filters}:"
        if summary:
            intro = f"{intro}\n{summary}"
        lines = tuple(
            self._list_line(item, index if ranking else None)
            for index, item in enumerate(result.items, start=1)
        )
        return self._paginate(intro, lines, total_items=len(lines))

    def _format_aggregate(
        self, result: ExpenseAggregateResult, context: ExpenseQueryFormattingContext
    ) -> ExpenseQueryFormattedMessages:
        if result.record_count == 0:
            return self._single(self._empty_message(result.metadata, context), total_items=0)
        filters, summary = self._filter_display(result.metadata)
        period_lead = self._period_lead(result.metadata.period, context)
        period_inline = self._period_inline(result.metadata.period, context)
        if result.metric is ExpenseQueryMetric.TOTAL:
            total = (
                " no total"
                if not filters and self._uses_total_qualifier(result.metadata, context)
                else ""
            )
            message = f"{period_lead}, você gastou {self._real(result.value)}{filters}{total}."
        elif result.metric is ExpenseQueryMetric.COUNT:
            count = int(result.value)
            noun = "gasto" if count == 1 else "gastos"
            message = f"{period_lead}, você registrou {count} {noun}{filters}."
        elif result.metric is ExpenseQueryMetric.AVERAGE:
            message = (
                f"Sua média de gastos {period_inline}{filters} foi de {self._real(result.value)}."
            )
        elif result.metric is ExpenseQueryMetric.MINIMUM:
            message = f"Seu menor gasto {period_inline}{filters} foi de {self._real(result.value)}."
        else:
            message = f"Seu maior gasto {period_inline}{filters} foi de {self._real(result.value)}."
        if summary:
            message = f"{message}\n{summary}"
        return self._single(message, total_items=1)

    def _format_group(
        self, result: ExpenseGroupResult, context: ExpenseQueryFormattingContext
    ) -> ExpenseQueryFormattedMessages:
        if not result.items:
            return self._single(self._empty_message(result.metadata, context), total_items=0)
        period = self._period_inline(result.metadata.period, context)
        filters, summary = self._filter_display(result.metadata)
        group = _GROUP_LABELS[result.group_by]
        introductions = {
            ExpenseQueryMetric.TOTAL: f"Seus gastos {period}{filters} por {group}:",
            ExpenseQueryMetric.COUNT: f"Sua quantidade de gastos {period}{filters} por {group}:",
            ExpenseQueryMetric.AVERAGE: f"Sua média de gastos {period}{filters} por {group}:",
            ExpenseQueryMetric.MINIMUM: f"Seus menores gastos {period}{filters} por {group}:",
            ExpenseQueryMetric.MAXIMUM: f"Seus maiores gastos {period}{filters} por {group}:",
        }
        intro = introductions[result.metric]
        if summary:
            intro = f"{intro}\n{summary}"
        lines = tuple(
            self._group_line(item, result.metric, result.group_by) for item in result.items
        )
        return self._paginate(intro, lines, total_items=len(lines))

    def _format_comparison(
        self, result: ExpenseComparisonResult, context: ExpenseQueryFormattingContext
    ) -> ExpenseQueryFormattedMessages:
        if result.record_count == 0 and result.comparison_record_count == 0:
            message = "Não encontrei despesas registradas nos dois períodos comparados."
            return self._single(self._with_filter_summary(message, result.metadata), total_items=0)

        filters, summary = self._filter_display(result.metadata)
        current_lead = self._period_lead(result.metadata.period, context)
        current_inline = self._period_inline(result.metadata.period, context)
        previous_lead = self._period_lead(result.metadata.comparison_period, context)
        previous_inline = self._period_inline(result.metadata.comparison_period, context)
        if result.metric is ExpenseQueryMetric.TOTAL:
            current = self._real(result.value)
            previous = self._real(result.comparison_value)
            if result.absolute_change == 0:
                message = f"Os gastos permaneceram iguais nos dois períodos: {current}{filters}."
            elif result.comparison_record_count == 0:
                message = (
                    f"Você gastou {current} {current_inline}{filters}. Como não houve gastos "
                    f"registrados {previous_inline}, não é possível calcular uma variação "
                    "percentual."
                )
            else:
                direction = "aumento" if result.absolute_change > 0 else "redução"
                change = self._real(abs(result.absolute_change))
                variation = self._variation_suffix(change, result.percentage_change, direction)
                message = (
                    f"{current_lead}, você gastou {current}{filters}. "
                    f"{previous_lead}, foram {previous} — {variation}."
                )
        else:
            message = self._comparison_for_other_metric(
                result, current_inline, previous_inline, filters
            )
        if summary:
            message = f"{message}\n{summary}"
        return self._single(message, total_items=1)

    def _comparison_for_other_metric(
        self,
        result: ExpenseComparisonResult,
        current_period: str,
        previous_period: str,
        filters: str,
    ) -> str:
        current = self._format_metric(result.metric, result.value)
        previous = self._format_metric(result.metric, result.comparison_value)
        labels = {
            ExpenseQueryMetric.COUNT: "A quantidade de gastos",
            ExpenseQueryMetric.AVERAGE: "A média de gastos",
            ExpenseQueryMetric.MINIMUM: "O menor gasto",
            ExpenseQueryMetric.MAXIMUM: "O maior gasto",
        }
        label = labels[result.metric]
        if result.absolute_change == 0:
            return f"{label} permaneceu igual nos dois períodos: {current}{filters}."
        direction = "aumento" if result.absolute_change > 0 else "redução"
        change = self._format_metric(result.metric, abs(result.absolute_change))
        variation = self._variation_suffix(change, result.percentage_change, direction)
        return (
            f"{label} {current_period}{filters} foi de {current}, contra {previous} "
            f"{previous_period} — {variation}."
        )

    def _paginate(
        self, header: str, lines: tuple[str, ...], *, total_items: int
    ) -> ExpenseQueryFormattedMessages:
        reserve = len(f"\nPágina {self._max_pages} de {self._max_pages}")
        truncation_reserve = len(
            f"\nResultado limitado: exibindo {total_items} de {total_items} itens."
        )
        safe_header = self._fit_multiline(
            header, self._max_message_chars - reserve - truncation_reserve - 18
        )
        line_limit = self._max_message_chars - len(safe_header) - reserve - 1
        if line_limit < 16:
            raise ValueError("max_message_chars cannot fit safe query metadata")
        safe_lines = tuple(self._fit_line(line, line_limit) for line in lines)
        pages: list[list[str]] = [[]]
        for line in safe_lines:
            current = pages[-1]
            size = len(safe_header) + reserve + sum(len(item) + 1 for item in current)
            if current and size + len(line) + 1 > self._max_message_chars:
                if len(pages) == self._max_pages:
                    break
                pages.append([])
            pages[-1].append(line)

        displayed_items = sum(len(page) for page in pages)
        truncated = displayed_items < len(safe_lines)
        while truncated:
            footer = f"Resultado limitado: exibindo {displayed_items} de {total_items} itens."
            candidate = self._page_message(
                safe_header,
                pages[-1],
                page_index=len(pages),
                total_pages=len(pages),
                truncation=footer,
            )
            if len(candidate) <= self._max_message_chars:
                break
            if not pages[-1]:
                if len(pages) == 1:
                    raise ValueError("max_message_chars cannot fit truncation notice")
                pages.pop()
            else:
                pages[-1].pop()
                displayed_items -= 1

        total_pages = len(pages)
        if total_pages == 1 and not truncated:
            messages = ("\n".join((safe_header, *pages[0])),)
        else:
            messages = tuple(
                self._page_message(
                    safe_header,
                    page,
                    page_index=index,
                    total_pages=total_pages,
                    truncation=(
                        f"Resultado limitado: exibindo {displayed_items} de {total_items} itens."
                        if truncated and index == total_pages
                        else None
                    ),
                )
                for index, page in enumerate(pages, start=1)
            )
        if any(len(message) > self._max_message_chars for message in messages):
            raise ValueError("formatted message exceeds configured limit")
        return ExpenseQueryFormattedMessages(
            messages=messages,
            truncated=truncated,
            total_items=total_items,
            displayed_items=displayed_items,
        )

    def _single(self, message: str, *, total_items: int) -> ExpenseQueryFormattedMessages:
        safe = self._fit_multiline(message, self._max_message_chars)
        return ExpenseQueryFormattedMessages(
            messages=(safe,), truncated=False, total_items=total_items, displayed_items=total_items
        )

    @staticmethod
    def _page_message(
        header: str,
        lines: list[str],
        *,
        page_index: int,
        total_pages: int,
        truncation: str | None,
    ) -> str:
        parts = [header, *lines, f"Página {page_index} de {total_pages}"]
        if truncation:
            parts.append(truncation)
        return "\n".join(parts)

    def _list_line(self, item: ExpenseListItem, position: int | None) -> str:
        prefix = f"{position}. " if position is not None else "• "
        details = [self._clean(item.description)]
        if item.merchant:
            details.append(f"no {self._clean(item.merchant)}")
        details.append(f"com {self._lower_first(self._clean(item.category.value))}")
        if item.payment_method:
            details.append(f"via {self._clean(item.payment_method.value)}")
        if item.source_type is not MessageSourceType.TEXT:
            details.append(f"registrada por {_SOURCE_LABELS[item.source_type]}")
        details.append(f"em {self._date(item.expense_date)}")
        return f"{prefix}{self._real(item.amount)} — {', '.join(details)}"

    def _group_line(
        self,
        item: ExpenseGroupItem,
        metric: ExpenseQueryMetric,
        group_by: ExpenseQueryGroup,
    ) -> str:
        key = self._group_key(item.key, group_by)
        value = self._format_metric(metric, item.value)
        if metric is ExpenseQueryMetric.COUNT:
            count = int(item.value)
            value = f"{count} {'gasto' if count == 1 else 'gastos'}"
        return f"• {key}: {value}"

    def _empty_message(
        self, metadata: ExpenseQueryMetadata, context: ExpenseQueryFormattingContext
    ) -> str:
        filters, summary = self._filter_display(metadata)
        period = self._period_inline(metadata.period, context)
        if not filters and not summary:
            return f"Você ainda não possui gastos registrados {period}."
        if summary:
            return (
                "Não encontrei despesas que correspondam a esses filtros no período informado."
                f"\n{summary}"
            )
        return f"Não encontrei gastos{filters} {period}."

    def _filter_display(self, metadata: ExpenseQueryMetadata) -> tuple[str, str | None]:
        entries = self._filter_entries(metadata)
        if not entries:
            return "", None
        if len(entries) <= 2:
            return " " + " ".join(phrase for phrase, _ in entries), None
        labels = self._join_pt_br([label for _, label in entries])
        return "", f"Filtros: {labels}."

    def _with_filter_summary(
        self, message: str, metadata: ExpenseQueryMetadata, *, inline: bool = True
    ) -> str:
        filters, summary = self._filter_display(metadata)
        if summary:
            return f"{message}\n{summary}"
        if inline and filters:
            return f"{message.rstrip('.')}{filters}."
        return message

    def _filter_entries(self, metadata: ExpenseQueryMetadata) -> list[tuple[str, str]]:
        filters = metadata.filters
        entries: list[tuple[str, str]] = []
        if filters.category:
            category = self._lower_first(self._clean(filters.category.value))
            entries.append((f"com {category}", category))
        if filters.merchant:
            merchant = self._clean(filters.merchant)
            entries.append((f"no {merchant}", merchant))
        if filters.payment_method:
            payment = self._clean(filters.payment_method.value)
            entries.append((f"via {payment}", f"via {payment}"))
        if filters.source_type:
            source = _SOURCE_LABELS[filters.source_type]
            entries.append((f"em despesas registradas por {source}", f"registradas por {source}"))
        if filters.min_amount is not None and filters.max_amount is not None:
            interval = (
                f"valores entre {self._real(filters.min_amount)} e {self._real(filters.max_amount)}"
            )
            entries.append((f"com {interval}", interval))
        elif filters.min_amount is not None:
            amount = f"valores a partir de {self._real(filters.min_amount)}"
            entries.append((f"com {amount}", amount))
        elif filters.max_amount is not None:
            amount = f"valores de até {self._real(filters.max_amount)}"
            entries.append((f"com {amount}", amount))
        return entries

    @staticmethod
    def _join_pt_br(parts: list[str]) -> str:
        if len(parts) == 1:
            return parts[0]
        if len(parts) == 2:
            return " e ".join(parts)
        return f"{', '.join(parts[:-1])} e {parts[-1]}"

    def _period_lead(
        self, period: ExpenseQueryPeriod | None, context: ExpenseQueryFormattingContext
    ) -> str:
        inline = self._period_inline(period, context)
        for prefix, replacement in (
            ("em ", "Em "),
            ("entre ", "Entre "),
            ("no ", "No "),
            ("na ", "Na "),
        ):
            if inline.startswith(prefix):
                return replacement + inline[len(prefix) :]
        return inline[0].upper() + inline[1:]

    def _period_inline(
        self, period: ExpenseQueryPeriod | None, context: ExpenseQueryFormattingContext
    ) -> str:
        if period is None:
            return "no período disponível"
        start, end = period.start_date, period.end_date
        reference = context.reference_date
        if start == end == reference:
            return "hoje"
        if start == end == reference - timedelta(days=1):
            return "ontem"
        week_start = reference - timedelta(days=reference.weekday())
        if start == week_start and end == week_start + timedelta(days=6):
            return "nesta semana"
        previous_week = week_start - timedelta(days=7)
        if start == previous_week and end == previous_week + timedelta(days=6):
            return "na semana passada"
        month_start = reference.replace(day=1)
        month_end = reference.replace(day=monthrange(reference.year, reference.month)[1])
        if start == month_start and end == month_end:
            return "neste mês"
        previous_month_end = month_start - timedelta(days=1)
        if start == previous_month_end.replace(day=1) and end == previous_month_end:
            return "no mês passado"
        if start == date(reference.year, 1, 1) and end == date(reference.year, 12, 31):
            return "neste ano"
        expected_end = date(start.year, start.month, monthrange(start.year, start.month)[1])
        if start.day == 1 and end == expected_end:
            return f"em {_MONTH_NAMES[start.month - 1]} de {start.year}"
        if start == end:
            return f"em {self._date(start)}"
        return f"entre {self._date(start)} e {self._date(end)}"

    def _uses_total_qualifier(
        self, metadata: ExpenseQueryMetadata, context: ExpenseQueryFormattingContext
    ) -> bool:
        return self._period_inline(metadata.period, context) not in {"hoje", "ontem"}

    @staticmethod
    def _variation_suffix(change: str, percentage: Decimal | None, direction: str) -> str:
        article = "um" if direction == "aumento" else "uma"
        if percentage is None:
            return f"{article} {direction} de {change}"
        percent = WhatsAppExpenseQueryResultFormatter._percent(abs(percentage))
        return f"{article} {direction} de {change} ({percent})"

    @staticmethod
    def _format_metric(metric: ExpenseQueryMetric, value: Decimal | int) -> str:
        if metric is ExpenseQueryMetric.COUNT:
            return str(int(value))
        if not isinstance(value, Decimal):
            raise TypeError("monetary result must be Decimal")
        return WhatsAppExpenseQueryResultFormatter._real(value)

    @staticmethod
    def _real(value: Decimal | int) -> str:
        if not isinstance(value, Decimal):
            raise TypeError("monetary result must be Decimal")
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
        text = f"{rounded:f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return f"{text.replace('.', ',')}%"

    @staticmethod
    def _date(value: date) -> str:
        return value.strftime("%d/%m/%Y")

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
    def _lower_first(value: str) -> str:
        return value[:1].lower() + value[1:]

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

    def _fit_multiline(self, value: str, limit: int) -> str:
        if len(value) <= limit:
            return value
        lines = value.splitlines()
        if len(lines) == 1:
            return self._fit_line(value, limit)
        first = lines[0]
        remaining = limit - len(first) - 1
        if remaining < 2:
            return self._fit_line(first, limit)
        return f"{first}\n{self._fit_line(' '.join(lines[1:]), remaining)}"

    @staticmethod
    def _fit_line(line: str, limit: int) -> str:
        if len(line) <= limit:
            return line
        if limit < 2:
            raise ValueError("configured line limit is too small")
        clusters: list[str] = []
        regional_open = False
        for character in line:
            codepoint = ord(character)
            category = unicodedata.category(character)
            is_modifier = 0x1F3FB <= codepoint <= 0x1F3FF
            is_variation = 0xFE00 <= codepoint <= 0xFE0F
            is_regional = 0x1F1E6 <= codepoint <= 0x1F1FF
            if clusters and (category.startswith("M") or is_modifier or is_variation):
                clusters[-1] += character
            elif is_regional and regional_open:
                clusters[-1] += character
                regional_open = False
            else:
                clusters.append(character)
                regional_open = is_regional
        kept: list[str] = []
        size = 1
        for cluster in clusters:
            if size + len(cluster) > limit:
                break
            kept.append(cluster)
            size += len(cluster)
        return f"{''.join(kept).rstrip()}…"
