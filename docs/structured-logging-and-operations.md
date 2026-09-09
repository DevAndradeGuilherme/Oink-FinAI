# Logs estruturados e diagnóstico operacional

Esta fase usa somente `logging` da biblioteca padrão e a instrumentação `PipelineTiming`
existente. Não há agente de logs, Redis, endpoint administrativo, alerta externo ou
autorreconciliação. Em produção, API, worker e Alembic escrevem um objeto JSON por linha em stdout;
o Docker aplica retenção limitada.

## Contrato fechado dos logs

Produção exige `LOG_FORMAT=json` e `LOG_INCLUDE_TRACEBACK=false`. O default de
`LOG_LEVEL` é `INFO`. O Compose de desenvolvimento declara `LOG_FORMAT=console` no exemplo para
preservar saída legível.

Todo JSON contém `timestamp` UTC, `level`, `event` e `service`. Somente estes campos adicionais
podem aparecer:

`correlation_id`, `processed_message_id`, `outbound_message_id`, `operation`, `stage`, `outcome`,
`attempt_number`, `duration_ms`, `status_code`, `error_code`, `next_attempt_at`, `sequence_no`,
`sequence_count`, `source_type`, `size_bytes`, `mime_type`, `audio_duration_seconds`, `row_count`,
`group_count`, `page_count`, `intent`, `metric`, `group`, `method`, `route` e
`exception_class`.

UUIDs são normalizados e campos numéricos possuem faixas finitas. Códigos, operações, stages,
outcomes, MIME, método e rota também passam por allowlists. Campo desconhecido é descartado;
evento desconhecido vira `application_log`; código inválido vira `UNKNOWN`. A mensagem livre do
`LogRecord`, seus argumentos e valores não reconhecidos nunca são serializados.

Eventos operacionais permitidos são: `application_log`, `asyncio_runtime`, `database_library`,
`library_log`, `migration_runtime`, `openai_operation_failed`, `server_lifecycle`,
`webhook_http_completed`, `worker_heartbeat_failed`, `worker_iteration_failed` e
`worker_resource_close_failed`.

Os eventos do pipeline permanecem os já existentes:

- webhook: `webhook_received`, `access_filter_completed`, `inbound_persisted` e
  `webhook_completed`;
- processamento: `processing_claimed`, `queue_wait_completed`, `processing_completed`, downloads,
  transcrição, análise/checkpoint de imagem, interpretação e persistência com sufixos
  `_started`/`_completed`;
- consulta: interpretação, plano, execução, formatação, criação da outbox e conclusão;
- outbox: `outbox_claimed`, `outbox_queue_wait_completed`, `outbound_send_started`,
  `outbound_send_completed` e `outbound_accepted`.

O mesmo UUID interno acompanha webhook, worker, operação OpenAI e outbox. Eventos da outbox também
incluem o UUID interno da mensagem de saída e, quando existente, o UUID de `ProcessedMessage`,
além de sequência e tentativa. O acesso bruto do webhook registra somente `POST`, a rota fixa,
status, outcome e duração monotônica; query string, IP, headers e corpo não entram no registro.

## Dados proibidos

Não entram em logs telefone, JID/`remoteJid`, nome, mensagem, `accepted_text`, transcript, descrição,
valor, categoria, estabelecimento, pagamento, legenda, prompt, resposta OpenAI, imagem, áudio,
bytes brutos, base64, `mediaKey`, payload, body, headers, API key, token, secret, `DATABASE_URL` nem
URL completa com query string. A política vale mesmo que esses dados sejam passados como mensagem,
argumento, exceção ou campo desconhecido.

Exceções preservam apenas classe validada e código interno. `str(exception)`, `repr(exception)` e
traceback não são formatados em JSON. A propagação das exceções do pipeline não muda. Traceback de
desenvolvimento continua responsabilidade do formato console; produção rejeita configuração que
o habilite.

## Logs de bibliotecas

`uvicorn.error`, FastAPI, SQLAlchemy, Alembic e asyncio propagam somente para o formatter seguro;
suas mensagens livres não são copiadas. `uvicorn.access` fica desabilitado e é substituído pelo
evento sanitizado do middleware do webhook. Os loggers de `httpx`, `httpcore` e OpenAI ficam
desabilitados em produção, sem request, response, headers, payload ou URL. Eventos técnicos de
startup/shutdown do Uvicorn e Alembic permanecem visíveis como `server_lifecycle` e
`migration_runtime`, sem mensagem livre.

## Rotação Docker

PostgreSQL, migrate, API e worker usam o driver `local` com `max-size=10m` e `max-file=5` em
`docker-compose.prod.yml`. Assim, cada container possui tamanho limitado e o job one-shot também
não acumula logs indefinidamente. A política não altera o Compose de desenvolvimento.

## Diagnóstico somente leitura

Execute no ambiente runtime, nunca com credencial bootstrap ou migrator:

```console
python -m oink_finai.operational_check
```

O comando imprime exatamente uma linha JSON sanitizada e encerra conexões. Exit codes:

- `0`: todos os checks em `ok`;
- `1`: existe ao menos um `warning`, sem condição crítica;
- `2`: existe condição `critical` ou PostgreSQL/configuração está indisponível.

Checks disponíveis:

| Check | Critério |
| --- | --- |
| `database` | conexão e relógio UTC básico do PostgreSQL |
| `schema` | `alembic_version` exatamente em `20260910_0014` |
| `api_heartbeat` | `not applicable`; a API usa `/live` e `/ready` |
| `worker_heartbeat` | ao menos um worker `RUNNING` recente; stale/ausente/parado é crítico |
| `processing_queue` | contagem e idade de `PENDING/PROCESSING` |
| `processing_retries_overdue` | retries cujo `next_attempt_at` já venceu |
| `processing_locks_expired` | locks além do timeout; qualquer ocorrência é crítica |
| `outbox_queue` | contagem e idade de `PENDING/SENDING` |
| `outbox_unknown` / `outbox_failed` | ocorrências que exigem análise manual |
| `query_sequences_blocked` | páginas bloqueadas atrás de `UNKNOWN/FAILED` |
| `usage_reserved_stale` | reservas `RESERVED` antigas |
| `usage_ambiguous` | reservas com transmissão ambígua |

Defaults de fila de processamento: warning em 25 registros ou 120s; critical em 100 registros ou
600s. Para outbox: warning em 10 registros ou 60s; critical em 50 registros ou 300s. O timeout
total de banco é 5s. Os limites são configurados pelas variáveis `OPERATIONAL_*`; stale de worker,
lock e reserva reutilizam os thresholds duráveis já existentes.

O comando nunca atualiza heartbeat, reenfileira mensagem, libera reserva, desbloqueia sequência,
reenvia `UNKNOWN`/`FAILED`, altera Expense ou apaga dados. Cron, Uptime Kuma ou um monitor futuro
pode consumir somente stdout e exit code. A integração com um canal de alertas e a reconciliação
manual da outbox pertencem a fases posteriores.
