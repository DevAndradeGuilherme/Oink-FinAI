# Limites HTTP, saúde e heartbeat

Esta fase protege a entrada HTTP do webhook, separa liveness de readiness e registra a saúde do
worker no PostgreSQL. Ela não cria alertas externos, endpoint público do worker, integração com
Cloudflare, retenção de dados financeiros ou dependência de Redis.

## Limite bruto e capacidade do webhook

`POST /api/v1/webhooks/evolution` passa primeiro por middleware ASGI puro, antes de FastAPI,
Pydantic ou parsing JSON. O default `EVOLUTION_WEBHOOK_MAX_BODY_BYTES=262144` limita o corpo bruto
a 256 KiB, valor conservador para payloads Evolution sem mídia em base64. A instância deve manter
`webhookBase64=false`; bytes de mídia continuam sendo baixados separadamente e sujeitos aos
limites próprios já existentes.

O middleware rejeita antecipadamente `Content-Length` maior que o limite e também soma os bytes
reais de todos os chunks. Assim, ausência, valor menor que o corpo e divisão em vários chunks não
contornam o limite. Chunks são encaminhados sem construir uma segunda cópia do corpo. Um tamanho
exatamente igual ao limite é aceito. `Content-Length` malformado ou desconexão durante a leitura
recebem resposta mínima `400`; corpo vazio e JSON inválido seguem para a validação normal, sem
acionar provedores.

- `413`: corpo declarado ou efetivo acima do limite;
- `503`: capacidade simultânea esgotada ou timeout global do processamento;
- `EVOLUTION_WEBHOOK_MAX_CONCURRENCY=8`: limite por processo, com rejeição imediata em vez de
  espera;
- `EVOLUTION_WEBHOOK_HTTP_TIMEOUT_SECONDS=10`: prazo total de leitura e processamento HTTP.

O cancelamento por timeout é propagado ao handler e a sessão SQLAlchemy executa rollback antes de
fechar. O middleware não registra corpo, headers, URL completa nem exceções. Nenhuma identidade é
inferida de IP ou `X-Forwarded-For`. Os códigos `503` deliberadamente não revelam capacidade,
timeout ou valores internos.

Não há bloqueio obrigatório por freshness do timestamp do evento: o contrato de reentrega da
Evolution 2.3.7 não garante um timestamp confiável para essa decisão. Reentregas continuam
protegidas pela deduplicação durável e pelos limites de uso já existentes.

## Endpoints de saúde

| Endpoint | Semântica | Dependências | Falha |
| --- | --- | --- | --- |
| `/live` | processo HTTP ativo | nenhuma | não consulta PostgreSQL ou provedores |
| `/ready` | API apta a receber webhook | PostgreSQL e schema no head esperado | `503 {"status":"unavailable"}` |
| `/health` | alias compatível de readiness | igual a `/ready` | igual a `/ready` |

O sucesso permanece `200 {"status":"ok"}`. A readiness usa um `SELECT` curto em
`alembic_version`, com `READINESS_DATABASE_TIMEOUT_SECONDS=2`, e exige exatamente o head que o
código suporta. Não chama OpenAI nem Evolution e não inclui URL, SQL, versão encontrada ou texto
de exceção na resposta. A role runtime recebe somente `SELECT` nessa tabela; `INSERT`, `UPDATE`,
`DELETE`, `TRUNCATE` e DDL continuam negados. Toda nova migration deve atualizar o head esperado
junto com seus testes.

## Heartbeat durável do worker

Cada processo gera um UUID opaco distinto e mantém uma linha em `worker_heartbeats`. A tabela
contém somente `worker_id`, `started_at`, `last_seen_at`, enum `RUNNING/STOPPING/STOPPED`, e release
opaca opcional. Não contém hostname, IP, PID, telefone, JID, credencial ou conteúdo financeiro.
Todos os instantes são UTC e comparados com o relógio do PostgreSQL.

Defaults:

- `WORKER_HEARTBEAT_INTERVAL_SECONDS=15`;
- `WORKER_HEARTBEAT_STALE_SECONDS=60`, validado como pelo menos três intervalos;
- `WORKER_HEARTBEAT_DATABASE_TIMEOUT_SECONDS=3`, menor que o intervalo;
- `WORKER_HEARTBEAT_RETENTION_DAYS=7`;
- `WORKER_HEARTBEAT_ID_PATH=/tmp/oink-finai-worker-id`.

O worker grava no início, renova apenas no intervalo e usa uma transação curta independente por
escrita. No shutdown gracioso passa por `STOPPING` e termina em `STOPPED`; um crash não executa
essa transição e a linha fica stale. Falha do heartbeat gera somente código sanitizado e não
substitui a exceção principal. Ao iniciar, cada worker remove apenas heartbeats cujo
`last_seen_at` já excedeu a retenção, limitando crescimento sem uma rotina externa destrutiva.

O comando `python -m oink_finai.worker_healthcheck` lê o UUID opaco local e retorna sucesso somente
se a linha daquele worker estiver `RUNNING` e recente. Ausente, stale e `STOPPED` falham; nenhuma
leitura altera a linha e nenhum detalhe é impresso. O estado agregado pode ser consultado
internamente pelo serviço: basta um worker `RUNNING` recente para o conjunto ser saudável.

## Healthchecks e diagnóstico após reboot

O Compose de produção preserva `pg_isready` para PostgreSQL. A API usa o `urllib` da própria
imagem contra `/ready`; o worker usa o módulo Python acima, sem `curl` ou `wget`. Os checks não
contêm segredo e nenhuma porta adicional é publicada. O job `migrate` continua one-shot e API e
worker dependem de sua conclusão bem-sucedida.

Após reboot ou release, verifique `postgres`, o exit code de `migrate` e então a saúde dos dois
processos. Interprete os casos assim:

- PostgreSQL indisponível: `/live` pode responder, mas `/ready`, `/health` e o worker ficam
  unhealthy;
- migration atrás: API continua viva, porém readiness retorna `503`;
- migration falha: o Compose não inicia API/worker devido a
  `service_completed_successfully`;
- worker stale, ausente ou parado: somente o healthcheck do worker falha;
- API viva mas não pronta: não encaminhe webhooks até `/ready` voltar a `200`.

Em restart automático do daemon, `depends_on` não é uma barreira reaplicada. O procedimento de
release deve executar novamente `docker compose ... up -d`, conferir o job one-shot e os
healthchecks. Não diagnostique com `docker inspect` completo ou dump de ambiente, pois isso pode
expor segredos.
