# Runtime Docker de produção

Esta configuração cobre a fundação do runtime, identidades PostgreSQL separadas, limites duráveis
de uso, proteção HTTP, readiness e heartbeat. Ela ainda não inclui Named Tunnel, backup
automatizado, alertas externos, retenção/LGPD ou CI/CD. O procedimento de banco está em
[`postgres-least-privilege.md`](postgres-least-privilege.md).

## Arquivos e pré-requisitos

Produção usa **somente** `docker-compose.prod.yml`; não combine esse arquivo com
`docker-compose.yml`, pois o segundo contém bind mounts, portas e `--reload` exclusivos do
desenvolvimento. Os arquivos indicados por `OINK_RUNTIME_ENV_FILE` e
`OINK_MIGRATION_ENV_FILE` devem existir apenas no host, fora do repositório, e ter permissão
restritiva. O primeiro contém a configuração da API/worker e somente a URL de runtime; o segundo
contém somente `MIGRATION_DATABASE_URL`. Eles não são incorporados à imagem.

Defina no ambiente do operador:

- `OINK_IMAGE_REPOSITORY` e `OINK_IMAGE_DIGEST`: repositório e digest `sha256:` da imagem da
  aplicação;
- `POSTGRES_IMAGE_REPOSITORY` e `POSTGRES_IMAGE_DIGEST`: repositório e digest `sha256:` da
  imagem PostgreSQL;
- `OINK_RUNTIME_ENV_FILE`: arquivo externo exclusivo de API e worker;
- `OINK_MIGRATION_ENV_FILE`: arquivo externo exclusivo do migrator;
- `POSTGRES_DB` e `POSTGRES_USER`: banco da aplicação e identidade bootstrap;
- `POSTGRES_BOOTSTRAP_PASSWORD_FILE`: arquivo externo com o segredo bootstrap, nunca uma senha
  literal no Compose.

Também são obrigatórios no arquivo de runtime `DATABASE_URL`, `OPENAI_API_KEY`,
`EVOLUTION_BASE_URL`, `EVOLUTION_API_KEY`, `EVOLUTION_INSTANCE`,
`EVOLUTION_WEBHOOK_SECRET` e `WHATSAPP_ALLOWED_NUMBERS`. Use valores fortes e não reutilize
os exemplos. `EVOLUTION_BASE_URL` é uma comunicação externa e precisa usar HTTPS. A URL
interna `DATABASE_URL` usa a role runtime na rede Docker privada. O arquivo de migration contém
somente `MIGRATION_DATABASE_URL`, usando a role migrator. O Alembic em produção falha se essa
variável estiver ausente e nunca usa `DATABASE_URL` como fallback.

Renderize e revise a configuração efetiva antes do deploy:

```console
docker compose --env-file /etc/oink-finai/compose.env -f docker-compose.prod.yml config
```

Depois, inicie exatamente uma réplica de cada processo:

```console
docker compose --env-file /etc/oink-finai/compose.env -f docker-compose.prod.yml up -d
```

Não use `--scale`. O Compose fixa `scale: 1` e `deploy.replicas: 1`; os serviços `api`,
`worker` e `migrate` compartilham a mesma referência por digest. `migrate` executa
`alembic upgrade head`, termina e precisa concluir com
sucesso antes que API ou worker iniciem. Compose v2 aceita
`condition: service_completed_successfully`; `docker compose up` não libera os dependentes
quando o job falha. Ao recriar explicitamente `migrate`, o job roda novamente e o Alembic
mantém a operação idempotente no head atual.

Esse comportamento foi validado com Docker Compose v5.1.4: um `migrate` sintético com exit
23 fez `docker compose up` retornar 1 e deixou a API apenas em estado `created`, sem iniciá-la.
Versões anteriores do Compose que não reconheçam `service_completed_successfully` não são
suportadas por este arquivo; valide a versão durante o deploy.

## Rede, saúde e encerramento

PostgreSQL participa apenas da rede interna `database` e não publica portas. A API publica
somente `127.0.0.1:8000`, para consumo futuro por um Named Tunnel executado no host. O worker
não publica portas. API e worker também usam `edge` para as integrações externas. Redis não
é iniciado e não participa da arquitetura atual porque não existe consumidor no código.

O PostgreSQL é liberado por `pg_isready`; a API é verificada por `/ready` e o worker pelo comando
`python -m oink_finai.worker_healthcheck`. Ambos usam apenas ferramentas presentes na imagem. A
documentação `/docs`, `/redoc` e `/openapi.json` fica desabilitada em produção. A semântica de
`/live`, `/ready`, `/health`, os thresholds do heartbeat e o diagnóstico estão em
[`http-health-and-worker-heartbeat.md`](http-health-and-worker-heartbeat.md).
API e worker recebem sinais diretamente, usam um init mínimo e têm 330 segundos para
encerrar. Esse prazo cobre o máximo validado de uma operação corrente (300 segundos) com
30 segundos de margem. Após SIGTERM/SIGINT, o worker termina o item já reivindicado e não
reivindica outro; um encerramento forçado continua coberto pela recuperação de claims stale.

Após reinício da VPS, a política `unless-stopped` religa PostgreSQL, API e worker; o job
`migrate` concluído não é reiniciado. As políticas automáticas do daemon Docker não
reavaliam `depends_on`, portanto o procedimento de release continua sendo executar
`docker compose up -d` com o arquivo de produção e conferir saúde e o exit code do job.
Se o banco ainda não estiver disponível, o worker falha e é reiniciado; a API não deve ser
considerada pronta para tráfego apenas porque o processo está em execução.

Settings é validado no início do processo. Para rotacionar chaves, substitua atomicamente o
arquivo externo de ambiente e recrie API e worker; valores não são recarregados em runtime.
A rotação do segredo de webhook exige coordenação com a Evolution e uma janela operacional
planejada, pois esta fase não implementa aceitação simultânea de segredo antigo e novo.

API, worker e migration usam JSON estruturado em produção. PostgreSQL e os três serviços possuem
rotação Docker de 10 MiB por arquivo e cinco arquivos por container. Para diagnóstico somente
leitura, execute `python -m oink_finai.operational_check` com o ambiente runtime. Campos, eventos,
exit codes e thresholds estão em
[`structured-logging-and-operations.md`](structured-logging-and-operations.md).

## Operação do job de migration

Para inspecionar o resultado do job one-shot:

```console
docker compose --env-file /etc/oink-finai/compose.env -f docker-compose.prod.yml ps -a migrate
docker compose --env-file /etc/oink-finai/compose.env -f docker-compose.prod.yml logs migrate
```

Um exit code diferente de zero mantém API e worker bloqueados. Corrija a causa e recrie
somente o job antes de subir os dependentes; não execute Alembic dentro do comando da API.

Os limites de abuso e custo são carregados do mesmo arquivo runtime exclusivo de API/worker.
O job `migrate` não precisa de chaves de provedores para criar o ledger da revisão
`20260909_0013`. A matriz de limites, estados de reserva, consultas sanitizadas e resposta a abuso
está em [durable-usage-control.md](durable-usage-control.md).
