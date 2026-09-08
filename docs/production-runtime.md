# Runtime Docker de produção

Esta configuração cobre somente a fundação do runtime. Ela ainda não inclui Named Tunnel,
backup automatizado, papéis PostgreSQL separados, quotas, rate limiting, heartbeat, alertas,
retenção/LGPD ou CI/CD.

## Arquivos e pré-requisitos

Produção usa **somente** `docker-compose.prod.yml`; não combine esse arquivo com
`docker-compose.yml`, pois o segundo contém bind mounts, portas e `--reload` exclusivos do
desenvolvimento. O arquivo indicado por `OINK_ENV_FILE` deve existir apenas no host, fora do
repositório, ter permissão restritiva e conter todas as variáveis da aplicação. Ele não é
incorporado à imagem.

Defina no ambiente do operador:

- `OINK_IMAGE_REPOSITORY` e `OINK_IMAGE_DIGEST`: repositório e digest `sha256:` da imagem da
  aplicação;
- `POSTGRES_IMAGE_REPOSITORY` e `POSTGRES_IMAGE_DIGEST`: repositório e digest `sha256:` da
  imagem PostgreSQL;
- `OINK_ENV_FILE`: caminho absoluto do arquivo de ambiente de produção;
- `POSTGRES_DB`, `POSTGRES_USER` e `POSTGRES_PASSWORD`: bootstrap do PostgreSQL;
- `MIGRATION_DATABASE_URL`: URL usada exclusivamente pelo job de migration. Nesta fase ela
  pode apontar para a mesma credencial da aplicação; uma role privilegiada separada poderá
  substituí-la sem mudar o Compose.

Também são obrigatórios no arquivo de ambiente `DATABASE_URL`, `OPENAI_API_KEY`,
`EVOLUTION_BASE_URL`, `EVOLUTION_API_KEY`, `EVOLUTION_INSTANCE`,
`EVOLUTION_WEBHOOK_SECRET` e `WHATSAPP_ALLOWED_NUMBERS`. Use valores fortes e não reutilize
os exemplos. `EVOLUTION_BASE_URL` é uma comunicação externa e precisa usar HTTPS. A URL
interna `DATABASE_URL` pode usar a rede Docker privada sem TLS nesta fase.

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

O PostgreSQL é liberado por `pg_isready`; a API é verificada pelo endpoint compatível
`/health`. A documentação `/docs`, `/redoc` e `/openapi.json` fica desabilitada em produção.
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

## Operação do job de migration

Para inspecionar o resultado do job one-shot:

```console
docker compose --env-file /etc/oink-finai/compose.env -f docker-compose.prod.yml ps -a migrate
docker compose --env-file /etc/oink-finai/compose.env -f docker-compose.prod.yml logs migrate
```

Um exit code diferente de zero mantém API e worker bloqueados. Corrija a causa e recrie
somente o job antes de subir os dependentes; não execute Alembic dentro do comando da API.
