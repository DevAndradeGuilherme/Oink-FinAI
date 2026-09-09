# PostgreSQL com least privilege

Esta operação pressupõe um cluster PostgreSQL dedicado ao Oink FinAI. Os scripts validam todos
os identificadores, falham com variável/placeholder ausente, usam arquivos de senha e nunca
recebem senha em argumento. Não habilite `trust`: a imagem inicia com SCRAM para conexões locais e
de rede. O baseline de init remove também `CONNECT`/`TEMPORARY` de `PUBLIC` nos bancos de
manutenção. A validação falha se qualquer role da aplicação puder conectar a outro banco.

## Matriz de identidades

| Identidade | Uso | Propriedade/permissões | Nunca recebe |
| --- | --- | --- | --- |
| bootstrap/admin | Instalação inicial e emergência | Administração do cluster; não é usada em operação normal | API, worker, migrate ou backup |
| migrator | Alembic one-shot | Dona do banco, schema e objetos; `LOGIN`, sem atributos administrativos | Segredos OpenAI, Evolution ou webhook |
| runtime | API e worker | `CONNECT`, `USAGE`, DML nas tabelas e `USAGE, SELECT` nas sequences | URL de migration, bootstrap ou backup |
| backup | Ferramentas futuras | `CONNECT`, `USAGE`, `SELECT`; transação read-only por padrão | DML, DDL ou outras credenciais |

As três roles da aplicação são `NOSUPERUSER`, `NOCREATEDB`, `NOCREATEROLE`, `NOINHERIT`,
`NOREPLICATION` e `NOBYPASSRLS`, não têm memberships e não podem assumir migrator ou bootstrap.
`SELECT FOR UPDATE` funciona para runtime porque ela possui `SELECT` e `UPDATE`. `TRUNCATE`,
`CREATE`, `ALTER` e `DROP` não são concedidos.

## Preparação de segredos sem histórico de shell

Crie cada arquivo fora do repositório em um diretório acessível somente ao operador. O arquivo
deve conter uma única linha, com no mínimo 20 caracteres. Este padrão não põe o valor no histórico
nem o mostra no terminal:

```console
umask 077
secret_file="$(mktemp)"
read -r -s -p 'Novo segredo: ' secret_value
printf '\n'
printf '%s\n' "$secret_value" > "$secret_file"
unset secret_value
```

Mova o arquivo para o diretório seguro com nome operacional. Não use senha em URL digitada na
linha de comando. Edite os arquivos de ambiente com editor seguro e permissões `0600`; não rode
`echo`, `env`, `set`, `docker inspect` completo nem `docker compose config` com arquivos que
contenham valores reais.

O arquivo de variáveis do Compose contém apenas referências de imagem, nomes de roles e caminhos:

```dotenv
OINK_IMAGE_REPOSITORY=registry.example.invalid/oink-finai
OINK_IMAGE_DIGEST=sha256:replace-me
POSTGRES_IMAGE_REPOSITORY=postgres
POSTGRES_IMAGE_DIGEST=sha256:replace-me
OINK_RUNTIME_ENV_FILE=/secure/path/runtime.env
OINK_MIGRATION_ENV_FILE=/secure/path/migration.env
POSTGRES_BOOTSTRAP_PASSWORD_FILE=/secure/path/bootstrap.password
POSTGRES_DB=application_database
POSTGRES_USER=bootstrap_role
APP_SCHEMA=public
MIGRATOR_ROLE=migrator_role
RUNTIME_ROLE=runtime_role
BACKUP_ROLE=backup_role
MIGRATOR_PASSWORD_FILE=/secure/path/migrator.password
RUNTIME_PASSWORD_FILE=/secure/path/runtime.password
BACKUP_PASSWORD_FILE=/secure/path/backup.password
POSTGRES_INVENTORY_DIRECTORY=/secure/path/inventory
```

Os repositórios e digests acima são placeholders, não referências de deploy. `runtime.env` contém
`DATABASE_URL` e as configurações normais da aplicação, mas não `MIGRATION_DATABASE_URL`.
`migration.env` contém somente `MIGRATION_DATABASE_URL`. A senha de backup não entra em Settings
nem em nenhum container normal.

## Instalação nova

1. Inicie somente o PostgreSQL com `docker-compose.prod.yml`.
2. Defina `PROVISION_MODE=new` no arquivo não secreto do Compose.
3. Execute manualmente o profile administrativo, combinando os arquivos nesta ordem:

```console
docker compose --env-file /secure/path/compose.env \
  -f docker-compose.prod.yml -f docker-compose.postgres-admin.yml \
  --profile postgres-admin run --rm postgres-admin
```

O serviço não tem porta, restart ou acesso à rede externa; monta os quatro secrets somente durante
essa execução. O script cria roles ausentes com senha nula, define cada senha por prompt interno
do `psql`, transfere banco/schema para migrator, revoga `PUBLIC`, aplica grants atuais e default
privileges. Repetir não duplica roles ou grants e não substitui senhas já existentes.

Depois execute o `migrate` one-shot e suba API/worker pelo Compose normal. O migrator cria todas as
revisions 0001→0012 e objetos futuros; os default privileges entregam automaticamente DML ao
runtime e leitura ao backup para tabelas, além dos privilégios mínimos em sequences.

## Adaptação de banco existente

Faça backup e janela de manutenção antes. Informe `LEGACY_OWNER` com o proprietário esperado e
defina exatamente:

```dotenv
PROVISION_MODE=existing
ALLOW_EXISTING_DATABASE_ADAPTATION=I_UNDERSTAND_THIS_DATABASE_WILL_BE_MODIFIED
```

Então execute o mesmo comando do profile administrativo. O script interrompe antes de qualquer
mudança se banco, schema, relation, índice, tipo ou rotina tiver owner fora de `LEGACY_OWNER`,
migrator ou `pg_database_owner`. Ele grava inventários CSV antes/depois no diretório indicado e
altera apenas banco, schema e objetos desse schema. Não usa `REASSIGN OWNED`.

O preflight também exige um cluster dedicado: nenhuma regra ativa de `pg_hba.conf` pode usar
`trust`, e `PUBLIC` não pode conectar a outro banco permitido. Em instalação nova o baseline em
`docker-entrypoint-initdb.d` satisfaz isso. Em volume antigo, corrija e recarregue o `pg_hba.conf`
e revogue `CONNECT`/`TEMPORARY` de `PUBLIC` nos bancos de manutenção em uma janela administrativa
separada; o script deliberadamente não altera objetos fora do banco da aplicação.

Tabelas/partições, sequences, views, materialized views, foreign tables, ENUMs, domains, ranges,
composites e rotinas são tratados explicitamente. Índices e array types acompanham seus objetos
base. Dados, IDs, constraints, índices, ENUMs e `alembic_version` não são recriados. Após a
adaptação, compare inventários, contagens e digests da janela e execute novamente para comprovar
idempotência antes de liberar a aplicação.

## Rotação individual

Crie um arquivo novo com o procedimento sem eco acima, sem substituir ainda o arquivo atual.
Monte-o somente no container administrativo e chame `rotate-password.sh`; `ROTATE_ROLE` aceita
bootstrap, migrator, runtime ou backup:

```console
docker compose --env-file /secure/path/compose.env \
  -f docker-compose.prod.yml -f docker-compose.postgres-admin.yml \
  --profile postgres-admin run --rm --entrypoint sh \
  --volume "$STAGED_PASSWORD_FILE:/run/secrets/new_password:ro" \
  --env ROTATE_ROLE="$TARGET_ROLE" \
  --env NEW_PASSWORD_FILE=/run/secrets/new_password \
  postgres-admin /opt/oink/postgres/rotate-password.sh
```

Após sucesso:

- runtime: atualize `DATABASE_URL` no arquivo runtime e recrie API/worker;
- migrator: atualize somente `MIGRATION_DATABASE_URL` no arquivo migration;
- backup: atualize somente o secret da futura ferramenta de backup;
- bootstrap: substitua atomicamente o arquivo bootstrap e recrie somente PostgreSQL na janela
  planejada para garantir que o mount aponte ao novo inode.

Para revogar acesso emergencial sem remover a identidade, rotacione bootstrap para segredo novo
guardado offline e remova o arquivo staged. Não entregue o arquivo à stack de aplicação.

## Validação

Execute `validate-privileges.sh` pelo mesmo container administrativo ou use o profile com uma
entrypoint sobrescrita. Ele confere atributos, memberships, ownership, conexão exclusiva ao banco
da aplicação e ausência de `CREATE` para runtime/backup. A validação destrutiva completa exige um
marcador inequívoco e uma imagem local já construída:

```console
export OINK_DESTRUCTIVE_POSTGRES_TEST_MARKER=I_UNDERSTAND_ONLY_A_DISPOSABLE_POSTGRES_WILL_BE_DESTROYED
export OINK_TEST_APP_IMAGE=local-image-name
python scripts/postgres/run-temporary-tests.py
unset OINK_DESTRUCTIVE_POSTGRES_TEST_MARKER OINK_TEST_APP_IMAGE
```

O runner cria nomes/segredos exclusivos, rede interna, volume e containers temporários. Ele cobre
instalação nova, migrations, API/worker, DML/locking, negações DDL/admin, default privileges,
`pg_dump` + `pg_restore --list`, upgrade com ownership legado, digests/contagens/IDs e repetição
idempotente. Dump e todos os recursos temporários são removidos em `finally`. Nunca forneça uma URL
de banco real ao runner; ele não aceita URL externa.
