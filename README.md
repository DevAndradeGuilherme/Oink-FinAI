# Oink FinAI

Backend de controle financeiro cuja interface principal será o WhatsApp. A integração inicial com
Evolution API recebe webhooks de texto com idempotência e permite o envio de mensagens; ainda não
cria gastos nem conecta IA.

## Arquitetura

O código fica em `src/oink_finai/`:

- `api/`: rotas FastAPI;
- `config/`: variáveis de ambiente;
- `database/`: sessão SQLAlchemy e modelos;
- `domain/`: enums e regras centrais;
- `schemas/`: contratos validados;
- `services/` e `repositories/`: casos de uso e persistência;
- `providers/whatsapp/`: contrato `WhatsAppProvider`, independente da Evolution API.

Migrações ficam em `migrations/`; testes, em `tests/`.

## Execução com Docker

Requisitos: Docker e Docker Compose.

```bash
cp .env.example .env
docker compose up --build
```

No Windows PowerShell, use `Copy-Item .env.example .env`. Troque os valores `change-me` no `.env`. A API executa as migrations ao iniciar e fica disponível em `http://localhost:8000`. Verifique:

```bash
curl http://localhost:8000/health
```

Resposta esperada: `{"status":"ok"}`. Encerre com `docker compose down`. Use `docker compose down -v` somente para apagar também os dados locais do PostgreSQL.

Configure `EVOLUTION_BASE_URL`, `EVOLUTION_API_KEY`, `EVOLUTION_INSTANCE` e
`EVOLUTION_WEBHOOK_SECRET` no ambiente. Na Evolution API 2.3.7, o webhook da instância deve enviar o
cabeçalho customizado `X-Evolution-Webhook-Secret` com o mesmo segredo. O endpoint local é
`POST /api/v1/webhooks/evolution`.

O autoteste temporário pelo WhatsApp pessoal permanece desabilitado por padrão. Para ativá-lo,
configure `WHATSAPP_ACCESS_MODE=allowlist`, inclua o próprio número em
`WHATSAPP_ALLOWED_NUMBERS` e `WHATSAPP_SELF_TEST_NUMBER`, e defina
`WHATSAPP_SELF_TEST_ENABLED=true`. Somente mensagens enviadas na conversa com o próprio número e
iniciadas pelo `WHATSAPP_SELF_TEST_PREFIX` (padrão: `!oink`) serão processadas.

## Desenvolvimento local

Python 3.12 é obrigatório.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
```

Com PostgreSQL configurado no `.env`, use `alembic upgrade head` para aplicar migrations e `alembic downgrade -1` para reverter uma revisão.

## Configuração e segurança

Configurações são lidas por Pydantic Settings. Nunca versione `.env`, tokens, chaves da Evolution API ou senhas. `.env.example` contém somente valores locais ilustrativos. Dinheiro usa `Decimal`/`NUMERIC(14,2)`; exclusões de gastos devem preencher `deleted_at`, nunca remover a linha.
