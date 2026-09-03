# Oink FinAI

Fundação do backend de controle financeiro cuja interface principal será o WhatsApp. Esta etapa fornece API, persistência, cache, modelos e contratos; não conecta IA nem Evolution API.

## Arquitetura

O código fica em `src/oink_finai/`:

- `api/`: rotas FastAPI;
- `config/`: variáveis de ambiente;
- `database/`: sessão SQLAlchemy e modelos;
- `domain/`: enums e regras centrais;
- `schemas/`: contratos validados;
- `services/` e `repositories/`: casos de uso e persistência;
- `providers/whatsapp/`: contrato `WhatsAppProvider`, independente da Evolution API.

O contrato de áudio confirmado para a Evolution API 2.3.7 está documentado em
[`docs/evolution-api-2.3.7-media-contract.md`](docs/evolution-api-2.3.7-media-contract.md).

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
