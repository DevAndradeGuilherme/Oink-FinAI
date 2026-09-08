# Oink FinAI

Backend de controle financeiro cuja interface principal é o WhatsApp. O registro de gastos por
texto está disponível e a confirmação também é enviada como texto. Edição e remoção foram adiadas
para depois do MVP. Botões interativos não estão habilitados no engine Baileys atual da Evolution
API 2.3.7.

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

Para manter API e worker como servicos permanentes gerenciados pelo Compose, use:

```bash
docker compose up -d api worker
```

Use `docker compose run --rm <servico> <comando>` somente para comandos oneoff. Nunca use
`docker compose run` para manter API ou worker ativos; isso cria containers temporarios duplicados.

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

Texto, esclarecimentos e imagens usam a API OpenAI. Os modelos padrão de
`OPENAI_EXPENSE_MODEL` e `OPENAI_IMAGE_MODEL` são `gpt-4.1-mini`; áudio usa
`OPENAI_AUDIO_TRANSCRIPTION_MODEL=gpt-transcribe`. Configure `OPENAI_API_KEY` somente no ambiente.
Cada operação faz uma chamada por tentativa durável, com timeout externo, sem retry interno do SDK
e sem fallback de modelo.

As tentativas e os intervalos do processamento durável de gastos são configurados por
`EXPENSE_PROCESSING_MAX_ATTEMPTS`, `EXPENSE_RETRY_BASE_SECONDS` e
`EXPENSE_RETRY_MAX_SECONDS`.

Os valores históricos `GEMINI_*` de `error_code` permanecem somente como identificadores duráveis
compatíveis com registros e alertas existentes; eles não indicam uso do provider Gemini.

### Timing seguro do pipeline

A instrumentacao estruturada fica desabilitada por padrao. Para teste supervisionado, defina
`PIPELINE_TIMING_ENABLED=true` no ambiente da API e do worker e reinicie ambos. Para desabilitar,
defina `PIPELINE_TIMING_ENABLED=false` e reinicie os processos.

Eventos emitidos: `webhook_received`, `access_filter_completed`, `inbound_persisted`,
`webhook_completed`, `processing_claimed`, `queue_wait_completed`, `media_download_started`,
`media_download_completed`, `transcription_started`, `transcription_completed`,
`transcript_checkpoint_started`, `transcript_checkpoint_completed`, `interpretation_started`,
`interpretation_completed`, `image_download_started`, `image_download_completed`,
`image_analysis_started`, `image_analysis_completed`, `image_checkpoint_started`,
`image_checkpoint_completed`, `expense_persistence_started`, `expense_persistence_completed`,
`processing_completed`, `outbox_claimed`, `outbox_queue_wait_completed`,
`outbound_send_started`, `outbound_send_completed` e `outbound_accepted`.

Cada evento usa UUID interno de `ProcessedMessage` como `correlation_id`. Demais campos possiveis:
timestamp UTC ISO 8601, duracao, tentativa, etapa, tipo de origem, resultado, codigo de erro
sanitizado, status HTTP numerico, tamanho em bytes, MIME normalizado, duracao declarada do audio e
proxima tentativa. Nunca sao registrados telefone, JID, identificador externo, texto, transcript,
dados do gasto, prompt, resposta OpenAI, audio/base64, referencia ou URL de midia, segredos,
chaves, headers, respostas ou payloads brutos.

Duracoes internas ao processo usam relogio monotonico. Timestamps UTC correlacionam API e worker;
diferencas entre processos nao usam relogio monotonico. Eventos permitem calcular tempo do
webhook, espera em fila, download e transcricao por tentativa, checkpoint, interpretacao,
persistencia, processamento total, espera da outbox, envio, backoff e tempo interno entre
`webhook_received` e `outbound_accepted`. Esse tempo interno termina na aceitacao pela Evolution:
HTTP 201 representa aceitacao, nao entrega ao aparelho nem confirmacao visual. Tempo percebido pelo
usuario pode ser maior.

### Pipeline durável de imagens

`OpenAIImageAnalyzer` recebe somente bytes e metadados técnicos já validados, além de legenda
opcional não confiável. Retorna observações estruturadas e evidências literais do texto visível.
Webhook persiste somente referência opaca, MIME normalizado e legenda validada. Worker baixa e
valida mídia, executa análise isolada e confirma checkpoint JSONB em transação própria antes da
interpretação financeira. Após checkpoint, referência de download é removida; retries reutilizam
checkpoint sem novo download ou análise. Expense e confirmação textual usam pipeline e outbox
existentes.

Checkpoint contém exatamente versão, tipo de documento, flags financeiro/legível, confiança,
warnings e listas limitadas de candidatos de valor, data, estabelecimento e pagamento com suas
evidências mínimas. Não contém bytes, base64, URL, mediaKey, payload, headers, EXIF, resposta bruta,
legenda ou `visible_text` integral. Legenda fica em coluna limitada separada, como contexto não
visual. JSONB é validado novamente antes do uso.

O grounding preserva `visible_text` e evidências originais. Para comparar representações visuais
equivalentes, aplica somente composição Unicode NFC, remoção de variation selectors, normalização
de sequências de whitespace e comparação sem diferença de caixa para merchant e pagamento.
Dígitos, acentos e pontuação não são removidos; separadores monetários e limites de token continuam
significativos. Assim, `20` nunca fundamenta `120`, valores diferentes nunca são aproximados e a
legenda nunca participa da busca por evidência.

`amount_candidate.value` usa `Decimal` no domínio. O modelo recebe instrução para produzir decimal
canônico sem moeda ou milhar; o parser defensivo também aceita representações brasileiras
inequívocas com `R$`, vírgula decimal e ponto de milhar. `evidence` continua sendo transcrição
visual separada. Nenhum formato é arredondado, truncado ou limpo por remoção permissiva.
