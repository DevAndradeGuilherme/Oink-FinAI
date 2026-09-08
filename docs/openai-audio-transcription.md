# Transcrição de áudio OpenAI

O worker usa exclusivamente `OpenAIAudioTranscriber`, com `gpt-transcribe` por padrão. Não
há seletor de provider, retry interno nem fallback de modelo. Texto, esclarecimento e imagem
também usam a camada OpenAI, por adaptadores separados e contratos neutros.

## Configuração

| Variável | Padrão | Uso |
| --- | --- | --- |
| `OPENAI_API_KEY` | Ausente | Obrigatória no worker; omitida do repr das configurações. |
| `OPENAI_AUDIO_TRANSCRIPTION_MODEL` | `gpt-transcribe` | Modelo do endpoint de transcrição. |
| `OPENAI_AUDIO_TRANSCRIPTION_TIMEOUT_SECONDS` | `90` | Timeout HTTP e limite externo da operação. |
| `OPENAI_AUDIO_TRANSCRIPTION_LANGUAGE` | `pt` | Código de idioma com duas letras minúsculas. |

Os limites `MEDIA_MAX_BYTES` e `MEDIA_MAX_DURATION_SECONDS` permanecem ativos. Somente
`.env.example` documenta essas variáveis; credenciais reais não fazem parte do repositório.

## Contrato e execução

O adaptador usa `AsyncOpenAI` e `POST /v1/audio/transcriptions`. O áudio permanece em memória:
não há arquivo local, Files API, upload prévio nem prompt de interpretação. OGG, MP3,
M4A/MP4, WAV, FLAC e WebM recebem nome técnico e MIME coerentes; formatos não suportados são
rejeitados antes da chamada.

Cada tentativa durável válida faz exatamente uma chamada. O SDK usa `max_retries=0`, o cliente
HTTP não segue redirects e um timeout externo cobre preparação, requisição e resposta.
Cancelamentos continuam propagados. Clientes injetados pertencem ao chamador; `aclose()` fecha
somente o cliente criado pelo adaptador.

O retorno é validado como `AudioTranscription`, sem inventar idioma detectado. Timeout, conexão,
HTTP 408/429/5xx seguem a política transitória atual; erros de configuração, autenticação,
modelo, requisição ou resposta inválida são sanitizados e classificados sem preservar corpo,
headers ou exceção bruta. Logs do SDK e transporte são filtrados durante operações privadas.

## Validação isolada

Os testes usam chaves sintéticas e `httpx.MockTransport`; não chamam OpenAI nem Evolution.
Um smoke real posterior deve executar apenas `OpenAIAudioTranscriber.transcribe()` com áudio
explicitamente autorizado e segredo injetado pelo gerenciador apropriado. Confirme uma chamada
por tentativa, transcrição coerente e ausência de conteúdo sensível nos logs; não registre o
transcript. Nenhuma migration é necessária.
