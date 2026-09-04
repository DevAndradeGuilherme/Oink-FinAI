# Transcrição de áudio OpenAI preparada para revisão

O worker seleciona o transcritor por `AUDIO_TRANSCRIPTION_PROVIDER`. O padrão continua sendo
`gemini`, com o modelo, timeout e limites de mídia anteriores. A presença de `OPENAI_API_KEY`
não seleciona OpenAI. Não há fallback entre provedores. Texto, interpretação financeira e
análise de imagens continuam usando suas implementações existentes.

## Configuração

| Variável | Padrão | Uso |
| --- | --- | --- |
| `AUDIO_TRANSCRIPTION_PROVIDER` | `gemini` | Aceita `gemini` ou `openai`. |
| `OPENAI_API_KEY` | Ausente | Obrigatória somente ao construir o transcritor OpenAI; omitida do repr das configurações. |
| `OPENAI_AUDIO_TRANSCRIPTION_MODEL` | `gpt-4o-mini-transcribe` | Modelo do endpoint de transcrição. |
| `OPENAI_AUDIO_TRANSCRIPTION_TIMEOUT_SECONDS` | `90` | Prazo externo da operação completa, além do timeout HTTP. |
| `OPENAI_AUDIO_TRANSCRIPTION_LANGUAGE` | `pt` | Código de idioma com duas letras minúsculas. |

Os limites `MEDIA_MAX_BYTES` e `MEDIA_MAX_DURATION_SECONDS` continuam sendo aplicados.
Nenhum valor do `.env` real foi necessário para implementar ou testar esta camada.

## Contrato e execução

O [contrato oficial de criação de transcrição](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create)
documenta `POST /v1/audio/transcriptions`, upload multipart de arquivo, idioma ISO-639-1 e
`response_format=json` para `gpt-4o-mini-transcribe`. O adaptador usa `AsyncOpenAI` do pacote
oficial `openai` e envia uma tupla de nome técnico fixo, bytes e MIME. Não cria arquivo local,
não usa Files API, não faz upload prévio e não envia prompt de interpretação.

O áudio permanece em memória. OGG usa `audio.ogg` e `audio/ogg`, inclusive quando o MIME de
entrada é `audio/opus` com contêiner OGG. MP3, M4A/MP4, WAV, FLAC e WebM também possuem nomes
técnicos próprios. AAC bruto e Opus sem contêiner OGG são rejeitados antes da requisição:
não há conversão ou tentativa de disfarçar formatos não documentados pelo endpoint.
As validações de download e formatos admitidos pela Evolution permanecem iguais.

Cada tentativa válida realiza uma chamada ao endpoint. O SDK tem `max_retries=0`; não há loop
de retry ou fallback no adaptador. O cliente HTTP criado pelo adaptador não segue redirects
e usa o transporte padrão sem retries. Clientes SDK injetados recebem uma cópia com retries
do SDK desabilitados e timeout configurado; seu transporte HTTP deve igualmente estar sem
retries. Clientes injetados com redirects habilitados são rejeitados antes de qualquer chamada.
O chamador continua responsável por fechar clientes injetados.
`aclose()` fecha o cliente criado pelo adaptador e impede novas transcrições nessa instância.

O timeout externo cobre preparação, requisição e leitura da resposta. Cancelamento do worker
continua propagando para a política durável existente. O JSON é convertido para
`AudioTranscription`, com texto não vazio e limite de tamanho existente. O idioma enviado é
uma indicação de entrada; por isso `detected_language` fica `None`, sem inventar detecção.
O transcript é omitido do repr do resultado, mantendo a mesma serialização e validação.

Timeout (incluindo HTTP 408/504), transporte, 429 e demais 5xx são transitórios. HTTP 400,
401, 403, 404, demais respostas HTTP não previstas, configuração inválida e resposta inválida
são terminais. Os códigos `TranscriptionErrorCode` existentes alimentam a política durável.
O adaptador não preserva resposta, headers ou exceção original em seus erros sanitizados.
Logs internos do SDK, HTTPX e HTTPCore são filtrados no contexto da operação, inclusive em
DEBUG; operações concorrentes fora desse contexto mantêm seus logs.

## Validação e habilitação posterior

Os testes usam chaves sintéticas, o SDK oficial com `httpx.MockTransport` e bancos SQLite de
teste. A coleta de testes desabilita a leitura automática de `.env`. A integração simulada
exercita retry durável, checkpoint, retomada sem retranscrição e confirmação textual única.
Nenhum teste exige chamadas reais a OpenAI, Gemini ou Evolution.

Após aprovar o review, execute um smoke separado do worker e webhook reais, usando um áudio
sintético ou explicitamente autorizado. Nesse processo isolado, forneça a chave pelo
gerenciador de segredos, sem imprimi-la, e configure:

```dotenv
AUDIO_TRANSCRIPTION_PROVIDER=openai
OPENAI_AUDIO_TRANSCRIPTION_MODEL=gpt-4o-mini-transcribe
OPENAI_AUDIO_TRANSCRIPTION_TIMEOUT_SECONDS=90
OPENAI_AUDIO_TRANSCRIPTION_LANGUAGE=pt
```

O smoke deve chamar somente `OpenAIAudioTranscriber.transcribe()` e fechar a instância em
`finally`, sem conectar banco, Evolution ou outbox. Confirme uma requisição por tentativa,
transcrição coerente e ausência de conteúdo sensível nos logs. Não registre o transcript.

Somente após o review e esse smoke, forneça as variáveis acima ao ambiente do worker de
destino, reconstrua a imagem para incluir a dependência e reinicie esse worker pelo processo
normal de implantação. Preserve as credenciais Gemini necessárias à interpretação e às
imagens. Para voltar ao transcritor antigo, defina `AUDIO_TRANSCRIPTION_PROVIDER=gemini` e
reinicie o worker. Nenhuma migration ou alteração de banco é necessária.
