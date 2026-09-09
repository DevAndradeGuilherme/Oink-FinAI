# Controle durável de custo e abuso

O controle de admissão usa exclusivamente PostgreSQL. Redis não participa desta decisão. Toda
janela é calculada em UTC, e a aplicação rejeita qualquer outro valor para
`USAGE_WINDOW_TIMEZONE`.

## Limites padrão

| Limite | Padrão |
|---|---:|
| mensagens por usuário/minuto | 10 |
| mensagens por usuário/dia | 150 |
| mensagens globais/dia | 600 |
| operações OpenAI por usuário/dia | 60 |
| operações OpenAI globais/dia | 240 |
| operações OpenAI simultâneas | 2 |
| texto por usuário/global ao dia | 40 / 160 |
| consulta por usuário/global ao dia | 20 / 80 |
| imagem por usuário/global ao dia | 12 / 48 |
| áudio por usuário/global ao dia | 10 / 40 |
| saída máxima de texto | 600 tokens |
| saída máxima de consulta | 600 tokens |
| saída máxima de imagem | 500 tokens |

Esses valores acomodam aproximadamente seis usuários em uso normal e deixam margem sem permitir
crescimento ilimitado. Os tetos totais continuam valendo mesmo que a soma dos tetos por tipo seja
maior. Ajuste primeiro os limites por tipo e só depois os totais. Em produção, zero, números
negativos, limites por usuário maiores que os globais e valores acima dos limites de segurança da
configuração impedem a inicialização.

Os limites existentes de bytes, duração de áudio, dimensões/pixels de imagem e tamanho de mensagem
continuam ativos. Áudio não recebe `max_output_tokens` porque a operação de transcrição usada não
oferece esse parâmetro.

## Admissão e contabilização

O webhook valida, nesta ordem: segredo, instância, allowlist, identidade externa e quota. A
identidade é inserida sob a constraint de deduplicação antes de contabilizar; uma reentrega do
mesmo evento não consome novamente. Uma decisão PostgreSQL protegida por advisory lock
transacional verifica simultaneamente minuto, dia, usuário e global. Isso também serializa dois
webhooks concorrentes com IDs sempre novos.

Ao exceder a quota de mensagens, o `ProcessedMessage` termina como `FAILED` com código sanitizado:
o worker não baixa mídia, não chama OpenAI e não cria `Expense`. Uma única orientação por usuário e
janela entra na outbox com chave idempotente; reentregas ou novos IDs na mesma janela não criam um
loop de orientações. Nenhum estado `WAITING` é criado. Uma nova mensagem é admitida automaticamente
quando a próxima janela UTC começa.

Antes de cada operação OpenAI, o worker cria uma reserva única por
`(processed_message_id, operation, durable_attempt)`. O mesmo lock transacional protege os tetos
global, por usuário, por tipo e de concorrência. O adapter marca a reserva como transmitida
imediatamente antes de entregar a chamada ao SDK, cujo retry interno permanece zero.

Estados do ledger:

- `RESERVED`: ocupa quota diária e uma vaga concorrente;
- `COMPLETED`: chamada concluída; preserva a quota diária e libera a vaga concorrente;
- `AMBIGUOUS`: envio pode ter ocorrido, portanto continua contando de forma conservadora;
- `RELEASED`: foi comprovado que a chamada não chegou ao ponto de transmissão e deixa de contar.

Reservas antigas sem marca de transmissão são liberadas. Reservas antigas já transmitidas tornam-se
ambíguas. Assim, crash depois da reserva, timeout, cancelamento e crash entre a resposta e o
checkpoint não permitem gasto invisível nem bloqueiam a concorrência para sempre.

Quota OpenAI diária adia uma tentativa legítima para `00:00 UTC` da próxima janela, enquanto houver
tentativas duráveis; ao esgotá-las, a mensagem segue o encerramento terminal existente. Falta de
vaga concorrente usa o atraso configurado por `OPENAI_CONCURRENCY_RETRY_SECONDS`. Um checkpoint já
concluído pula a chamada e não cria reserva. Uma nova tentativa durável pode reservar outra
operação. O mesmo checkpoint/tentativa nunca reserva duas vezes.

## Dados persistidos e consultas seguras

`usage_ledger` contém somente UUID interno do usuário e da mensagem processada, operação, nome
validado do modelo, número da tentativa, início das janelas UTC, estado, timestamps técnicos,
tokens numéricos de entrada/saída e segundos de áudio quando disponíveis. Não existem colunas para
telefone, JID, texto, transcript, imagem, prompt, resposta, URL, headers ou chaves. Respostas brutas
da OpenAI nunca são persistidas.

Exemplos de consulta que não expõem conteúdo:

```sql
SELECT window_day_start, operation, state, count(*) AS operations,
       sum(input_tokens) AS input_tokens, sum(output_tokens) AS output_tokens,
       sum(audio_seconds) AS audio_seconds
FROM usage_ledger
GROUP BY window_day_start, operation, state
ORDER BY window_day_start DESC, operation, state;
```

```sql
SELECT user_id, operation, count(*) AS operations
FROM usage_ledger
WHERE window_day_start = date_trunc('day', now() AT TIME ZONE 'UTC') AT TIME ZONE 'UTC'
  AND state IN ('RESERVED', 'COMPLETED', 'AMBIGUOUS')
GROUP BY user_id, operation;
```

Use a conexão de backup/read-only ou uma sessão administrativa supervisionada para auditoria. Não
faça join com `users` quando a finalidade for somente consumo.

## Orçamento e resposta a abuso

As métricas locais são unidades técnicas confiáveis, não estimativa de cobrança. Preços variam por
modelo e provedor; por isso nenhuma tabela de preços está hardcoded. Configure também o teto
monetário definitivo e alertas no projeto da OpenAI. Esse controle externo é a última barreira se
uma credencial for usada fora da aplicação.

Para bloquear um usuário comprometido, remova seu número de `WHATSAPP_ALLOWED_NUMBERS` no arquivo de
ambiente runtime mantido fora do repositório e recrie API/worker pelo procedimento operacional. O
access filter rejeita o evento antes da deduplicação e da quota. Não coloque o número em commits,
logs ou comandos compartilhados; edite o arquivo secreto com um editor que não grave histórico de
shell.

## Migration e least privilege

A revisão `20260909_0013` é linear sobre `20260908_0012`, cria os enums e o ledger sem alterar
registros históricos. Mensagens anteriores não são inseridas retroativamente. O downgrade é
forward-only para preservar auditoria.

A migration deve rodar somente como `migrator`. Os default privileges da fase 5B concedem à role
runtime apenas `SELECT`, `INSERT`, `UPDATE` e `DELETE` na nova tabela, e à role backup apenas
`SELECT`; nenhuma role de aplicação recebe DDL. Valide após a migration com
`scripts/postgres/validate.sh` no profile administrativo descartável documentado em
`postgres-least-privilege.md`.
