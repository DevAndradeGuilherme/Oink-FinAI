# Contrato de mídia da Evolution API 2.3.7

Contrato verificado na tag oficial `2.3.7`, commit
`cd800f2976e1e5b682fbf86a01ee4d85ae61f370`.

## Webhook

O evento emitido é `messages.upsert`. O envelope contém `event`, `instance` e `data`;
`data` é o `messageRaw` preparado pela Evolution e contém, entre outros campos, `key`,
`message`, `messageType` e `messageTimestamp`. Para áudio sem wrapper, o conteúdo está em
`data.message.audioMessage`. O objeto protobuf/Baileys contém `mimetype`, `seconds` e `ptt`,
além de dados de transporte criptografado que esta aplicação não retém.

A versão 2.3.7 declara e remove, para download, os wrappers `ephemeralMessage`,
`documentWithCaptionMessage`, `viewOnceMessage` e `viewOnceMessageV2`, cada um com o próximo
conteúdo em `.message`. A normalização local reconhece os mesmos wrappers.

## Recuperação

- Rota: `POST /chat/getBase64FromMediaMessage/{instanceName}`.
- Autenticação: header `apikey` (chave global ou token da instância).
- Corpo: `{"message": WebMessageInfo, "convertToMp4": false}`.
- Dados mínimos adotados: `message.key` com `id`, `remoteJid` e `fromMe`. No código 2.3.7,
  quando `message.message` não existe, o serviço chama `getMessage(message.key, true)` e
  recupera a mensagem armazenada. Assim, o Oink não retém URL, `directPath` ou `mediaKey`.
- Sucesso: HTTP 201, com objeto que inclui `mediaType`, `fileName`, `caption`, `size`,
  `mimetype` e `base64`; `buffer` é nulo na chamada HTTP.
- Falhas da aplicação: 400 para falha de processamento/entrada, 401 para `apikey` ausente
  ou inválida e 404 para instância inexistente. O formato comum de erro contém `status`,
  `error` e `message`. Intermediários também podem produzir 403, 429 e 5xx; o cliente os
  converte em erros locais sanitizados.

O próprio método 2.3.7 tenta novamente internamente após uma primeira falha de download.
O provider Oink faz exatamente uma chamada HTTP: não repete timeout ou resultado ambíguo.
Retries futuros devem ficar no worker durável.

## Fontes oficiais inspecionadas

- `src/api/types/wa.types.ts`: tipos de mídia e wrappers.
- `src/api/integrations/channel/whatsapp/whatsapp.baileys.service.ts`:
  `prepareMessage` e `getBase64FromMediaMessage`.
- `src/api/dto/chat.dto.ts`: DTO da requisição.
- `src/api/routes/chat.router.ts`: rota, método e status de sucesso.
- `src/api/guards/auth.guard.ts` e `instance.guard.ts`: autenticação e instância.
- `src/api/integrations/event/webhook/webhook.controller.ts`: envelope do webhook.
