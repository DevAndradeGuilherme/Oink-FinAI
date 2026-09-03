# Contrato de mídia da Evolution API 2.3.7

Contrato verificado na tag oficial `2.3.7`, commit
`cd800f2976e1e5b682fbf86a01ee4d85ae61f370`.

O webhook `messages.upsert` contém `instance` e `data`. A mensagem está em
`data.message.audioMessage`, com `mimetype`, `seconds` e `ptt`. A versão declara os wrappers
`ephemeralMessage`, `documentWithCaptionMessage`, `viewOnceMessage` e `viewOnceMessageV2`,
cada qual contendo a mensagem seguinte em `.message`.

## Recuperação

- `POST /chat/getBase64FromMediaMessage/{instanceName}`;
- autenticação pelo header `apikey`;
- corpo `{"message": WebMessageInfo, "convertToMp4": false}`;
- resposta HTTP 201 com `mediaType`, `fileName`, `caption`, `size`, `mimetype` e `base64`.

Quando `message.message` está ausente, o código 2.3.7 executa
`getMessage(message.key, true)`. Por isso o Oink envia somente `message.key`, formada por
`id`, `remoteJid` e `fromMe`, sem reter URL, `directPath`, `mediaKey` ou payload bruto.

Os guards e serviços da versão produzem 400 para entrada/processamento, 401 para `apikey`
ausente ou inválida e 404 para instância inexistente. O cliente também classifica 403, 429,
falhas de transporte e 5xx sem incorporar resposta ou conteúdo nas exceções.

Embora o serviço Evolution possua fallback interno próprio, o Oink faz uma única chamada ao
endpoint e não repete timeout ambíguo. Retry durável de mídia fica para uma fase posterior.

Arquivos oficiais inspecionados: `src/api/types/wa.types.ts`,
`src/api/integrations/channel/whatsapp/whatsapp.baileys.service.ts`,
`src/api/dto/chat.dto.ts`, `src/api/routes/chat.router.ts`, `src/api/guards/auth.guard.ts`,
`src/api/guards/instance.guard.ts` e
`src/api/integrations/event/webhook/webhook.controller.ts`.

## Retenção para processamento durável

O webhook persiste somente o `id` externo, o `remoteJid` original exigido pela recuperação,
`fromMe=false`, MIME, duração e indicação PTT. Essa referência existe exclusivamente para o
worker recuperar a mídia depois da resposta do webhook. Payload bruto, `mediaKey`, bytes,
base64, URLs, `remoteJidAlt` e `participant` não são persistidos.

O `remoteJid` é removido após o transcript ser confirmado no banco, em falhas terminais e no
esgotamento de tentativas. Ele permanece apenas durante retries duráveis de download ou
transcrição. O transcript, limitado a 10.000 caracteres, ocupa `accepted_text`; o checkpoint
`transcribed_at` impede novo download ou nova transcrição após uma retomada.
