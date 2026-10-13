# Bot Control

TensorFold Studio can guide multi-agent rooms from phone channels. Telegram polls from the desktop app. WhatsApp Cloud API events enter through a loopback-only, Meta-signed webhook route; the QR bridge remains a separate local connector.

## Commands

- `rooms` / `status` - list recent and active rooms.
- `enter room <id>` / `enter latest` - bind this chat to a room.
- Any normal message while inside a room - relay it as a user room event.
- `leave` - return to central bot control.
- `close room <id>` - abort a running room and mark it stopped.
- `remove room <id>` - delete a stopped/finished room record.
- `clean up old rooms` - remove finished/stopped/failed rooms.

Short room IDs are accepted as prefixes.

## Telegram

Settings live in `Settings -> Bot control`.

1. Create a Telegram bot and paste the bot token.
2. Add the allowed chat ID and each trusted sender's numeric Telegram user ID. Both must match, including in group chats.
3. Enable Telegram and background polling.
4. Use `Poll now` to smoke-test the path.

The token is stored through the app settings secret path and redacted before settings are sent to the renderer.

## WhatsApp

The Cloud API path requires an access token, phone number ID, webhook verify token, Meta App Secret, and explicit sender allowlist. When enabled, the app's loopback webhook listener exposes `/integrations/whatsapp`; publish that route only through a TLS reverse proxy or tunnel you control.

Subscription challenges must match the configured verify token. Every POST must carry Meta's `X-Hub-Signature-256`; TensorFold Studio verifies that HMAC against the raw request body before parsing it. The sender principal comes only from the signed `messages[].from` field, the signed phone-number ID must match the configured endpoint, and stable Meta message IDs are deduplicated. There is no renderer-callable or unsigned `bot:ingest` contract.
