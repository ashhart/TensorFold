# TensorFold Connect

TensorFold Connect signs a person in to their organisation's TensorFold Starlight and gives desktop tools a local endpoint
that speaks the OpenAI and Anthropic APIs. Tools never see the person's Starlight credentials; each one gets its own app key,
and Starlight sees every request as the signed-in person, labelled with the app that sent it.

Connect runs on macOS, Windows and Linux as one binary, `tensorfold-connect`, with a command line and a menu-bar or
system-tray app.

## Sign in

```bash
tensorfold-connect login --server https://starlight.example.com
```

Connect reads Starlight's discovery document from `/.well-known/tensorfold-starlight`, opens the system browser at Starlight's sign-in
page, and waits on a one-time listener at `http://127.0.0.1:<random port>/callback`. The exchange is OAuth 2.0
authorization code with PKCE (S256) and a state check; the listener accepts only the redirect that carries its own state and
closes after five minutes. Starlight then runs the organisation's single sign-on.

On a machine without a browser, sign in with a code instead:

```bash
tensorfold-connect login --device
```

Connect prints a short code and an address to open on any device (RFC 8628 device grant).

| Command | What it does |
| --- | --- |
| `tensorfold-connect status` | Server, organisation, signed-in user, project, credential store, local endpoint, managed policy |
| `tensorfold-connect projects` | Projects you belong to; `--use SLUG` makes one the default for requests |
| `tensorfold-connect models` | Models you may use in the current project; `--project SLUG` for another |
| `tensorfold-connect logout` | Revokes this device's sign-in at Starlight and forgets it locally |

Connect stores its tokens in the operating system's credential store: the macOS login keychain, Windows Credential
Manager, or the Secret Service (GNOME Keyring, KWallet) on Linux. Where none is available, such as a headless Linux
server, it uses a file readable only by the user. `TENSORFOLD_CONNECT_STORE=file` selects the file explicitly.

## The local endpoint

```bash
tensorfold-connect bridge
```

The local endpoint listens on `127.0.0.1:7171` (`--port` to change it). The tray app runs it too. Give each tool its own
key:

```bash
tensorfold-connect apps add notebook
tensorfold-connect apps list
tensorfold-connect apps remove notebook
```

A key is printed once; Connect keeps only its SHA-256. Removing an app, or issuing it a new key, stops the old key at once,
including in an endpoint that is already running.

Every request passes three checks before anything is forwarded:

1. The `Host` header must name `127.0.0.1`, `localhost` or `::1`, which defeats DNS rebinding from web pages.
2. A request with an `Origin` header is refused, so browser pages cannot call the endpoint. Origins listed in the
   profile's `allowed_origins` are the only exception.
3. The request must carry an app key, as `Authorization: Bearer` or `x-api-key`. Starlight tokens are not accepted.

Connect then removes the client's own credentials, cookies and forwarding headers, adds the signed-in person's access token
and `X-Starlight-Client: connect/<app>`, adds `X-Starlight-Project` when the tool did not send one, and streams the reply back
chunk by chunk. If Starlight rejects the access token, Connect refreshes it and retries once when the request body was small
enough to keep (32 MiB).

| Method | Path |
| --- | --- |
| `GET` | `/v1/models`, `/v1/models/{id}` |
| `POST` | `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/responses`, `/v1/messages`, `/v1/messages/count_tokens` |
| `GET`, `DELETE` | `/v1/responses/{id}` |

Anything else answers 404. `GET /healthz` answers `{"status":"ok"}` without a key, for health checks.

## Set up tools

`tensorfold-connect env TOOL` issues a fresh key for the tool's app and prints its settings. Running it again for the same
tool replaces that tool's key. Add `--model ID` with an id from `tensorfold-connect models`, and `--shell powershell` on
Windows (the default there).

Claude Code and the Anthropic SDKs:

```bash
eval "$(tensorfold-connect env claude-code --model qwen3.8-27b)"
```

which sets `ANTHROPIC_BASE_URL=http://127.0.0.1:7171`, `ANTHROPIC_AUTH_TOKEN` to the app key, and `ANTHROPIC_MODEL`.

The OpenAI SDKs and tools that read the same variables:

```bash
eval "$(tensorfold-connect env openai)"
```

which sets `OPENAI_BASE_URL=http://127.0.0.1:7171/v1` and `OPENAI_API_KEY`.

Codex:

```bash
tensorfold-connect env codex --model qwen3.8-27b
```

prints `export TENSORFOLD_CONNECT_KEY=...` and a `config.toml` block: `model_provider = "tensorfold"` with a
`[model_providers.tensorfold]` table whose `base_url` is the local endpoint, `env_key` is `TENSORFOLD_CONNECT_KEY` and
`wire_api` is `responses`. Top-level keys in `~/.codex/config.toml` must come before its first table.

## The tray app

```bash
tensorfold-connect tray
```

The menu shows who is signed in, the organisation and project, the models available, actions that copy a tool's setup to
the clipboard, a switch for the local endpoint, the console, and sign in or out. It refreshes every minute and starts the
local endpoint unless policy turns it off. Add the command to your login items to start it automatically.

The tray needs cgo on macOS. Headless builds use `go build -tags notray ./cmd/tensorfold-connect`, which keeps every other
command.

## Finding Starlight on the network

```bash
tensorfold-connect discover
```

Starlight servers that advertise themselves answer multicast DNS queries for `_tf-starlight._tcp` with three TXT records: `v=1`,
`url=` (the Starlight address) and `id=` (the first eight characters of the installation id). Anyone on a network can
advertise anything, so a discovered address is only a suggestion: confirm it with your IT team, then sign in with
`--server`, where TLS and the discovery document are checked as for any other address. Advertising is off in Starlight unless
an administrator turns it on, and many corporate networks do not pass multicast between subnets, so managed configuration
remains the dependable way to set the address.

## Managed configuration

Administrators can set Connect's options for every user. A managed value overrides the user's own setting.

| Key | Type | Effect |
| --- | --- | --- |
| `ServerURL` | string | Starlight address. Users cannot sign in elsewhere unless `AllowManualServer` is true |
| `AllowManualServer` | boolean | Lets users choose another server even though `ServerURL` is set |
| `CACertificatePEM` | string | PEM of the CA that issued Starlight's certificate, trusted in addition to the system store |
| `CAFile` | string | Path to such a PEM file; `CACertificatePEM` wins when both are set |
| `BridgePort` | integer | Local endpoint port, 1024 to 65535 |
| `DisableBridge` | boolean | Turns the local endpoint off; `env`, `bridge` and the tray's switch refuse to run it |

Unknown keys are ignored and listed by `tensorfold-connect status`, so a misspelt key is visible.

| Platform | Location | Precedence |
| --- | --- | --- |
| macOS | `/Library/Managed Preferences/dev.tensorfold.connect.plist`, then `/Library/Managed Preferences/<user>/dev.tensorfold.connect.plist` | The per-user file wins key by key |
| Windows | `HKCU\SOFTWARE\Policies\TensorFold\Connect`, then `HKLM\SOFTWARE\Policies\TensorFold\Connect` | The machine policy wins key by key |
| Linux | `/etc/tensorfold/connect.json` | One file |

Templates are in `deploy/mdm`:

| File | Use |
| --- | --- |
| `macos/dev.tensorfold.connect.mobileconfig` | Configuration profile with the Connect settings |
| `macos/dev.tensorfold.connect-with-root-ca.mobileconfig` | The same, plus a root certificate payload for a private Starlight CA |
| `macos/dev.tensorfold.connect.plist` | Preference file for tools that take a domain and a plist |
| `windows/TensorFoldConnect.admx`, `windows/en-US/TensorFoldConnect.adml` | Administrative template for Group Policy and Intune |
| `windows/tensorfold-connect.reg` | The same settings as registry values |
| `linux/connect.json` | The Linux policy file |

Replace every value marked `REPLACE`, and run `uuidgen` for new `PayloadUUID` values when you make another profile from
a template.

### macOS with Jamf Pro

Either upload `dev.tensorfold.connect.mobileconfig` as a computer configuration profile, or create a configuration profile
with an Application & Custom Settings payload, set the preference domain to `dev.tensorfold.connect`, and upload
`dev.tensorfold.connect.plist`. Scope it to the computers that run Connect. If Starlight's certificate comes from a private
CA, deploy the CA with a Certificate payload, or set `CACertificatePEM` so only Connect trusts it.

### macOS with Microsoft Intune

Create a macOS configuration profile from the Preference file template with the preference domain
`dev.tensorfold.connect` and upload `dev.tensorfold.connect.plist`, or upload `dev.tensorfold.connect.mobileconfig` as a
Custom profile. Deploy a private CA with a Trusted certificate profile, or through `CACertificatePEM`.

### Windows with Group Policy or Intune

For Group Policy, copy `TensorFoldConnect.admx` to the `PolicyDefinitions` folder (the domain's central store or
`%SystemRoot%\PolicyDefinitions`) and `TensorFoldConnect.adml` to its `en-US` subfolder. The settings appear under
TensorFold > Connect for both computers and users. In Intune, import the ADMX and ADML files as imported administrative
templates and build a profile from them, or deploy the registry values in `tensorfold-connect.reg` with your own tooling.

### Linux

Install `connect.json` as `/etc/tensorfold/connect.json`, readable by every user and writable only by root, with your
configuration management tool.

## Security

- **Sign-in.** PKCE with S256 on a loopback redirect, an unguessable state, and a five-minute window. The device grant
  honours the server's polling interval and `slow_down`.
- **Tokens.** Access tokens are short-lived. Refresh tokens rotate on every use and Starlight revokes a token family when a
  used refresh token comes back, so Connect refreshes under a lock shared by every Connect process for the user and
  rereads the store first: the tray, the endpoint and the command line never race each other into a revocation.
  `invalid_grant` signs Connect out.
- **Local endpoint.** Other accounts on the same machine can reach `127.0.0.1`, which is why every request needs an app
  key. Keys are hashed at rest, scoped to one app, and revoked by removing the app. Browsers are refused through the
  `Origin` and `Host` checks.
- **Transport.** Connect talks only to the configured Starlight, over HTTPS; plain HTTP is accepted only for a Starlight on the
  same machine, for testing. A managed CA is trusted in addition to the system store, never instead of it. `HTTPS_PROXY`
  and `NO_PROXY` are honoured.
- **What Starlight learns.** Starlight records this device as `TensorFold Connect on <hostname>` in the person's list of
  signed-in devices, and sees which app sent each request.

## What Connect needs from Starlight

| Endpoint | Expectation |
| --- | --- |
| `GET /.well-known/tensorfold-starlight` | The discovery document; every endpoint on the issuer's origin |
| `GET /oauth/authorize` | Public client `tensorfold-connect`; loopback redirect URIs on any port (RFC 8252); `code_challenge_method=S256`; optional `device_name` |
| `POST /oauth/token` | Grants `authorization_code` (with `code_verifier`), `refresh_token` (rotating) and `urn:ietf:params:oauth:grant-type:device_code`; OAuth errors as JSON with HTTP 400 |
| `POST /oauth/device/code` | Device authorization for `tensorfold-connect` with `scope` and optional `device_name` |
| `POST /oauth/revoke` | RFC 7009 revocation of a refresh token |
| `GET /oauth/userinfo` | The `Me` document: user, organisation and projects, with `default` on one project |
| `GET /v1/models` | OpenAI list format, filtered for the caller and the `X-Starlight-Project` header |
| Inference routes | `Authorization: Bearer` access tokens, `X-Starlight-Client` and `X-Starlight-Project` headers; 401 when a token is expired or revoked |

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `TensorFold Connect is signed out` | Run `tensorfold-connect login`; Starlight may have revoked the sign-in or the account |
| `x509: certificate signed by unknown authority` | Starlight uses a private CA: deploy it to the system store, or set `CACertificatePEM` or `CAFile` |
| `local endpoint: address already in use` | Another program holds the port: stop it, or set another port with `--port` or `BridgePort` |
| A keyring prompt appears on Linux | Unlock the keyring, or use `TENSORFOLD_CONNECT_STORE=file` on machines without a desktop session |
| A tool gets `invalid_api_key` | Its app key was replaced or removed: run `tensorfold-connect env TOOL` again |

Connect keeps its own files in the user's configuration directory (`~/Library/Application Support/tensorfold-connect` on
macOS, `%AppData%\tensorfold-connect` on Windows, `~/.config/tensorfold-connect` on Linux), or in `TENSORFOLD_CONNECT_HOME`
when it is set.
