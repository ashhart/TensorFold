# Installing Starlight

Starlight needs Postgres 15 or newer, a master key, and network paths between its three listeners and the machines that use
them. Choose Docker Compose for a single host, Helm for Kubernetes, or the systemd unit for a Linux server.

## Network paths

| Listener | Default | Who connects | TLS |
| --- | --- | --- | --- |
| Public (`listen.public`) | `:8443` | Browsers, Connect, SDKs, harnesses, SCIM from the IdP | Starlight's certificate, or terminated at a proxy |
| Node (`listen.nodes`) | `:9443` | Starlight Node agents | Always Starlight's internal CA; must reach Starlight unmodified (TCP passthrough) |
| Metrics (`listen.metrics`) | `127.0.0.1:9090` | Prometheus; load-balancer health checks on `/healthz` and `/readyz` | Plain HTTP; protect with `metrics.bearer_token_file` |

Starlight also opens connections to Postgres, to each node's inference listener (`starlight-node run --listen`, default `:7443`),
to identity providers over HTTPS, and to frontier providers over HTTPS unless the installation is offline. Nodes connect
to the node listener and, for Hugging Face artifacts, to the Hub or an internal mirror.

## Configuration

Starlight reads `starlight.yaml` (path from `--config` or `STARLIGHT_CONFIG`) over built-in defaults, then applies environment
overrides. Unknown keys are refused, so typos fail at start-up.

| Key | Environment | Default | Meaning |
| --- | --- | --- | --- |
| `public_url` | `STARLIGHT_PUBLIC_URL` | none | The origin users reach; `https://` unless it is a loopback address |
| `node_url` | `STARLIGHT_NODE_URL` | none | The `https://` origin of the node listener that nodes dial; used in enrolment commands |
| `roles` | `STARLIGHT_ROLES` | all | Any of `gateway`, `control`, `console`, `scim` |
| `trusted_proxies` | `STARLIGHT_TRUSTED_PROXIES` | none | CIDRs whose `X-Forwarded-For` is believed |
| `listen.public`, `listen.nodes`, `listen.metrics` | `STARLIGHT_LISTEN_PUBLIC`, `_NODES`, `_METRICS` | see above | Listen addresses |
| `tls.cert_file`, `tls.key_file` | `STARLIGHT_TLS_CERT_FILE`, `_KEY_FILE` | none | Serve the public listener with TLS 1.2+ |
| `database.url` or `database.url_file` | `STARLIGHT_DATABASE_URL`, `_URL_FILE` | none | Postgres connection string |
| `database.max_conns` | none | 32 | Pool size per instance |
| `database.min_conns` | none | 4 | Connections each instance keeps open |
| `keys.master_key_file` | `STARLIGHT_MASTER_KEY_FILE`, or the key in `STARLIGHT_MASTER_KEY` | none | Base64 32-byte master key |
| `keys.previous_key_files` | none | none | Older master keys, kept only to unwrap during rotation |
| `identity.ca_file` | `STARLIGHT_IDP_CA_FILE` | none | PEM bundle added to the system roots for identity providers |
| `sessions.*` | none | see [security.md](security.md) | Console sessions, Connect tokens, credential cache |
| `gateway.*` | none | see [admission.md](admission.md) | Body limit, queue timeouts, request timeout, default output tokens |
| `mdns.advertise`, `mdns.instance` | none | off | Advertise `public_url` on the LAN for Connect's `discover` |
| `metrics.bearer_token_file` | none | none | Require `Authorization: Bearer` on `/metrics` |
| `log.level`, `log.format` | `STARLIGHT_LOG_LEVEL` | `info`, `json` | Structured logs on stderr |

`deploy/compose/starlight.yaml` is a complete example.

## Docker Compose

```bash
cd deploy/compose
./bootstrap.sh
docker compose build
docker compose run --rm starlight init --org "Example Corp"
docker compose up -d
```

`bootstrap.sh` creates `secrets/` and `.env` with mode 0600 and refuses to overwrite them. The `starlight` service runs read-only,
with every capability dropped, as your user id so it can read those files. It starts with `serve --migrate`, so upgrades
apply migrations before serving.

Profiles add optional services:

| Profile | Services | Notes |
| --- | --- | --- |
| `monitoring` | Prometheus on `127.0.0.1:9090`, Grafana on `127.0.0.1:3000` | The Starlight dashboard and alert rules are provisioned; Grafana's admin password is in `.env` |
| `idp` | Keycloak 26 in development mode, realm `example` | Shares the `starlight` container's network so the issuer `http://localhost:8080/realms/example` works for both the browser and Starlight |

The Keycloak profile is for trying SSO only. Its realm defines client `starlight` with PKCE, a `groups` claim, back-channel
logout for a connection with the slug `keycloak`, groups `starlight-admins` and `ml-engineers`, and one test user. Use
connection slug `keycloak` in Starlight's setup wizard so the redirect URIs match.

The compose file publishes the node listener on port 9443. When nodes are on other machines, make sure that port is
reachable and set `STARLIGHT_NODE_URL` in `.env` to the name they use. Do not put an HTTP proxy in front of it: a node
authenticates with its client certificate, which a TLS-terminating proxy would remove.

For anything beyond a trial, give the public listener TLS (below) and keep the master key in a secret manager.

## Kubernetes with Helm

The chart in `deploy/helm/starlight` runs Starlight as a Deployment with a public Service, a node Service of type `LoadBalancer`
(plain TCP), a metrics Service, a PodDisruptionBudget, a NetworkPolicy and optional HPA, Ingress and ServiceMonitor.
It does not run Postgres; use a managed database or an operator such as CloudNativePG.

Create the secret first:

```bash
kubectl create namespace starlight
kubectl --namespace starlight create secret generic starlight-secrets \
  --from-literal=database-url='postgres://starlight:REPLACE@db.example.internal:5432/starlight?sslmode=verify-full' \
  --from-literal=master-key="$(openssl rand -base64 32)" \
  --from-literal=metrics-token="$(openssl rand -hex 24)"
```

Then install:

```bash
helm install starlight deploy/helm/starlight --namespace starlight \
  --set publicURL=https://ai.example.com --set nodeURL=https://starlight-nodes.example.com:9443 \
  --set secrets.metricsTokenKey=metrics-token --set ingress.enabled=true --set ingress.host=ai.example.com
kubectl --namespace starlight logs job/starlight-init
```

A pre-install hook runs `starlight init` once and prints the setup link to that job's log; a pre-install and pre-upgrade hook
runs `starlight migrate`. Set `initialise.enabled=false` when you initialise by hand. Probes use `/healthz` and `/readyz` on
the metrics port, which `/readyz` answers only when the schema matches the image.

Point the node Service's DNS name at `nodeURL`. Cloud load balancers need TCP mode; set `service.nodes.annotations`
accordingly and restrict sources with `service.nodes.loadBalancerSourceRanges`. Every replica serves the node listener;
long-polls work against any of them.

Extra configuration goes in `config:` and is merged over the keys the chart sets. For TLS directly on the public
listener, put a `kubernetes.io/tls` secret name in `tls.secretName`. For a private IdP CA, put the PEM in a ConfigMap and
set `identityCA.configMapName`.

## Linux with systemd

```bash
sudo install -m 0755 starlight /usr/local/bin/starlight
sudo install -d -m 0755 /etc/starlight && sudo install -d -m 0700 /etc/starlight/credentials
openssl rand -base64 32 | sudo tee /etc/starlight/credentials/master-key >/dev/null
printf '%s' 'postgres://starlight:REPLACE@db.example.internal:5432/starlight?sslmode=verify-full' | sudo tee /etc/starlight/credentials/database-url >/dev/null
sudo chmod 0600 /etc/starlight/credentials/*
sudo install -m 0644 deploy/compose/starlight.yaml /etc/starlight/starlight.yaml
sudo install -m 0644 deploy/systemd/starlight.service /etc/systemd/system/starlight.service
```

Add `public_url` and `node_url` to `/etc/starlight/starlight.yaml`, initialise once, then start the service:

```bash
sudo STARLIGHT_CONFIG=/etc/starlight/starlight.yaml STARLIGHT_MASTER_KEY_FILE=/etc/starlight/credentials/master-key \
  STARLIGHT_DATABASE_URL_FILE=/etc/starlight/credentials/database-url starlight init --org "Example Corp"
sudo systemctl daemon-reload && sudo systemctl enable --now starlight
```

The unit runs Starlight as a dynamic user with no capabilities, a read-only file system and a narrow system-call filter. The
master key and database URL reach it as systemd credentials, so they never sit in its environment or in a readable
path. `starlight init --write-master-key PATH` creates a new key file instead of `openssl` when you prefer.

## After the first start

Run the doctor on any instance. It checks the configuration, database version and schema, clock skew against Postgres,
the master and data keys, setup state and owners, the workload CA and permit keys, the controller lease, nodes and
replicas, identity-provider reachability and provider DNS, and exits non-zero on any failure:

```bash
starlight doctor
```

## First owner and single sign-on

`starlight init` prints `<public_url>/auth/setup?code=...`, valid once for 24 hours. The wizard creates the first identity
provider connection from a preset (Okta, Microsoft Entra ID, Google Workspace, Auth0, OneLogin, PingOne, PingFederate,
JumpCloud, Keycloak, Authentik, Zitadel, AWS IAM Identity Center, ADFS, Shibboleth, Duo, GitLab, generic OIDC or SAML),
shows the URLs to register at the IdP, and makes the first person to finish signing in the owner. If the link expires
while no owner exists, `starlight admin setup-code` prints a new one. Per-IdP steps are in [sso/](sso/README.md); SCIM
provisioning is in [scim.md](scim.md).

## TLS and proxies

- **Direct:** set `tls.cert_file` and `tls.key_file`. Starlight serves TLS 1.2 or newer on the public listener.
- **At a proxy or ingress:** leave `tls` unset, keep `public_url` as the external `https://` origin, and list the proxy
  networks in `trusted_proxies` so client addresses in audit records, rate limits and key CIDR allowlists are real.
  Starlight takes the rightmost untrusted address from `X-Forwarded-For`. Entries may carry a port (`203.0.113.7:51544`,
  `[2001:db8::1]:443`), as Azure Application Gateway sends them. When an entry cannot be read, Starlight records no client
  address: key CIDR allowlists refuse the request, and such requests share one rate-limit budget.
- **Node listener:** always TLS from Starlight's internal CA, with client certificates. Forward it as TCP only.
- **Private IdP CAs:** set `identity.ca_file` to a PEM bundle; it is added to the system roots for discovery, token and
  metadata requests only.

## Nodes

Create an enrolment command on any instance:

```bash
starlight nodes token --pool default --label rack=a --max-class confidential
```

It prints `starlight-node enroll --server <node_url> --token sle_... --ca-sha256 <fingerprint>`. Tokens are single-use by
default (`--uses`), expire after 24 hours (`--ttl`), and can approve nodes automatically (`--auto-approve`); otherwise
approve with `starlight nodes approve NAME` or in the console.

- **macOS (Mac Studio, Mac Pro):** create the `_tensorfold` role account, run
  `sudo starlight-node service install --user _tensorfold -- --engine /path/to/tensorfold --prevent-sleep`, enrol as that
  account, and start the daemon. It runs as a LaunchDaemon under the unprivileged account, starts at boot without a
  login and keeps the machine awake.
- **Linux with NVIDIA GPUs:** run `starlight-node` under systemd (`deploy/node/starlight-node.service`) or as the CUDA container
  in `deploy/node/container/Dockerfile` with the NVIDIA Container Toolkit.
- **Windows:** `deploy/node/install-windows.ps1` copies the agent, enrols it, installs the service and opens its port.

[nodes.md](nodes.md) covers each platform, sleep, artifacts and troubleshooting.

## Rolling out Connect

Distribute `tensorfold-connect` with your MDM and set `ServerURL` by policy so people never type an address:

- macOS: `deploy/mdm/macos/dev.tensorfold.connect.mobileconfig` (Jamf Pro or Intune custom profile); the variant with a root
  CA payload trusts a private CA.
- Windows: `deploy/mdm/windows/TensorFoldConnect.admx` and `en-US/TensorFoldConnect.adml` for Group Policy or Intune.
- Linux: `/etc/tensorfold/connect.json` from `deploy/mdm/linux/connect.json`.

Managed values override and lock the user's settings. [connect.md](connect.md) has the details.
