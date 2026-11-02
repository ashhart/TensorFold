#!/usr/bin/env bash
# Generates the compose stack's secrets and .env once; it never overwrites existing ones.
set -euo pipefail

cd "$(dirname "$0")"
umask 077

if [ -e secrets/master-key ] || [ -e .env ]; then
  echo "secrets/ or .env already exists; delete them deliberately before generating new ones" >&2
  exit 1
fi
command -v openssl >/dev/null || { echo "openssl is required" >&2; exit 1; }

public_url="${STARLIGHT_PUBLIC_URL:-http://localhost:8443}"
node_url="${STARLIGHT_NODE_URL:-https://$(hostname):9443}"
mkdir -p secrets

openssl rand -base64 32 > secrets/master-key
db_password="$(openssl rand -hex 24)"
printf '%s' "$db_password" > secrets/postgres-password
printf 'postgres://starlight:%s@postgres:5432/starlight?sslmode=disable' "$db_password" > secrets/database-url

client_secret="$(openssl rand -hex 24)"
user_password="$(openssl rand -base64 18 | tr -d '/+=' | cut -c1-20)"
sed -e "s|__CLIENT_SECRET__|$client_secret|" -e "s|__USER_PASSWORD__|$user_password|" -e "s|__PUBLIC_URL__|$public_url|g" \
  keycloak/realm.json.template > secrets/keycloak-realm.json

cat > .env <<ENV
STARLIGHT_PUBLIC_URL=$public_url
STARLIGHT_NODE_URL=$node_url
STARLIGHT_UID=$(id -u)
STARLIGHT_GID=$(id -g)
GRAFANA_ADMIN_PASSWORD=$(openssl rand -hex 16)
KEYCLOAK_ADMIN_PASSWORD=$(openssl rand -hex 16)
ENV

chmod 600 secrets/* .env
echo "Wrote secrets/ and .env (mode 0600). Back up secrets/master-key: without it Starlight cannot read its sealed secrets."
echo
echo "Development Keycloak (profile idp), realm 'example':"
echo "  issuer         http://localhost:8080/realms/example"
echo "  client id      starlight"
echo "  client secret  $client_secret"
echo "  test user      ada@example.com / $user_password"
