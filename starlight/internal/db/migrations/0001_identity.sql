CREATE TABLE settings (
    id boolean PRIMARY KEY DEFAULT true CHECK (id),
    installation_id uuid NOT NULL,
    trust_domain text NOT NULL,
    org_name text NOT NULL DEFAULT 'TensorFold Starlight',
    offline boolean NOT NULL DEFAULT false,
    policy jsonb NOT NULL DEFAULT '{}'::jsonb,
    branding jsonb NOT NULL DEFAULT '{}'::jsonb,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE idp_connections (
    id uuid PRIMARY KEY,
    slug text NOT NULL UNIQUE CHECK (slug ~ '^[a-z0-9][a-z0-9-]{1,62}$'),
    name text NOT NULL,
    protocol text NOT NULL CHECK (protocol IN ('oidc', 'saml')),
    preset text NOT NULL,
    config jsonb NOT NULL DEFAULT '{}'::jsonb,
    secrets bytea,
    domains text[] NOT NULL DEFAULT '{}',
    enabled boolean NOT NULL DEFAULT true,
    jit boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE users (
    id uuid PRIMARY KEY,
    email text NOT NULL,
    email_lower text GENERATED ALWAYS AS (lower(email)) STORED,
    display_name text NOT NULL DEFAULT '',
    status text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'suspended', 'deactivated')),
    source text NOT NULL CHECK (source IN ('jit', 'scim', 'recovery', 'api')),
    scim_connection_id uuid REFERENCES idp_connections (id) ON DELETE SET NULL,
    scim_user_name text,
    external_id text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    last_login_at timestamptz,
    deactivated_at timestamptz
);
CREATE UNIQUE INDEX users_email_live ON users (email_lower) WHERE status <> 'deactivated';
CREATE UNIQUE INDEX users_scim_name ON users (scim_connection_id, lower(scim_user_name)) WHERE scim_user_name IS NOT NULL;

CREATE TABLE identities (
    id uuid PRIMARY KEY,
    user_id uuid NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    connection_id uuid NOT NULL REFERENCES idp_connections (id) ON DELETE CASCADE,
    subject text NOT NULL,
    email text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now(),
    last_login_at timestamptz,
    UNIQUE (connection_id, subject)
);
CREATE INDEX identities_user ON identities (user_id);

CREATE TABLE groups (
    id uuid PRIMARY KEY,
    connection_id uuid REFERENCES idp_connections (id) ON DELETE CASCADE,
    source text NOT NULL CHECK (source IN ('claim', 'scim', 'local')),
    external_id text NOT NULL,
    display_name text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE NULLS NOT DISTINCT (connection_id, source, external_id)
);

CREATE TABLE group_members (
    group_id uuid NOT NULL REFERENCES groups (id) ON DELETE CASCADE,
    user_id uuid NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (group_id, user_id)
);
CREATE INDEX group_members_user ON group_members (user_id);

CREATE TABLE scim_tokens (
    id uuid PRIMARY KEY,
    connection_id uuid NOT NULL REFERENCES idp_connections (id) ON DELETE CASCADE,
    token_hash bytea NOT NULL UNIQUE,
    label text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz,
    last_used_at timestamptz,
    revoked_at timestamptz
);

CREATE TABLE sessions (
    id uuid PRIMARY KEY,
    token_hash bytea NOT NULL UNIQUE,
    user_id uuid NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    connection_id uuid REFERENCES idp_connections (id) ON DELETE SET NULL,
    idp_subject text NOT NULL DEFAULT '',
    idp_session_id text NOT NULL DEFAULT '',
    auth_time timestamptz NOT NULL,
    amr text[] NOT NULL DEFAULT '{}',
    csrf_key bytea NOT NULL,
    ip text NOT NULL DEFAULT '',
    user_agent text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now(),
    last_seen_at timestamptz NOT NULL DEFAULT now(),
    idle_expires_at timestamptz NOT NULL,
    expires_at timestamptz NOT NULL,
    revoked_at timestamptz,
    revoked_reason text NOT NULL DEFAULT ''
);
CREATE INDEX sessions_user ON sessions (user_id) WHERE revoked_at IS NULL;
CREATE INDEX sessions_idp ON sessions (connection_id, idp_session_id) WHERE revoked_at IS NULL;

CREATE TABLE auth_requests (
    id uuid PRIMARY KEY,
    state_hash bytea NOT NULL UNIQUE,
    connection_id uuid NOT NULL REFERENCES idp_connections (id) ON DELETE CASCADE,
    binding_hash bytea NOT NULL,
    nonce text NOT NULL DEFAULT '',
    pkce_verifier bytea,
    saml_request_id text NOT NULL DEFAULT '',
    return_to text NOT NULL DEFAULT '/console',
    max_age_seconds int,
    created_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz NOT NULL,
    used_at timestamptz
);

CREATE TABLE oauth_codes (
    code_hash bytea PRIMARY KEY,
    client_id text NOT NULL,
    user_id uuid NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    session_id uuid REFERENCES sessions (id) ON DELETE CASCADE,
    redirect_uri text NOT NULL,
    code_challenge text NOT NULL,
    scope text NOT NULL,
    device_name text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz NOT NULL,
    used_at timestamptz
);

CREATE TABLE device_grants (
    id uuid PRIMARY KEY,
    device_code_hash bytea NOT NULL UNIQUE,
    user_code text NOT NULL UNIQUE,
    client_id text NOT NULL,
    scope text NOT NULL,
    device_name text NOT NULL DEFAULT '',
    status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'denied', 'consumed')),
    user_id uuid REFERENCES users (id) ON DELETE CASCADE,
    interval_seconds int NOT NULL DEFAULT 5,
    last_polled_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz NOT NULL
);

CREATE TABLE refresh_tokens (
    id uuid PRIMARY KEY,
    family_id uuid NOT NULL,
    token_hash bytea NOT NULL UNIQUE,
    user_id uuid NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    client_id text NOT NULL,
    scope text NOT NULL,
    device_name text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now(),
    family_expires_at timestamptz NOT NULL,
    idle_expires_at timestamptz NOT NULL,
    used_at timestamptz,
    revoked_at timestamptz,
    revoked_reason text NOT NULL DEFAULT ''
);
CREATE INDEX refresh_tokens_family ON refresh_tokens (family_id);
CREATE INDEX refresh_tokens_user ON refresh_tokens (user_id) WHERE revoked_at IS NULL;

CREATE TABLE access_tokens (
    id uuid PRIMARY KEY,
    token_hash bytea NOT NULL UNIQUE,
    user_id uuid NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    family_id uuid,
    client_id text NOT NULL,
    scope text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz NOT NULL,
    revoked_at timestamptz
);
CREATE INDEX access_tokens_user ON access_tokens (user_id) WHERE revoked_at IS NULL;
CREATE INDEX access_tokens_family ON access_tokens (family_id);
