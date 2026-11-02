ALTER TABLE recovery_codes ADD COLUMN user_id uuid REFERENCES users (id) ON DELETE CASCADE;

ALTER TABLE auth_requests ADD COLUMN setup_code_id uuid REFERENCES recovery_codes (id) ON DELETE SET NULL;
CREATE INDEX auth_requests_expiry ON auth_requests (expires_at);

CREATE TABLE sso_replay (
    connection_id uuid NOT NULL REFERENCES idp_connections (id) ON DELETE CASCADE,
    kind text NOT NULL CHECK (kind IN ('saml_assertion', 'logout_jti', 'saml_logout')),
    value text NOT NULL,
    expires_at timestamptz NOT NULL,
    PRIMARY KEY (connection_id, kind, value)
);
CREATE INDEX sso_replay_expiry ON sso_replay (expires_at);

CREATE TABLE idp_metadata (
    connection_id uuid PRIMARY KEY REFERENCES idp_connections (id) ON DELETE CASCADE,
    body text NOT NULL,
    fetched_at timestamptz NOT NULL
);

ALTER TABLE sessions ADD COLUMN idp_name_format text NOT NULL DEFAULT '',
    ADD COLUMN idp_name_qualifier text NOT NULL DEFAULT '',
    ADD COLUMN idp_sp_name_qualifier text NOT NULL DEFAULT '';
CREATE INDEX sessions_idp_subject ON sessions (connection_id, idp_subject) WHERE revoked_at IS NULL;

CREATE TABLE saml_logout_requests (
    connection_id uuid NOT NULL REFERENCES idp_connections (id) ON DELETE CASCADE,
    request_id text NOT NULL,
    user_id uuid REFERENCES users (id) ON DELETE CASCADE,
    expires_at timestamptz NOT NULL,
    PRIMARY KEY (connection_id, request_id)
);
CREATE INDEX saml_logout_requests_expiry ON saml_logout_requests (expires_at);
