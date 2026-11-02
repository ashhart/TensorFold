CREATE TABLE data_keys (
    id text PRIMARY KEY,
    wrapped bytea NOT NULL,
    master_key_id text NOT NULL,
    status text NOT NULL CHECK (status IN ('active', 'retired')),
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX data_keys_one_active ON data_keys (status) WHERE status = 'active';

CREATE TABLE pki (
    name text PRIMARY KEY,
    cert_pem text NOT NULL,
    key_sealed bytea NOT NULL,
    not_after timestamptz NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE permit_keys (
    kid text PRIMARY KEY,
    public_key bytea NOT NULL,
    private_sealed bytea NOT NULL,
    status text NOT NULL CHECK (status IN ('next', 'active', 'verify', 'retired')),
    created_at timestamptz NOT NULL DEFAULT now(),
    changed_at timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX permit_keys_one_active ON permit_keys (status) WHERE status = 'active';
CREATE UNIQUE INDEX permit_keys_one_next ON permit_keys (status) WHERE status = 'next';

CREATE TABLE controller_leader (
    id boolean PRIMARY KEY DEFAULT true CHECK (id),
    holder uuid,
    epoch bigint NOT NULL DEFAULT 0,
    expires_at timestamptz NOT NULL DEFAULT 'epoch'
);
INSERT INTO controller_leader (id) VALUES (true);

CREATE TABLE recovery_codes (
    id uuid PRIMARY KEY,
    code_hash bytea NOT NULL UNIQUE,
    purpose text NOT NULL CHECK (purpose IN ('setup', 'recovery')),
    created_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz NOT NULL,
    used_at timestamptz,
    used_by uuid
);

CREATE TABLE audit_events (
    seq bigint PRIMARY KEY,
    created_at timestamptz NOT NULL,
    actor_type text NOT NULL,
    actor_id text NOT NULL,
    actor_label text NOT NULL DEFAULT '',
    action text NOT NULL,
    target_type text NOT NULL DEFAULT '',
    target_id text NOT NULL DEFAULT '',
    outcome text NOT NULL DEFAULT 'success',
    ip text NOT NULL DEFAULT '',
    user_agent text NOT NULL DEFAULT '',
    request_id text NOT NULL DEFAULT '',
    details jsonb NOT NULL DEFAULT '{}'::jsonb,
    prev_hash bytea NOT NULL,
    hash bytea NOT NULL
);
CREATE INDEX audit_events_time ON audit_events (created_at);
CREATE INDEX audit_events_action ON audit_events (action, created_at);
CREATE INDEX audit_events_actor ON audit_events (actor_id, created_at);

CREATE TABLE subscriptions (
    id uuid PRIMARY KEY,
    vendor text NOT NULL,
    product text NOT NULL,
    seats int NOT NULL CHECK (seats >= 0),
    seat_price_micros bigint NOT NULL DEFAULT 0 CHECK (seat_price_micros >= 0),
    billing_period text NOT NULL DEFAULT 'month' CHECK (billing_period IN ('month', 'year')),
    renews_on date,
    notes text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE subscription_seats (
    subscription_id uuid NOT NULL REFERENCES subscriptions (id) ON DELETE CASCADE,
    user_id uuid NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    assigned_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (subscription_id, user_id)
);
