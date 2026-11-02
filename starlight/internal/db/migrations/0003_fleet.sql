CREATE TABLE models (
    id uuid PRIMARY KEY,
    name text NOT NULL UNIQUE CHECK (name ~ '^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$'),
    display_name text NOT NULL,
    description text NOT NULL DEFAULT '',
    kind text NOT NULL DEFAULT 'chat' CHECK (kind IN ('chat', 'embedding')),
    capabilities jsonb NOT NULL DEFAULT '{}'::jsonb,
    route_policy text NOT NULL DEFAULT 'local_first' CHECK (route_policy IN ('local_only', 'local_first', 'cloud_first', 'cloud_only')),
    chargeback jsonb NOT NULL DEFAULT '{}'::jsonb,
    status text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'disabled')),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE model_aliases (
    alias text NOT NULL CHECK (alias ~ '^[A-Za-z0-9][A-Za-z0-9._:/@-]{0,127}$'),
    project_id uuid REFERENCES projects (id) ON DELETE CASCADE,
    model_id uuid NOT NULL REFERENCES models (id) ON DELETE CASCADE,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE NULLS NOT DISTINCT (alias, project_id)
);

CREATE TABLE model_grants (
    id uuid PRIMARY KEY,
    model_id uuid NOT NULL REFERENCES models (id) ON DELETE CASCADE,
    project_id uuid REFERENCES projects (id) ON DELETE CASCADE,
    max_context int CHECK (max_context > 0),
    max_output_tokens int CHECK (max_output_tokens > 0),
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE NULLS NOT DISTINCT (model_id, project_id)
);

CREATE TABLE providers (
    id uuid PRIMARY KEY,
    name text NOT NULL UNIQUE CHECK (name ~ '^[a-z0-9][a-z0-9-]{1,62}$'),
    kind text NOT NULL CHECK (kind IN ('openai', 'anthropic', 'azure_openai', 'gemini', 'openai_compatible')),
    base_url text NOT NULL,
    credential bytea,
    max_classification text NOT NULL DEFAULT 'internal' CHECK (max_classification IN ('public', 'internal', 'confidential', 'restricted')),
    region text NOT NULL DEFAULT '',
    headers jsonb NOT NULL DEFAULT '{}'::jsonb,
    internal boolean NOT NULL DEFAULT false,
    enabled boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE provider_routes (
    id uuid PRIMARY KEY,
    model_id uuid NOT NULL REFERENCES models (id) ON DELETE CASCADE,
    provider_id uuid NOT NULL REFERENCES providers (id) ON DELETE CASCADE,
    upstream_model text NOT NULL,
    protocols text[] NOT NULL,
    priority int NOT NULL DEFAULT 100,
    price_input_micros bigint NOT NULL DEFAULT 0 CHECK (price_input_micros >= 0),
    price_output_micros bigint NOT NULL DEFAULT 0 CHECK (price_output_micros >= 0),
    price_cached_micros bigint NOT NULL DEFAULT 0 CHECK (price_cached_micros >= 0),
    max_concurrency int NOT NULL DEFAULT 64 CHECK (max_concurrency > 0),
    enabled boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (model_id, provider_id, upstream_model)
);

CREATE TABLE nodes (
    id uuid PRIMARY KEY,
    name text NOT NULL UNIQUE CHECK (name ~ '^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$'),
    status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'active', 'draining', 'disabled', 'revoked')),
    pool text NOT NULL DEFAULT 'default',
    labels jsonb NOT NULL DEFAULT '{}'::jsonb,
    max_classification text NOT NULL DEFAULT 'confidential' CHECK (max_classification IN ('public', 'internal', 'confidential', 'restricted')),
    address text NOT NULL DEFAULT '',
    facts jsonb NOT NULL DEFAULT '{}'::jsonb,
    cached_artifacts text[] NOT NULL DEFAULT '{}',
    agent_version text NOT NULL DEFAULT '',
    sync_generation bigint NOT NULL DEFAULT 1,
    last_seen_at timestamptz,
    cert_serial text NOT NULL DEFAULT '',
    cert_not_after timestamptz,
    approved_by uuid,
    approved_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE enrollment_tokens (
    id uuid PRIMARY KEY,
    token_hash bytea NOT NULL UNIQUE,
    pool text NOT NULL DEFAULT 'default',
    labels jsonb NOT NULL DEFAULT '{}'::jsonb,
    max_classification text NOT NULL DEFAULT 'confidential' CHECK (max_classification IN ('public', 'internal', 'confidential', 'restricted')),
    auto_approve boolean NOT NULL DEFAULT false,
    max_uses int NOT NULL DEFAULT 1 CHECK (max_uses > 0),
    uses int NOT NULL DEFAULT 0,
    expires_at timestamptz NOT NULL,
    created_by uuid,
    created_at timestamptz NOT NULL DEFAULT now(),
    revoked_at timestamptz
);

CREATE TABLE node_certificates (
    serial text PRIMARY KEY,
    node_id uuid NOT NULL REFERENCES nodes (id) ON DELETE CASCADE,
    not_after timestamptz NOT NULL,
    revoked_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE deployments (
    id uuid PRIMARY KEY,
    model_id uuid NOT NULL REFERENCES models (id) ON DELETE CASCADE,
    name text NOT NULL UNIQUE CHECK (name ~ '^[a-z0-9][a-z0-9-]{1,62}$'),
    revision int NOT NULL DEFAULT 1,
    spec jsonb NOT NULL,
    pool text NOT NULL DEFAULT 'default',
    selector jsonb NOT NULL DEFAULT '{}'::jsonb,
    mode text NOT NULL DEFAULT 'warm' CHECK (mode IN ('warm', 'on_demand', 'batch')),
    min_replicas int NOT NULL DEFAULT 1 CHECK (min_replicas >= 0),
    max_replicas int NOT NULL DEFAULT 1 CHECK (max_replicas >= 1),
    idle_seconds int NOT NULL DEFAULT 900 CHECK (idle_seconds >= 60),
    status text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'paused', 'retired')),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CHECK (min_replicas <= max_replicas)
);

CREATE TABLE replicas (
    id uuid PRIMARY KEY,
    deployment_id uuid NOT NULL REFERENCES deployments (id) ON DELETE CASCADE,
    deployment_revision int NOT NULL,
    spec jsonb NOT NULL DEFAULT '{}'::jsonb,
    node_id uuid NOT NULL REFERENCES nodes (id) ON DELETE CASCADE,
    devices text[] NOT NULL DEFAULT '{}',
    rank_port int,
    desired text NOT NULL DEFAULT 'running' CHECK (desired IN ('running', 'stopped')),
    state text NOT NULL DEFAULT 'assigned' CHECK (state IN ('assigned', 'staging', 'starting', 'ready', 'draining', 'stopped', 'failed', 'lost')),
    slots int NOT NULL DEFAULT 1 CHECK (slots > 0),
    error text NOT NULL DEFAULT '',
    epoch bigint NOT NULL,
    generation bigint NOT NULL DEFAULT 1,
    last_request_at timestamptz,
    ready_at timestamptz,
    live jsonb NOT NULL DEFAULT '{}'::jsonb,
    live_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX replicas_deployment ON replicas (deployment_id);
CREATE INDEX replicas_node ON replicas (node_id);

CREATE TABLE replica_members (
    replica_id uuid NOT NULL REFERENCES replicas (id) ON DELETE CASCADE,
    member int NOT NULL CHECK (member > 0),
    node_id uuid NOT NULL REFERENCES nodes (id) ON DELETE CASCADE,
    devices text[] NOT NULL DEFAULT '{}',
    state text NOT NULL DEFAULT 'assigned' CHECK (state IN ('assigned', 'staging', 'starting', 'ready', 'draining', 'stopped', 'failed', 'lost')),
    error text NOT NULL DEFAULT '',
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (replica_id, member)
);
CREATE INDEX replica_members_node ON replica_members (node_id);

CREATE TABLE limits (
    id uuid PRIMARY KEY,
    scope_type text NOT NULL CHECK (scope_type IN ('org', 'project', 'member', 'user', 'service_account', 'api_key')),
    scope_id uuid,
    model_id uuid REFERENCES models (id) ON DELETE CASCADE,
    kind text NOT NULL CHECK (kind IN ('concurrency', 'requests_per_minute', 'tokens_per_minute', 'tokens_per_day', 'tokens_per_month', 'cost_per_day', 'cost_per_month')),
    value bigint NOT NULL CHECK (value >= 0),
    boost_value bigint CHECK (boost_value >= 0),
    boost_until timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE NULLS NOT DISTINCT (scope_type, scope_id, model_id, kind)
);
