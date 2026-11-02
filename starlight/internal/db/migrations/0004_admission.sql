CREATE TABLE gateways (
    id uuid PRIMARY KEY,
    hostname text NOT NULL DEFAULT '',
    version text NOT NULL DEFAULT '',
    started_at timestamptz NOT NULL DEFAULT now(),
    heartbeat_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE leases (
    id uuid PRIMARY KEY,
    request_id text NOT NULL,
    gateway_id uuid NOT NULL,
    model_id uuid NOT NULL,
    replica_id uuid,
    route_id uuid,
    scopes text[] NOT NULL,
    budget_keys text[] NOT NULL DEFAULT '{}',
    reserved_tokens bigint NOT NULL DEFAULT 0,
    reserved_micros bigint NOT NULL DEFAULT 0,
    priority text NOT NULL,
    principal_type text NOT NULL DEFAULT '',
    principal_id uuid,
    project_id uuid,
    client text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now(),
    heartbeat_at timestamptz NOT NULL DEFAULT now(),
    deadline timestamptz NOT NULL
);
CREATE INDEX leases_replica ON leases (replica_id) WHERE replica_id IS NOT NULL;
CREATE INDEX leases_route ON leases (route_id) WHERE route_id IS NOT NULL;
CREATE INDEX leases_scopes ON leases USING gin (scopes);
CREATE INDEX leases_gateway ON leases (gateway_id);

CREATE TABLE admission_queue (
    id uuid PRIMARY KEY,
    request_id text NOT NULL,
    gateway_id uuid NOT NULL,
    model_id uuid NOT NULL,
    priority text NOT NULL,
    base_rank int NOT NULL,
    candidates uuid[] NOT NULL,
    enqueued_at timestamptz NOT NULL DEFAULT now(),
    heartbeat_at timestamptz NOT NULL DEFAULT now(),
    deadline timestamptz NOT NULL
);
CREATE INDEX admission_queue_model ON admission_queue (model_id, enqueued_at);
CREATE INDEX admission_queue_gateway ON admission_queue (gateway_id);

CREATE TABLE counters (
    key text NOT NULL,
    window_start timestamptz NOT NULL,
    used bigint NOT NULL DEFAULT 0,
    reserved bigint NOT NULL DEFAULT 0,
    PRIMARY KEY (key, window_start)
);

CREATE TABLE usage_events (
    id uuid NOT NULL,
    created_at timestamptz NOT NULL,
    request_id text NOT NULL,
    principal_type text NOT NULL,
    principal_id uuid NOT NULL,
    user_id uuid,
    project_id uuid NOT NULL,
    api_key_id uuid,
    task_id uuid,
    model_id uuid NOT NULL,
    model_name text NOT NULL,
    route_kind text NOT NULL,
    route_id uuid,
    node_id uuid,
    protocol text NOT NULL,
    status int NOT NULL,
    outcome text NOT NULL,
    prompt_tokens bigint NOT NULL DEFAULT 0,
    completion_tokens bigint NOT NULL DEFAULT 0,
    cached_tokens bigint NOT NULL DEFAULT 0,
    cost_micros bigint NOT NULL DEFAULT 0,
    queue_ms int NOT NULL DEFAULT 0,
    ttft_ms int NOT NULL DEFAULT 0,
    duration_ms int NOT NULL DEFAULT 0,
    classification text NOT NULL,
    priority text NOT NULL,
    client text NOT NULL DEFAULT '',
    PRIMARY KEY (created_at, id)
) PARTITION BY RANGE (created_at);
CREATE TABLE usage_events_default PARTITION OF usage_events DEFAULT;
CREATE INDEX usage_events_project ON usage_events (project_id, created_at);
CREATE INDEX usage_events_user ON usage_events (user_id, created_at);

CREATE TABLE response_objects (
    response_id text PRIMARY KEY,
    principal_type text NOT NULL,
    principal_id uuid NOT NULL,
    project_id uuid NOT NULL,
    replica_id uuid REFERENCES replicas (id) ON DELETE CASCADE,
    route_id uuid REFERENCES provider_routes (id) ON DELETE CASCADE,
    created_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz NOT NULL,
    CHECK ((replica_id IS NULL) <> (route_id IS NULL))
);
