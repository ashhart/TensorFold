CREATE TABLE tasks (
    id uuid PRIMARY KEY,
    root_id uuid NOT NULL,
    parent_id uuid REFERENCES tasks (id) ON DELETE CASCADE,
    ancestors uuid[] NOT NULL DEFAULT '{}',
    depth int NOT NULL CHECK (depth >= 0),
    name text NOT NULL,
    owner_type text NOT NULL CHECK (owner_type IN ('user', 'service_account')),
    owner_id uuid NOT NULL,
    project_id uuid NOT NULL REFERENCES projects (id) ON DELETE CASCADE,
    token_hash bytea NOT NULL UNIQUE,
    models text[] NOT NULL,
    pins jsonb NOT NULL DEFAULT '{}'::jsonb,
    local_only boolean NOT NULL DEFAULT true,
    max_depth int NOT NULL CHECK (max_depth >= 0),
    max_concurrency int NOT NULL CHECK (max_concurrency > 0),
    budget_tokens bigint CHECK (budget_tokens > 0),
    plan_id uuid,
    api_key_id uuid REFERENCES api_keys (id) ON DELETE CASCADE,
    refresh_family uuid,
    expires_at timestamptz NOT NULL,
    revoked_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX tasks_root ON tasks (root_id);
CREATE INDEX tasks_ancestors ON tasks USING gin (ancestors);
CREATE INDEX tasks_owner ON tasks (owner_type, owner_id) WHERE revoked_at IS NULL;

CREATE TABLE plans (
    id uuid PRIMARY KEY,
    project_id uuid NOT NULL REFERENCES projects (id) ON DELETE CASCADE,
    owner_type text NOT NULL CHECK (owner_type IN ('user', 'service_account')),
    owner_id uuid NOT NULL,
    title text NOT NULL,
    planner_model text NOT NULL,
    executor_model text NOT NULL,
    egress_sha256 bytea NOT NULL,
    egress_bytes int NOT NULL,
    plan_sealed bytea,
    plan_sha256 bytea,
    status text NOT NULL CHECK (status IN ('drafting', 'pending_approval', 'approved', 'rejected', 'failed', 'revoked')),
    error text NOT NULL DEFAULT '',
    budget_tokens bigint CHECK (budget_tokens > 0),
    max_concurrency int NOT NULL DEFAULT 4 CHECK (max_concurrency > 0),
    max_depth int NOT NULL DEFAULT 2 CHECK (max_depth >= 0),
    ttl_seconds int NOT NULL DEFAULT 86400 CHECK (ttl_seconds > 0),
    approved_by uuid,
    approved_at timestamptz,
    task_id uuid REFERENCES tasks (id) ON DELETE SET NULL,
    task_token_sealed bytea,
    api_key_id uuid REFERENCES api_keys (id) ON DELETE CASCADE,
    refresh_family uuid,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX plans_project ON plans (project_id, created_at);
