CREATE TABLE projects (
    id uuid PRIMARY KEY,
    slug text NOT NULL UNIQUE CHECK (slug ~ '^[a-z0-9][a-z0-9-]{1,62}$'),
    name text NOT NULL,
    description text NOT NULL DEFAULT '',
    classification text NOT NULL DEFAULT 'internal' CHECK (classification IN ('public', 'internal', 'confidential', 'restricted')),
    max_priority text NOT NULL DEFAULT 'interactive' CHECK (max_priority IN ('batch', 'standard', 'interactive')),
    everyone boolean NOT NULL DEFAULT false,
    plan_approval text NOT NULL DEFAULT 'required' CHECK (plan_approval IN ('auto', 'required')),
    secrets_guard text NOT NULL DEFAULT 'local' CHECK (secrets_guard IN ('block', 'local', 'off')),
    isolate_prompt_cache boolean NOT NULL DEFAULT false,
    archived_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE service_accounts (
    id uuid PRIMARY KEY,
    project_id uuid NOT NULL REFERENCES projects (id) ON DELETE CASCADE,
    name text NOT NULL CHECK (name ~ '^[a-z0-9][a-z0-9-]{1,62}$'),
    description text NOT NULL DEFAULT '',
    status text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'disabled')),
    created_by uuid,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (project_id, name)
);

CREATE TABLE role_bindings (
    id uuid PRIMARY KEY,
    role text NOT NULL,
    subject_type text NOT NULL CHECK (subject_type IN ('user', 'group', 'service_account')),
    subject_id uuid NOT NULL,
    project_id uuid REFERENCES projects (id) ON DELETE CASCADE,
    created_by uuid,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE NULLS NOT DISTINCT (role, subject_type, subject_id, project_id)
);
CREATE INDEX role_bindings_subject ON role_bindings (subject_type, subject_id);
CREATE INDEX role_bindings_project ON role_bindings (project_id);

CREATE TABLE api_keys (
    id uuid PRIMARY KEY,
    public_id text NOT NULL UNIQUE,
    secret_hash bytea NOT NULL,
    name text NOT NULL,
    owner_type text NOT NULL CHECK (owner_type IN ('user', 'service_account')),
    owner_id uuid NOT NULL,
    project_id uuid NOT NULL REFERENCES projects (id) ON DELETE CASCADE,
    models text[],
    priority text NOT NULL DEFAULT 'standard' CHECK (priority IN ('batch', 'standard', 'interactive')),
    classification text CHECK (classification IN ('public', 'internal', 'confidential', 'restricted')),
    allowed_cidrs cidr[] NOT NULL DEFAULT '{}',
    expires_at timestamptz,
    last_used_at timestamptz,
    revoked_at timestamptz,
    rotated_from uuid REFERENCES api_keys (id) ON DELETE SET NULL,
    grace_until timestamptz,
    created_by uuid,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX api_keys_owner ON api_keys (owner_type, owner_id) WHERE revoked_at IS NULL;
CREATE INDEX api_keys_project ON api_keys (project_id);
