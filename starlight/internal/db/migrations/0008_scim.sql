ALTER TABLE users ADD COLUMN scim_attributes jsonb NOT NULL DEFAULT '{}'::jsonb;

DROP INDEX users_scim_name;
CREATE UNIQUE INDEX users_scim_name ON users (scim_connection_id, lower(scim_user_name))
    WHERE scim_user_name IS NOT NULL AND status <> 'deactivated';
CREATE INDEX users_scim_external ON users (scim_connection_id, external_id) WHERE external_id IS NOT NULL;

CREATE UNIQUE INDEX groups_scim_name ON groups (connection_id, lower(display_name)) WHERE source = 'scim';
