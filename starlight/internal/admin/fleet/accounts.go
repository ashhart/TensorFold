package fleet

import (
	"context"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/admin/keys"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

// ServiceAccount is a non-human identity owned by one project.
type ServiceAccount struct {
	ID          string
	ProjectID   string
	Name        string
	Description string
	Status      string
	LiveKeys    int
	CreatedAt   time.Time
}

// ServiceAccounts lists a project's service accounts.
func ServiceAccounts(ctx context.Context, q db.Querier, projectID string) ([]ServiceAccount, error) {
	rows, err := q.Query(ctx, `SELECT sa.id, sa.project_id, sa.name, sa.description, sa.status, sa.created_at,
		(SELECT count(*) FROM api_keys k WHERE k.owner_type = 'service_account' AND k.owner_id = sa.id AND k.revoked_at IS NULL
			AND (k.expires_at IS NULL OR k.expires_at > now()) AND (k.grace_until IS NULL OR k.grace_until > now()))
		FROM service_accounts sa WHERE sa.project_id = $1 ORDER BY sa.name`, projectID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []ServiceAccount
	for rows.Next() {
		var s ServiceAccount
		if err := rows.Scan(&s.ID, &s.ProjectID, &s.Name, &s.Description, &s.Status, &s.CreatedAt, &s.LiveKeys); err != nil {
			return nil, err
		}
		out = append(out, s)
	}
	return out, rows.Err()
}

// CreateServiceAccount adds a service account to a project.
func (t *Tx) CreateServiceAccount(ctx context.Context, projectID, name, description string) (string, error) {
	if err := slug("service account name", name); err != nil {
		return "", err
	}
	if err := text("description", description, 500, false); err != nil {
		return "", err
	}
	id := ids.New()
	_, err := t.tx.Exec(ctx, "INSERT INTO service_accounts (id, project_id, name, description, created_by) VALUES ($1,$2,$3,$4,$5)",
		id, projectID, name, description, optID(t.actor.ActorID))
	if err != nil {
		return "", uniqueMsg(err, "this project already has a service account with that name")
	}
	return id, t.record(ctx, "service_account.created", "service_account", id, map[string]any{"name": name, "project_id": projectID})
}

// SetServiceAccountStatus enables or disables an account; a disabled account's keys stop working at once.
func (t *Tx) SetServiceAccountStatus(ctx context.Context, projectID, id, status string) error {
	if err := oneOf("status", status, "active", "disabled"); err != nil {
		return err
	}
	tag, err := t.tx.Exec(ctx, "UPDATE service_accounts SET status = $3, updated_at = now() WHERE id = $1 AND project_id = $2", id, projectID, status)
	if err != nil {
		return err
	}
	if tag.RowsAffected() == 0 {
		return ErrNotFound
	}
	return t.record(ctx, "service_account."+status, "service_account", id, map[string]any{"project_id": projectID})
}

// IssueServiceKey mints a key for a service account in its project; the secret is returned once.
func (t *Tx) IssueServiceKey(ctx context.Context, projectID, accountID string, s keys.Spec) (string, keys.Key, error) {
	var owner string
	if err := t.tx.QueryRow(ctx, "SELECT project_id FROM service_accounts WHERE id = $1", accountID).Scan(&owner); err != nil || owner != projectID {
		return "", keys.Key{}, ErrNotFound
	}
	s.OwnerType, s.OwnerID, s.ProjectID, s.CreatedBy = "service_account", accountID, projectID, t.actor.ActorID
	tok, k, err := keys.Issue(ctx, t.tx, s)
	if err != nil {
		return "", k, Invalid(err.Error())
	}
	return tok, k, t.record(ctx, "api_key.created", "api_key", k.ID, map[string]any{"public_id": k.PublicID, "owner": accountID,
		"project_id": projectID, "priority": k.Priority})
}

// RevokeProjectKey revokes any key that belongs to the project.
func (t *Tx) RevokeProjectKey(ctx context.Context, projectID, keyID string) error {
	k, err := keys.Get(ctx, t.tx, keyID)
	if err != nil || k.ProjectID != projectID {
		return ErrNotFound
	}
	if err := keys.Revoke(ctx, t.tx, keyID); err != nil {
		return notFound(err)
	}
	return t.record(ctx, "api_key.revoked", "api_key", keyID, map[string]any{"public_id": k.PublicID, "project_id": projectID})
}

// RotateProjectKey replaces a project key, keeping the old one valid for grace.
func (t *Tx) RotateProjectKey(ctx context.Context, projectID, keyID string, grace time.Duration) (string, keys.Key, error) {
	k, err := keys.Get(ctx, t.tx, keyID)
	if err != nil || k.ProjectID != projectID {
		return "", keys.Key{}, ErrNotFound
	}
	tok, nk, err := keys.Rotate(ctx, t.tx, keyID, grace, t.actor.ActorID)
	if err != nil {
		return "", nk, Invalid(err.Error())
	}
	return tok, nk, t.record(ctx, "api_key.rotated", "api_key", nk.ID, map[string]any{"replaces": k.PublicID, "public_id": nk.PublicID,
		"grace_seconds": int(grace.Seconds())})
}

// ProjectKeys lists every key in a project, with owner labels.
func ProjectKeys(ctx context.Context, q db.Querier, projectID string) ([]keys.Key, map[string]string, error) {
	list, err := keys.List(ctx, q, keys.Filter{ProjectID: projectID})
	if err != nil {
		return nil, nil, err
	}
	owners := map[string]string{}
	rows, err := q.Query(ctx, `SELECT k.owner_id::text, coalesce(u.email, sa.name, '') FROM api_keys k
		LEFT JOIN users u ON k.owner_type = 'user' AND u.id = k.owner_id
		LEFT JOIN service_accounts sa ON k.owner_type = 'service_account' AND sa.id = k.owner_id WHERE k.project_id = $1`, projectID)
	if err != nil {
		return nil, nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var id, label string
		if err := rows.Scan(&id, &label); err != nil {
			return nil, nil, err
		}
		owners[id] = label
	}
	return list, owners, rows.Err()
}
