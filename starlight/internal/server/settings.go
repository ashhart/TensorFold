// Package server assembles Starlight's components behind its listeners according to the instance's roles.
package server

import (
	"context"
	"errors"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"

	"github.com/ashhart/TensorFold/starlight/internal/crypto/keyring"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/identity/login"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/trust"
)

// Installation is the singleton settings row.
type Installation struct {
	ID          string
	TrustDomain string
	OrgName     string
	Offline     bool
}

// ErrNotInitialised means `starlight init` has not run against this database.
var ErrNotInitialised = errors.New("starlight is not initialised; run `starlight init`")

// LoadInstallation reads the settings row.
func LoadInstallation(ctx context.Context, q db.Querier) (Installation, error) {
	var in Installation
	err := q.QueryRow(ctx, "SELECT installation_id, trust_domain, org_name, offline FROM settings WHERE id").Scan(&in.ID, &in.TrustDomain, &in.OrgName, &in.Offline)
	if db.NotFound(err) {
		return in, ErrNotInitialised
	}
	return in, err
}

// InitOptions configure a new installation.
type InitOptions struct {
	OrgName        string
	Offline        bool
	GeneralProject bool
}

// Init creates the settings row, first data key, workload CA, permit key and a setup code; it refuses to run twice.
func Init(ctx context.Context, tx pgx.Tx, keys *keyring.Keyring, o InitOptions) (Installation, string, error) {
	var exists bool
	if err := tx.QueryRow(ctx, "SELECT EXISTS (SELECT 1 FROM settings)").Scan(&exists); err != nil {
		return Installation{}, "", err
	}
	if exists {
		return Installation{}, "", errors.New("this database is already initialised")
	}
	id := ids.New()
	raw := strings.ReplaceAll(id, "-", "")
	in := Installation{ID: id, TrustDomain: "starlight-" + raw[len(raw)-12:], OrgName: strings.TrimSpace(o.OrgName), Offline: o.Offline}
	if in.OrgName == "" {
		in.OrgName = "TensorFold Starlight"
	}
	if _, err := tx.Exec(ctx, "INSERT INTO settings (installation_id, trust_domain, org_name, offline) VALUES ($1, $2, $3, $4)",
		in.ID, in.TrustDomain, in.OrgName, in.Offline); err != nil {
		return in, "", err
	}
	if _, err := keys.Rotate(ctx, tx); err != nil {
		return in, "", err
	}
	if err := trust.Bootstrap(ctx, tx, keys, in.TrustDomain); err != nil {
		return in, "", err
	}
	if o.GeneralProject {
		if _, err := tx.Exec(ctx, "INSERT INTO projects (id, slug, name, description, everyone) VALUES ($1, 'general', 'General', 'Every signed-in user', true)",
			ids.New()); err != nil {
			return in, "", err
		}
	}
	code, err := login.IssueSetupCode(ctx, tx, 24*time.Hour)
	return in, code, err
}

// Short is a stable eight-character public identifier for mDNS and discovery.
func (in Installation) Short() string {
	raw := strings.ReplaceAll(in.ID, "-", "")
	return raw[len(raw)-8:]
}
