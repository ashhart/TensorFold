package fleet

import (
	"context"
	"errors"

	"github.com/ashhart/TensorFold/starlight/internal/admin/settings"
	"github.com/ashhart/TensorFold/starlight/internal/db"
)

// SettingsInput sets installation settings from configuration; nil fields stay as they are.
type SettingsInput struct {
	OrgName            *string
	Offline            *bool
	AuditRetentionDays *int
}

// ApplySettings changes installation settings inside this transaction and returns what changed, keyed by setting.
func (t *Tx) ApplySettings(ctx context.Context, in SettingsInput) (map[string][2]any, error) {
	before, changes, err := settings.Apply(ctx, t.tx, settings.Change(in))
	var invalid settings.ErrInvalid
	switch {
	case errors.As(err, &invalid):
		return nil, Invalid(string(invalid))
	case db.NotFound(err):
		return nil, Invalid("Starlight is not initialised yet; run starlight init first")
	case err != nil || len(changes) == 0:
		return nil, err
	}
	changed := map[string][2]any{}
	for k, v := range changes {
		ft := v.(map[string]any)
		changed[k] = [2]any{ft["from"], ft["to"]}
	}
	return changed, t.record(ctx, settings.Action(changes), "settings", before.InstallationID, changes)
}
