package declare

import (
	"fmt"

	"github.com/ashhart/TensorFold/starlight/internal/admin/fleet"
)

func (a *applier) settings(f *File) error {
	if f.Settings == nil {
		return nil
	}
	s := f.Settings
	changed, err := a.tx.ApplySettings(a.ctx, fleet.SettingsInput{OrgName: s.OrgName, Offline: s.Offline, AuditRetentionDays: s.AuditRetentionDays})
	if err != nil {
		return fmt.Errorf("settings: %w", err)
	}
	var d diff
	for _, k := range []string{"org_name", "offline", "audit_retention_days"} {
		if v, ok := changed[k]; ok {
			d = append(d, fmt.Sprintf("%s %s → %s", k, show(v[0]), show(v[1])))
		}
	}
	if len(d) > 0 {
		a.update("settings", "", d...)
	}
	return nil
}

func (a *applier) providers(f *File) error {
	for _, p := range f.Providers {
		if err := a.provider(p); err != nil {
			return fmt.Errorf("provider %s: %w", p.Name, err)
		}
	}
	return nil
}

func (a *applier) provider(p Provider) error {
	in, err := fleet.ProviderInput{Name: p.Name, Kind: p.Kind, BaseURL: p.BaseURL, MaxClass: p.MaxClass, Region: p.Region, Headers: p.Headers,
		Internal: p.Internal, Enabled: p.Enabled == nil || *p.Enabled}.Normalize()
	if err != nil {
		return err
	}
	secret, hasSecret := a.secrets[p.Name]
	cur, err := fleet.ProviderByName(a.ctx, a.tx.Q(), p.Name)
	if notFound(err) {
		in.Credential = secret
		if _, err := a.tx.CreateProvider(a.ctx, in); err != nil {
			return err
		}
		detail := []string{in.Kind + " at " + in.BaseURL}
		if hasSecret {
			detail = append(detail, "credential from "+p.Credential)
		}
		a.create("provider", p.Name, detail...)
		return nil
	}
	if err != nil {
		return err
	}
	var d diff
	field(&d, "kind", cur.Kind, in.Kind)
	field(&d, "base_url", cur.BaseURL, in.BaseURL)
	field(&d, "max_classification", cur.MaxClass, in.MaxClass)
	field(&d, "region", cur.Region, in.Region)
	fieldMap(&d, "headers", cur.Headers, in.Headers)
	field(&d, "internal", cur.Internal, in.Internal)
	field(&d, "enabled", cur.Enabled, in.Enabled)
	if hasSecret {
		same, err := a.tx.CredentialMatches(a.ctx, cur.ID, secret)
		if err != nil {
			return err
		}
		if !same {
			in.Credential = secret
			d = append(d, "credential replaced from "+p.Credential)
		}
	}
	if len(d) == 0 {
		return nil
	}
	if _, err := a.tx.UpdateProvider(a.ctx, cur.ID, in); err != nil {
		return err
	}
	a.update("provider", p.Name, d...)
	return nil
}
