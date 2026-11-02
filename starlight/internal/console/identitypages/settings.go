package identitypages

import (
	"github.com/ashhart/TensorFold/starlight/internal/admin/settings"
	"github.com/ashhart/TensorFold/starlight/internal/authz"
	"github.com/ashhart/TensorFold/starlight/internal/console"
)

type settingsPage struct {
	settings.Settings
	CanWrite      bool
	Version       string
	PublicURL     string
	NodeURL       string
	CAFingerprint string
}

type offlinePage struct {
	Offline bool
	Turning bool
}

func (p pages) settings() {
	c := p.c
	c.Handle("GET /console/settings", authz.SettingsRead, func(q *console.Request) error {
		s, err := settings.Get(q.Ctx(), c.Pool)
		if err != nil {
			return err
		}
		page := settingsPage{Settings: s, CanWrite: q.Can(authz.SettingsWrite, ""), Version: c.Version, PublicURL: c.PublicURL, NodeURL: c.NodeURL}
		if c.Trust != nil {
			page.CAFingerprint = c.Trust.Fingerprint()
		}
		q.Render("identity/settings", "Settings", page)
		return nil
	})
	c.Handle("POST /console/settings", authz.SettingsWrite, func(q *console.Request) error {
		name := q.Form("org_name")
		days, err := q.Int("audit_retention_days", settings.DefaultRetentionDays)
		if err != nil {
			return err
		}
		if _, err := settings.Update(q.Ctx(), c.Pool, q.Audit("", "", "", nil), settings.Change{OrgName: &name, AuditRetentionDays: &days}); err != nil {
			return friendly(err)
		}
		c.Flush()
		return q.Redirect("/console/settings", "Settings saved.")
	})
	c.Handle("POST /console/settings/offline", authz.SettingsWrite, func(q *console.Request) error {
		turning := q.Form("offline") == "on"
		if q.Form("confirm") != "yes" {
			s, err := settings.Get(q.Ctx(), c.Pool)
			if err != nil {
				return err
			}
			q.Render("identity/settings_offline", "Confirm offline mode", offlinePage{Offline: s.Offline, Turning: turning})
			return nil
		}
		if _, err := settings.Update(q.Ctx(), c.Pool, q.Audit("", "", "", nil), settings.Change{Offline: &turning}); err != nil {
			return friendly(err)
		}
		c.Flush()
		notice := "Offline mode is off; approved cloud providers may be used again."
		if turning {
			notice = "Offline mode is on; no request can reach a cloud provider."
		}
		return q.Redirect("/console/settings", notice)
	})
}
