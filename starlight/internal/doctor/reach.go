package doctor

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

func (d *Doctor) fleet(ctx context.Context) {
	var active, stale, pending int
	_ = d.Pool.QueryRow(ctx, `SELECT count(*) FILTER (WHERE status = 'active' AND last_seen_at > now() - interval '90 seconds'),
		count(*) FILTER (WHERE status = 'active' AND (last_seen_at IS NULL OR last_seen_at <= now() - interval '90 seconds')),
		count(*) FILTER (WHERE status = 'pending') FROM nodes`).Scan(&active, &stale, &pending)
	switch {
	case stale > 0:
		d.add("nodes", Warn, fmt.Sprintf("%d online, %d not seen for 90 s", active, stale), "check those nodes' agents and their route to the node listener")
	case active == 0:
		d.add("nodes", Warn, "no nodes online", "enrol a node with `starlight nodes token`")
	default:
		d.add("nodes", OK, fmt.Sprintf("%d online", active), "")
	}
	if pending > 0 {
		d.add("nodes", Warn, fmt.Sprintf("%d awaiting approval", pending), "approve them in the console or with `starlight nodes approve`")
	}
	var expiring int
	_ = d.Pool.QueryRow(ctx, "SELECT count(*) FROM nodes WHERE status = 'active' AND cert_not_after < now() + interval '7 days'").Scan(&expiring)
	if expiring > 0 {
		d.add("nodes", Warn, fmt.Sprintf("%d node certificates expire within 7 days", expiring), "renewal is automatic; a stuck node needs its agent restarted")
	}
	var failed int
	_ = d.Pool.QueryRow(ctx, "SELECT count(*) FROM replicas WHERE state = 'failed'").Scan(&failed)
	if failed > 0 {
		d.add("replicas", Warn, fmt.Sprintf("%d replicas failed", failed), "open the deployment in the console to read the engine error")
	}
}

func (d *Doctor) identity(ctx context.Context) {
	rows, err := d.Pool.Query(ctx, "SELECT slug, protocol, config FROM idp_connections WHERE enabled ORDER BY slug")
	if err != nil {
		d.add("sso", Fail, err.Error(), "")
		return
	}
	defer rows.Close()
	n := 0
	for rows.Next() {
		n++
		var slug, protocol string
		var raw []byte
		if rows.Scan(&slug, &protocol, &raw) != nil {
			continue
		}
		var cfg struct {
			OIDC *struct {
				Issuer string `json:"issuer"`
			} `json:"oidc"`
			SAML *struct {
				MetadataURL string `json:"metadata_url"`
			} `json:"saml"`
		}
		_ = json.Unmarshal(raw, &cfg)
		target := ""
		switch {
		case cfg.OIDC != nil && cfg.OIDC.Issuer != "":
			target = strings.TrimRight(cfg.OIDC.Issuer, "/") + "/.well-known/openid-configuration"
		case cfg.SAML != nil && cfg.SAML.MetadataURL != "":
			target = cfg.SAML.MetadataURL
		}
		if target == "" {
			d.add("sso", OK, slug+": "+protocol+" with inline metadata", "")
			continue
		}
		d.reach(ctx, "sso", slug, target)
	}
	if n == 0 {
		d.add("sso", Warn, "no enabled identity provider connections", "add one in the console under Single sign-on")
	}
}

func (d *Doctor) reach(ctx context.Context, area, name, target string) {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, target, nil)
	if err != nil {
		d.add(area, Fail, name+": "+err.Error(), "")
		return
	}
	resp, err := d.HTTP.Do(req)
	if err != nil {
		d.add(area, Fail, name+": "+err.Error(), "check DNS, egress to the IdP and identity.ca_file for a private CA")
		return
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		d.add(area, Fail, fmt.Sprintf("%s: %s answered %d", name, target, resp.StatusCode), "check the issuer or metadata URL")
		return
	}
	d.add(area, OK, name+": reachable", "")
}

func (d *Doctor) providers(ctx context.Context) {
	var offline bool
	_ = d.Pool.QueryRow(ctx, "SELECT offline FROM settings WHERE id").Scan(&offline)
	rows, err := d.Pool.Query(ctx, "SELECT name, base_url, credential IS NOT NULL FROM providers WHERE enabled ORDER BY name")
	if err != nil {
		return
	}
	defer rows.Close()
	for rows.Next() {
		var name, base string
		var hasCred bool
		if rows.Scan(&name, &base, &hasCred) != nil {
			continue
		}
		switch {
		case offline:
			d.add("providers", OK, name+": disabled by offline mode", "")
		case !hasCred:
			d.add("providers", Warn, name+": no credential", "add the provider's API key in the console")
		default:
			if err := resolves(ctx, hostOf(base)); err != nil {
				d.add("providers", Fail, name+": "+err.Error(), "check DNS and egress to the provider")
			} else {
				d.add("providers", OK, name+": resolves", "")
			}
		}
	}
}
