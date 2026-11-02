package console

import (
	"strings"

	"github.com/ashhart/TensorFold/starlight/internal/authz"
)

type navItem struct {
	Label string
	Path  string
	Perm  authz.Perm
}

type navSection struct {
	Label string
	Items []navItem
}

// navigation lists every console page; items appear only for users who hold their permission.
var navigation = []navSection{
	{Label: "", Items: []navItem{{"Overview", "/console", ""}, {"My account", "/console/account", ""}, {"Plans", "/console/plans", ""}}},
	{Label: "Workspace", Items: []navItem{{"Projects", "/console/projects", ""}, {"Usage", "/console/usage", ""}}},
	{Label: "Fleet", Items: []navItem{{"Models", "/console/models", authz.ModelsRead}, {"Nodes", "/console/nodes", authz.NodesRead},
		{"Providers", "/console/providers", authz.ProvidersRead}, {"Limits", "/console/limits", authz.LimitsWrite},
		{"Subscriptions", "/console/subscriptions", authz.SubscriptionsRead}}},
	{Label: "Identity", Items: []navItem{{"Single sign-on", "/console/sso", authz.SSORead}, {"Users", "/console/users", authz.UsersRead},
		{"Groups", "/console/groups", authz.UsersRead}, {"Administrators", "/console/access", authz.RolesWrite}}},
	{Label: "Security", Items: []navItem{{"Audit log", "/console/audit", authz.AuditRead}, {"Settings", "/console/settings", authz.SettingsRead}}},
}

func (q *Request) nav() []navSection {
	var out []navSection
	for _, s := range navigation {
		var items []navItem
		for _, it := range s.Items {
			if it.Perm == "" || q.Grants.Can(it.Perm, "") {
				items = append(items, it)
			}
		}
		if len(items) > 0 {
			out = append(out, navSection{Label: s.Label, Items: items})
		}
	}
	return out
}

// Active reports whether path is the current page or one of its children, for nav highlighting.
func (v view) Current(path string) bool {
	if path == "/console" {
		return v.Active == "/console"
	}
	return v.Active == path || strings.HasPrefix(v.Active, path+"/")
}
