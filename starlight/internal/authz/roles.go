// Package authz maps role bindings to permissions at organisation and project scope.
package authz

// Perm is one permission checked at an API boundary.
type Perm string

// Organisation permissions.
const (
	SettingsRead       Perm = "settings.read"
	SettingsWrite      Perm = "settings.write"
	SSORead            Perm = "sso.read"
	SSOWrite           Perm = "sso.write"
	UsersRead          Perm = "users.read"
	UsersWrite         Perm = "users.write"
	RolesWrite         Perm = "roles.write"
	OwnersWrite        Perm = "owners.write"
	ProjectsRead       Perm = "projects.read"
	ProjectsWrite      Perm = "projects.write"
	NodesRead          Perm = "nodes.read"
	NodesWrite         Perm = "nodes.write"
	ModelsRead         Perm = "models.read"
	ModelsWrite        Perm = "models.write"
	ProvidersRead      Perm = "providers.read"
	ProvidersWrite     Perm = "providers.write"
	LimitsWrite        Perm = "limits.write"
	UsageRead          Perm = "usage.read"
	AuditRead          Perm = "audit.read"
	SubscriptionsRead  Perm = "subscriptions.read"
	SubscriptionsWrite Perm = "subscriptions.write"
)

// Project permissions.
const (
	ProjectView    Perm = "project.view"
	ProjectMembers Perm = "project.members.write"
	ProjectKeys    Perm = "project.keys.write"
	ProjectLimits  Perm = "project.limits.write"
	ProjectUsage   Perm = "project.usage.read"
	ProjectApprove Perm = "project.plans.approve"
	Inference      Perm = "inference.use"
	OwnKeys        Perm = "keys.self"
	Tasks          Perm = "tasks.create"
	Plans          Perm = "plans.create"
)

var orgPerms = []Perm{SettingsRead, SettingsWrite, SSORead, SSOWrite, UsersRead, UsersWrite, RolesWrite, OwnersWrite, ProjectsRead,
	ProjectsWrite, NodesRead, NodesWrite, ModelsRead, ModelsWrite, ProvidersRead, ProvidersWrite, LimitsWrite, UsageRead, AuditRead,
	SubscriptionsRead, SubscriptionsWrite}

var projectPerms = []Perm{ProjectView, ProjectMembers, ProjectKeys, ProjectLimits, ProjectUsage, ProjectApprove, Inference, OwnKeys, Tasks, Plans}

type role struct {
	org     []Perm
	project []Perm
	scoped  bool
}

var roles = map[string]role{
	"owner": {org: orgPerms, project: projectPerms},
	"security_admin": {org: []Perm{SettingsRead, SettingsWrite, SSORead, SSOWrite, UsersRead, UsersWrite, RolesWrite, ProjectsRead,
		NodesRead, ModelsRead, ProvidersRead, ProvidersWrite, UsageRead, AuditRead}, project: []Perm{ProjectView, ProjectUsage}},
	"fleet_operator":  {org: []Perm{NodesRead, NodesWrite, ModelsRead, ProjectsRead, UsageRead}, project: []Perm{ProjectView}},
	"model_publisher": {org: []Perm{ModelsRead, ModelsWrite, NodesRead, ProvidersRead, ProjectsRead}, project: []Perm{ProjectView}},
	"budget_admin": {org: []Perm{LimitsWrite, UsageRead, SubscriptionsRead, SubscriptionsWrite, ProjectsRead, ModelsRead, ProvidersRead},
		project: []Perm{ProjectView, ProjectUsage, ProjectLimits}},
	"auditor": {org: []Perm{SettingsRead, SSORead, UsersRead, ProjectsRead, NodesRead, ModelsRead, ProvidersRead, UsageRead, AuditRead,
		SubscriptionsRead}, project: []Perm{ProjectView, ProjectUsage}},
	"project_admin": {scoped: true, project: []Perm{ProjectView, ProjectMembers, ProjectKeys, ProjectLimits, ProjectUsage, ProjectApprove,
		Inference, OwnKeys, Tasks, Plans}},
	"member": {scoped: true, project: []Perm{ProjectView, Inference, OwnKeys, Tasks, Plans}},
	"viewer": {scoped: true, project: []Perm{ProjectView, ProjectUsage}},
}

// ValidRole reports whether name is a role that may be bound at the given scope.
func ValidRole(name string, projectScoped bool) bool {
	r, ok := roles[name]
	return ok && r.scoped == projectScoped
}

// Roles lists role names for one scope, in a stable order for forms.
func Roles(projectScoped bool) []string {
	order := []string{"owner", "security_admin", "fleet_operator", "model_publisher", "budget_admin", "auditor", "project_admin", "member", "viewer"}
	var out []string
	for _, n := range order {
		if roles[n].scoped == projectScoped {
			out = append(out, n)
		}
	}
	return out
}

func has(list []Perm, p Perm) bool {
	for _, x := range list {
		if x == p {
			return true
		}
	}
	return false
}
