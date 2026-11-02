// Package principal describes who is calling, as established by a verified credential.
package principal

import "context"

// Type is the kind of account a request acts as.
type Type string

const (
	User           Type = "user"
	ServiceAccount Type = "service_account"
)

// Credential records how the principal was authenticated.
type Credential string

const (
	ViaAPIKey  Credential = "api_key"
	ViaAccess  Credential = "access_token"
	ViaTask    Credential = "task_token"
	ViaIDPJWT  Credential = "idp_jwt"
	ViaSession Credential = "session"
)

// Principal is the authenticated caller; nothing in it comes from client-supplied metadata.
type Principal struct {
	Type        Type
	ID          string
	UserID      string
	Email       string
	DisplayName string
	ProjectID   string
	Groups      []string
	Via         Credential
	KeyID       string
	TaskID      string
	TaskRootID  string
	Client      string
	Scope       []string
}

// Label is a short human-readable name for logs and audit.
func (p *Principal) Label() string {
	if p.Email != "" {
		return p.Email
	}
	if p.DisplayName != "" {
		return p.DisplayName
	}
	return string(p.Type) + ":" + p.ID
}

// HasScope reports whether an OAuth-style scope was granted; principals without scopes carry all.
func (p *Principal) HasScope(s string) bool {
	if p.Scope == nil {
		return true
	}
	for _, have := range p.Scope {
		if have == s {
			return true
		}
	}
	return false
}

type ctxKey struct{}

// With attaches p to ctx.
func With(ctx context.Context, p *Principal) context.Context {
	return context.WithValue(ctx, ctxKey{}, p)
}

// From returns the principal attached to ctx, or nil.
func From(ctx context.Context) *Principal {
	p, _ := ctx.Value(ctxKey{}).(*Principal)
	return p
}
