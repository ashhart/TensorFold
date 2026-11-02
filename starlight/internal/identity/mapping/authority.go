package mapping

import (
	"context"
	"errors"
	"strings"

	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
)

// ErrForeignDomain means the email's domain is routed to another connection, which alone may vouch for it.
var ErrForeignDomain = errors.New("mapping: that email's domain belongs to another sign-in connection")

func domainOf(email string) string {
	_, d, ok := strings.Cut(strings.ToLower(strings.TrimSpace(email)), "@")
	if !ok {
		return ""
	}
	return d
}

// authoritative reports whether the connection owns the email's domain and may therefore vouch for it.
func authoritative(c connections.Connection, email string) bool {
	return inDomains(email, c.Domains) || c.Preset == "google" && inDomains(email, c.Config.Provider.GoogleDomains)
}

// OwnedElsewhere reports whether another connection's domains include the email's domain.
func OwnedElsewhere(ctx context.Context, q db.Querier, connectionID, email string) (bool, error) {
	d := domainOf(email)
	if d == "" {
		return false, nil
	}
	var owned bool
	err := q.QueryRow(ctx, "SELECT EXISTS (SELECT 1 FROM idp_connections WHERE id <> $1 AND $2 = ANY(domains))", connectionID, d).Scan(&owned)
	return owned, err
}
