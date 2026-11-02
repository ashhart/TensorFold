package mapping

import (
	"context"
	"errors"
	"strings"

	"github.com/jackc/pgx/v5"

	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
)

// CheckPolicy enforces the connection's allowed domains and required groups.
func CheckPolicy(c connections.Connection, id Identity) error {
	p := c.Config.Policy
	if len(p.AllowedDomains) > 0 && !inDomains(id.Email, p.AllowedDomains) {
		return ErrDomainNotAllowed
	}
	if len(p.RequiredGroups) == 0 {
		return nil
	}
	for _, g := range id.Groups {
		for _, want := range p.RequiredGroups {
			if strings.EqualFold(g.ExternalID, want) || strings.EqualFold(g.DisplayName, want) {
				return nil
			}
		}
	}
	return ErrGroupRequired
}

// Resolve finds or creates the user for a verified identity, links it and syncs claim groups, all inside tx.
func Resolve(ctx context.Context, tx pgx.Tx, c connections.Connection, id Identity) (directory.User, error) {
	if err := CheckPolicy(c, id); err != nil {
		return directory.User{}, err
	}
	u, err := linked(ctx, tx, c, id)
	if errors.Is(err, directory.ErrNotFound) {
		u, err = link(ctx, tx, c, id)
	}
	if err != nil {
		return directory.User{}, err
	}
	if id.GroupsKnown {
		if err := directory.SetClaimGroups(ctx, tx, u.ID, c.ID, id.Groups); err != nil {
			return directory.User{}, err
		}
	}
	if err := directory.TouchLogin(ctx, tx, u.ID); err != nil {
		return directory.User{}, err
	}
	return u, nil
}

// Lookup finds the user already linked to a verified bearer identity; it never creates or links accounts.
func Lookup(ctx context.Context, q db.Querier, c connections.Connection, id Identity) (directory.User, error) {
	userID, err := directory.FindIdentity(ctx, q, c.ID, id.Subject)
	if err != nil {
		if errors.Is(err, directory.ErrNotFound) {
			return directory.User{}, ErrNotProvisioned
		}
		return directory.User{}, err
	}
	u, err := directory.GetUser(ctx, q, userID)
	if err != nil {
		return directory.User{}, err
	}
	if u.Status != directory.Active {
		return directory.User{}, ErrDisabled
	}
	return u, nil
}

func linked(ctx context.Context, tx pgx.Tx, c connections.Connection, id Identity) (directory.User, error) {
	userID, err := directory.FindIdentity(ctx, tx, c.ID, id.Subject)
	if err != nil {
		return directory.User{}, err
	}
	u, err := directory.GetUser(ctx, tx, userID)
	if err != nil {
		return directory.User{}, err
	}
	if u.Status != directory.Active {
		return directory.User{}, ErrDisabled
	}
	if err := refreshProfile(ctx, tx, c, &u, id); err != nil {
		return directory.User{}, err
	}
	return u, directory.LinkIdentity(ctx, tx, u.ID, c.ID, id.Subject, id.Email)
}

// refreshProfile updates only just-in-time users, never into another connection's domain; SCIM owns its own users.
func refreshProfile(ctx context.Context, tx pgx.Tx, c connections.Connection, u *directory.User, id Identity) error {
	if u.Source != "jit" || u.SCIMConnectionID != nil {
		return nil
	}
	email, name := u.Email, u.DisplayName
	if id.Email != "" && id.EmailVerified {
		foreign, err := OwnedElsewhere(ctx, tx, c.ID, id.Email)
		if err != nil {
			return err
		}
		if !foreign {
			email = id.Email
		}
	}
	if id.Name != "" {
		name = id.Name
	}
	if email == u.Email && name == u.DisplayName {
		return nil
	}
	if _, err := tx.Exec(ctx, "SAVEPOINT profile"); err != nil {
		return err
	}
	err := directory.UpdateProfile(ctx, tx, u.ID, email, name)
	if errors.Is(err, directory.ErrConflict) {
		_, err = tx.Exec(ctx, "ROLLBACK TO SAVEPOINT profile")
		return err
	}
	if err != nil {
		return err
	}
	u.Email, u.DisplayName = email, name
	_, err = tx.Exec(ctx, "RELEASE SAVEPOINT profile")
	return err
}

// link attaches a new subject to an account; a connection may only vouch for emails in the domains it owns.
func link(ctx context.Context, tx pgx.Tx, c connections.Connection, id Identity) (directory.User, error) {
	if id.Email == "" {
		return directory.User{}, ErrMissingEmail
	}
	foreign, err := OwnedElsewhere(ctx, tx, c.ID, id.Email)
	if err != nil {
		return directory.User{}, err
	}
	if foreign {
		return directory.User{}, ErrForeignDomain
	}
	u, err := directory.FindByEmail(ctx, tx, id.Email)
	switch {
	case err == nil:
		if u.Status != directory.Active {
			return directory.User{}, ErrDisabled
		}
		if u.SCIMConnectionID != nil && *u.SCIMConnectionID != c.ID {
			return directory.User{}, ErrIdentityConflict
		}
		provisionedHere := u.SCIMConnectionID != nil && *u.SCIMConnectionID == c.ID
		if !provisionedHere && !(id.EmailVerified && authoritative(c, id.Email)) {
			return directory.User{}, ErrEmailUnverified
		}
		var other int
		if err := tx.QueryRow(ctx, "SELECT count(*) FROM identities WHERE user_id = $1 AND connection_id = $2", u.ID, c.ID).Scan(&other); err != nil {
			return directory.User{}, err
		}
		if other > 0 {
			return directory.User{}, ErrIdentityConflict
		}
	case errors.Is(err, directory.ErrNotFound):
		if !c.JIT {
			return directory.User{}, ErrNotProvisioned
		}
		if !id.EmailVerified {
			return directory.User{}, ErrUnverified
		}
		u, err = directory.CreateUser(ctx, tx, directory.NewUser{Email: id.Email, DisplayName: id.Name, Source: "jit"})
		if err != nil {
			return directory.User{}, err
		}
	default:
		return directory.User{}, err
	}
	if err := directory.LinkIdentity(ctx, tx, u.ID, c.ID, id.Subject, id.Email); err != nil {
		return directory.User{}, err
	}
	return u, nil
}
