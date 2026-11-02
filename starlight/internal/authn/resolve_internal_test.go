package authn

import (
	"context"
	"errors"
	"net"
	"testing"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/authz"
	"github.com/ashhart/TensorFold/starlight/internal/db"
)

func TestProjectLookupOutageIsNotAProjectError(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := ln.Addr().String()
	ln.Close()
	pool, err := pgxpool.New(context.Background(), "postgres://starlight@"+addr+"/x?sslmode=disable&connect_timeout=1")
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	a := &Authenticator{Pool: pool}
	for _, hint := range []string{"research", ""} {
		if _, err := a.pickProject(context.Background(), authz.Grants{}, hint); errors.Is(err, ErrProject) || !db.Unavailable(err) {
			t.Errorf("choosing a project with hint %q while the database is down: %v", hint, err)
		}
	}
}
