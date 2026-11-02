package doctor_test

import (
	"context"
	"testing"

	"github.com/jackc/pgx/v5"

	"github.com/ashhart/TensorFold/starlight/internal/config"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/keyring"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/doctor"
	"github.com/ashhart/TensorFold/starlight/internal/server"
	"github.com/ashhart/TensorFold/starlight/internal/testutil"
)

func TestDoctorOnAFreshInstallation(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	master := keyring.GenerateMasterKey()
	m, _ := keyring.LoadMaster(master, "")
	if err := db.Tx(ctx, pool, func(tx pgx.Tx) error {
		_, _, err := server.Init(ctx, tx, keyring.New(m), server.InitOptions{OrgName: "Example Corp"})
		return err
	}); err != nil {
		t.Fatal(err)
	}
	cfg := config.Defaults()
	cfg.PublicURL, cfg.Keys.MasterKey = "https://ai.example.com", master
	got := map[string]doctor.Status{}
	for _, c := range (&doctor.Doctor{Config: cfg, Pool: pool}).Run(ctx) {
		if got[c.Area] != doctor.Fail {
			got[c.Area] = c.Status
		}
	}
	for area, want := range map[string]doctor.Status{"database": doctor.OK, "schema": doctor.OK, "keys": doctor.OK, "installation": doctor.OK,
		"owners": doctor.Warn, "ca": doctor.OK, "permits": doctor.OK, "nodes": doctor.Warn, "sso": doctor.Warn} {
		if got[area] != want {
			t.Errorf("%s = %q, want %q (all: %v)", area, got[area], want, got)
		}
	}
	wrong := cfg
	wrong.Keys.MasterKey = keyring.GenerateMasterKey()
	for _, c := range (&doctor.Doctor{Config: wrong, Pool: pool}).Run(ctx) {
		if c.Area == "keys" && c.Status != doctor.Fail {
			t.Fatalf("a different master key passed: %+v", c)
		}
	}
}
