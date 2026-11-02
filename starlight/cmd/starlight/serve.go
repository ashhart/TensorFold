package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/jackc/pgx/v5"

	"github.com/ashhart/TensorFold/starlight/internal/cli"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/keyring"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/server"
)

func serveCmd() *cli.Command {
	var migrate bool
	return &cli.Command{Name: "serve", Summary: "Run Starlight with the roles in its configuration", Usage: "[--config starlight.yaml] [--migrate]",
		Flags: func(fs *flag.FlagSet) {
			withConfig(fs)
			fs.BoolVar(&migrate, "migrate", false, "apply pending schema migrations before starting")
		},
		Run: func(ctx context.Context, _ *flag.FlagSet) error {
			cfg, err := loadConfig()
			if err != nil {
				return err
			}
			log := logger(cfg)
			if migrate {
				if err := runMigrations(ctx); err != nil {
					return err
				}
			}
			t, err := server.Open(ctx, cfg, log)
			if err != nil {
				return err
			}
			defer t.Close()
			return t.Run(ctx)
		}}
}

func runMigrations(ctx context.Context) error {
	_, pool, err := openDB(ctx)
	if err != nil {
		return err
	}
	defer pool.Close()
	applied, err := db.Migrate(ctx, pool)
	for _, m := range applied {
		fmt.Fprintln(os.Stderr, "applied", m)
	}
	return err
}

func migrateCmd() *cli.Command {
	return &cli.Command{Name: "migrate", Summary: "Apply pending schema migrations", Usage: "[--config starlight.yaml]", Flags: withConfig,
		Run: func(ctx context.Context, _ *flag.FlagSet) error { return runMigrations(ctx) }}
}

func initCmd() *cli.Command {
	var org, keyOut string
	var offline, noGeneral bool
	return &cli.Command{Name: "init", Summary: "Initialise a new installation and print the first-owner setup link",
		Usage: "[--config starlight.yaml] [--org NAME] [--offline] [--write-master-key PATH]",
		Flags: func(fs *flag.FlagSet) {
			withConfig(fs)
			fs.StringVar(&org, "org", "", "organisation name shown to users")
			fs.BoolVar(&offline, "offline", false, "forbid all cloud provider egress")
			fs.BoolVar(&noGeneral, "no-general-project", false, "do not create the everyone project")
			fs.StringVar(&keyOut, "write-master-key", "", "generate a master key and write it to this new file (0600)")
		},
		Run: func(ctx context.Context, _ *flag.FlagSet) error {
			if keyOut != "" {
				if err := writeNewKey(keyOut); err != nil {
					return err
				}
				if os.Getenv("STARLIGHT_MASTER_KEY") == "" {
					os.Setenv("STARLIGHT_MASTER_KEY_FILE", keyOut)
				}
			}
			if err := runMigrations(ctx); err != nil {
				return err
			}
			cfg, pool, err := openDB(ctx)
			if err != nil {
				return err
			}
			defer pool.Close()
			master, err := keyring.LoadMaster(cfg.Keys.MasterKey, cfg.Keys.MasterKeyFile)
			if err != nil {
				return err
			}
			keys := keyring.New(master)
			var inst server.Installation
			var code string
			err = db.Tx(ctx, pool, func(tx pgx.Tx) error {
				inst, code, err = server.Init(ctx, tx, keys, server.InitOptions{OrgName: org, Offline: offline, GeneralProject: !noGeneral})
				return err
			})
			if err != nil {
				return err
			}
			fmt.Printf("Initialised %s (installation %s, trust domain %s).\n", inst.OrgName, inst.ID, inst.TrustDomain)
			fmt.Printf("Open this link within 24 hours to connect your identity provider and become the first owner:\n\n  %s/auth/setup?code=%s\n\n",
				strings.TrimRight(cfg.PublicURL, "/"), code)
			if keyOut != "" {
				fmt.Printf("The master key is in %s. Store it in your secret manager; without it, sealed secrets cannot be read.\n", keyOut)
			}
			return nil
		}}
}

func writeNewKey(path string) error {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if errors.Is(err, os.ErrExist) {
		return fmt.Errorf("%s already exists; refusing to overwrite a master key", path)
	}
	if err != nil {
		return err
	}
	if _, err := f.WriteString(keyring.GenerateMasterKey() + "\n"); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
