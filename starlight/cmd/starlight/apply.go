package main

import (
	"context"
	"flag"
	"os"

	"github.com/ashhart/TensorFold/starlight/internal/admin/fleet"
	"github.com/ashhart/TensorFold/starlight/internal/cli"
	"github.com/ashhart/TensorFold/starlight/internal/declare"
	"github.com/ashhart/TensorFold/starlight/internal/server"
)

func applyCmd() *cli.Command {
	var file string
	var dryRun bool
	return &cli.Command{Name: "apply", Summary: "Make Starlight match a fleet file in one transaction; --dry-run shows the changes only",
		Usage: "-f fleet-file [--dry-run]",
		Flags: func(fs *flag.FlagSet) {
			withConfig(fs)
			fs.StringVar(&file, "f", "", "fleet file to apply, such as deploy/examples/fleet.yaml")
			fs.BoolVar(&dryRun, "dry-run", false, "print what would change and write nothing")
		},
		Run: func(ctx context.Context, _ *flag.FlagSet) error {
			if file == "" {
				return cli.ErrUsage
			}
			f, err := declare.LoadFile(file)
			if err != nil {
				return err
			}
			cfg, pool, err := openDB(ctx)
			if err != nil {
				return err
			}
			defer pool.Close()
			keys, err := server.LoadKeyring(ctx, pool, cfg.Keys)
			if err != nil {
				return err
			}
			res, err := declare.Apply(ctx, &fleet.Service{Pool: pool, Keys: keys}, f, declare.Options{DryRun: dryRun, Actor: operator()})
			if err != nil {
				return err
			}
			return res.Write(os.Stdout)
		}}
}
