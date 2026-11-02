package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"text/tabwriter"

	"github.com/ashhart/TensorFold/starlight/internal/cli"
	"github.com/ashhart/TensorFold/starlight/internal/doctor"
	"github.com/ashhart/TensorFold/starlight/internal/server"
)

func doctorCmd() *cli.Command {
	return &cli.Command{Name: "doctor", Summary: "Check configuration, database, keys, fleet and identity providers", Flags: withConfig,
		Run: func(ctx context.Context, _ *flag.FlagSet) error {
			cfg, pool, err := openDB(ctx)
			if err != nil {
				return err
			}
			defer pool.Close()
			client, err := server.IDPClient(cfg.Identity.CAFile)
			if err != nil {
				return err
			}
			checks := (&doctor.Doctor{Config: cfg, Pool: pool, HTTP: client}).Run(ctx)
			w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
			failed := false
			for _, c := range checks {
				fmt.Fprintf(w, "%s\t%s\t%s\n", c.Status, c.Area, c.Detail)
				if c.Hint != "" && c.Status != doctor.OK {
					fmt.Fprintf(w, "\t\t  %s\n", c.Hint)
				}
				failed = failed || c.Status == doctor.Fail
			}
			_ = w.Flush()
			if failed {
				return fmt.Errorf("doctor found problems")
			}
			return nil
		}}
}
