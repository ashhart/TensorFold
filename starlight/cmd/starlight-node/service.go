package main

import (
	"context"
	"flag"
	"fmt"
	"runtime"

	"github.com/ashhart/TensorFold/starlight/internal/cli"
	"github.com/ashhart/TensorFold/starlight/internal/node/service"
)

func serviceCommand() *cli.Command {
	var o service.Options
	var scope string
	return &cli.Command{Name: "service", Summary: "Install, remove or inspect the system service.", Sub: []*cli.Command{
		{Name: "install", Summary: "Install and start the service; flags after -- are passed to `run`.",
			Usage: "[--user NAME] [--scope system|user] [--state-dir DIR] [--dry-run] [-- run flags]",
			Flags: func(fs *flag.FlagSet) {
				stateFlag(fs, &o.StateDir)
				defUser := "starlight-node"
				switch runtime.GOOS {
				case "darwin":
					defUser = "_tensorfold"
				case "windows":
					defUser = ""
				}
				fs.StringVar(&o.User, "user", defUser, "unprivileged account the service runs as (not used on Windows)")
				fs.StringVar(&o.Scope, "scope", "system", "system service, or user (macOS LaunchAgent)")
				fs.StringVar(&o.Binary, "binary", "", "agent binary path (default: this executable)")
				fs.BoolVar(&o.DryRun, "dry-run", false, "print the service definition without installing it")
			},
			Run: func(_ context.Context, fs *flag.FlagSet) error {
				o.Args = service.RunArgs(o.StateDir, fs.Args())
				msg, err := service.Install(o)
				if err != nil {
					return err
				}
				fmt.Println(msg)
				return nil
			},
		},
		{Name: "uninstall", Summary: "Stop and remove the service; state and models stay.",
			Flags: func(fs *flag.FlagSet) { fs.StringVar(&scope, "scope", "system", "system or user") },
			Run: func(context.Context, *flag.FlagSet) error {
				return service.Uninstall(scope)
			},
		},
		{Name: "status", Summary: "Show the service manager's state.",
			Flags: func(fs *flag.FlagSet) { fs.StringVar(&scope, "scope", "system", "system or user") },
			Run: func(context.Context, *flag.FlagSet) error {
				s, err := service.Status(scope)
				if err != nil {
					return err
				}
				fmt.Println(s)
				return nil
			},
		},
	}}
}
