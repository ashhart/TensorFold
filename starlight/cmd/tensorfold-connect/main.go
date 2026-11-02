// Command tensorfold-connect signs in to TensorFold Starlight and connects local tools to it.
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/ashhart/TensorFold/starlight/internal/buildinfo"
	"github.com/ashhart/TensorFold/starlight/internal/cli"
	"github.com/ashhart/TensorFold/starlight/internal/connect/core"
	"github.com/ashhart/TensorFold/starlight/internal/connect/tray"
)

var (
	stdout io.Writer = os.Stdout
	stderr io.Writer = os.Stderr
	load             = core.Load
)

func main() { cli.Main(root()) }

func root() *cli.Command {
	return &cli.Command{
		Name:    "tensorfold-connect",
		Summary: "Sign in to your organisation's TensorFold Starlight and connect local tools to it.",
		Sub: []*cli.Command{loginCmd(), logoutCmd(), statusCmd(), modelsCmd(), projectsCmd(), bridgeCmd(), appsCmd(), envCmd(),
			discoverCmd(), trayCmd(), versionCmd()},
	}
}

func trayCmd() *cli.Command {
	return &cli.Command{Name: "tray", Summary: "Run the menu-bar or system-tray app, with the local endpoint", Usage: "",
		Run: func(ctx context.Context, fs *flag.FlagSet) error {
			c, err := load()
			if err != nil {
				return err
			}
			return tray.Run(ctx, c)
		}}
}

func versionCmd() *cli.Command {
	return &cli.Command{Name: "version", Summary: "Print the version", Usage: "",
		Run: func(ctx context.Context, fs *flag.FlagSet) error {
			fmt.Fprintln(stdout, "tensorfold-connect", buildinfo.String())
			return nil
		}}
}

// positional collects arguments while still parsing flags that follow them, so "env openai --port 7300" works.
func positional(fs *flag.FlagSet) ([]string, error) {
	var out []string
	args := fs.Args()
	for len(args) > 0 {
		switch {
		case args[0] == "--":
			return append(out, args[1:]...), nil
		case strings.HasPrefix(args[0], "-") && args[0] != "-":
			if err := fs.Parse(args); err != nil {
				return nil, err
			}
			args = fs.Args()
		default:
			out = append(out, args[0])
			args = args[1:]
		}
	}
	return out, nil
}
