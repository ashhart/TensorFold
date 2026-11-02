package main

import (
	"context"
	"flag"
	"fmt"
	"net"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/cli"
	"github.com/ashhart/TensorFold/starlight/internal/connect/setup"
	"github.com/ashhart/TensorFold/starlight/internal/mdns"
)

func bridgeCmd() *cli.Command {
	var port int
	return &cli.Command{Name: "bridge", Summary: "Run the local endpoint for tools that take a base URL and an API key", Usage: "[--port N]",
		Flags: func(fs *flag.FlagSet) {
			fs.IntVar(&port, "port", 0, "port on 127.0.0.1 (default: managed policy, profile, or 7171)")
		},
		Run: func(ctx context.Context, fs *flag.FlagSet) error {
			c, err := load()
			if err != nil {
				return err
			}
			b, err := c.Bridge(ctx, port)
			if err != nil {
				return err
			}
			if err := b.Start(); err != nil {
				return err
			}
			fmt.Fprintf(stdout, "Local endpoint on %s. Give each tool its own key: tensorfold-connect apps add NAME\n", b.URL())
			<-ctx.Done()
			sctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			return b.Shutdown(sctx)
		}}
}

func appsCmd() *cli.Command {
	return &cli.Command{Name: "apps", Summary: "Manage the keys local tools use with the local endpoint", Sub: []*cli.Command{
		{Name: "add", Summary: "Create an app and print its key once", Usage: "NAME",
			Run: func(ctx context.Context, fs *flag.FlagSet) error {
				if fs.NArg() != 1 {
					return cli.ErrUsage
				}
				c, err := load()
				if err != nil {
					return err
				}
				key, err := c.Apps.Add(fs.Arg(0))
				if err != nil {
					return err
				}
				fmt.Fprintln(stdout, key)
				fmt.Fprintf(stderr, "This key is shown once. Use it with base URL %s; Starlight sees requests as you, from app %s.\n",
					c.BridgeURL(0), fs.Arg(0))
				return nil
			}},
		{Name: "list", Summary: "List apps", Usage: "",
			Run: func(ctx context.Context, fs *flag.FlagSet) error {
				c, err := load()
				if err != nil {
					return err
				}
				apps, err := c.Apps.List()
				if err != nil {
					return err
				}
				w := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
				defer w.Flush()
				fmt.Fprintln(w, "APP\tKEY\tCREATED")
				for _, a := range apps {
					fmt.Fprintf(w, "%s\t%s...\t%s\n", a.Name, a.PublicID, a.Created.Local().Format("2006-01-02 15:04"))
				}
				return nil
			}},
		{Name: "remove", Summary: "Remove an app; its key stops working at once", Usage: "NAME",
			Run: func(ctx context.Context, fs *flag.FlagSet) error {
				if fs.NArg() != 1 {
					return cli.ErrUsage
				}
				c, err := load()
				if err != nil {
					return err
				}
				return c.Apps.Remove(fs.Arg(0))
			}},
	}}
}

func envCmd() *cli.Command {
	var app, model, shell string
	var port int
	return &cli.Command{
		Name: "env", Summary: "Print setup for " + strings.Join(setup.Tools(), ", ") + " with a fresh app key",
		Usage: "TOOL [--app NAME] [--model ID] [--shell posix|powershell] [--port N]",
		Flags: func(fs *flag.FlagSet) {
			fs.StringVar(&app, "app", "", "app name for the key (default: the tool name)")
			fs.StringVar(&model, "model", "", "model id to preselect, from tensorfold-connect models")
			fs.StringVar(&shell, "shell", setup.DefaultShell(), "posix or powershell")
			fs.IntVar(&port, "port", 0, "local endpoint port (default: the configured port)")
		},
		Run: func(ctx context.Context, fs *flag.FlagSet) error {
			args, err := positional(fs)
			if err != nil {
				return err
			}
			if len(args) != 1 {
				return cli.ErrUsage
			}
			c, err := load()
			if err != nil {
				return err
			}
			if c.Settings.BridgeDisabled {
				return fmt.Errorf("your organisation has turned off the local endpoint")
			}
			text, replaced, err := c.Setup(args[0], app, model, shell, port)
			if err != nil {
				return err
			}
			fmt.Fprint(stdout, text)
			if replaced {
				fmt.Fprintln(stderr, "A new key replaced this app's previous key; update anything that used the old one.")
			}
			if !c.BridgeRunning(ctx, port) {
				fmt.Fprintln(stderr, "Start the local endpoint with `tensorfold-connect bridge` or the tray app.")
			}
			return nil
		},
	}
}

func discoverCmd() *cli.Command {
	var timeout time.Duration
	return &cli.Command{Name: "discover", Summary: "Look for Starlight servers advertised on this network", Usage: "[--timeout 3s]",
		Flags: func(fs *flag.FlagSet) {
			fs.DurationVar(&timeout, "timeout", 3*time.Second, "how long to listen for answers")
		},
		Run: func(ctx context.Context, fs *flag.FlagSet) error {
			found, err := mdns.Browse(ctx, timeout)
			if err != nil {
				return fmt.Errorf("local network discovery failed: %w", err)
			}
			if len(found) == 0 {
				fmt.Fprintln(stdout, "No Starlight servers answered on this network. Ask your IT team for the address.")
				return nil
			}
			w := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
			fmt.Fprintln(w, "NAME\tADDRESS\tID\tANSWERED FROM")
			for _, f := range found {
				var from []string
				for _, a := range f.Addrs {
					from = append(from, net.JoinHostPort(a.String(), fmt.Sprint(f.Port)))
				}
				fmt.Fprintf(w, "%s\t%s\t%s\t%s\n", f.Instance, f.URL, f.InstallationID, strings.Join(from, " "))
			}
			w.Flush()
			fmt.Fprintln(stdout, "\nAnyone on this network can advertise a server. Confirm the address with your IT team, then run:")
			fmt.Fprintf(stdout, "  tensorfold-connect login --server %s\n", found[0].URL)
			if c, err := load(); err == nil && c.Settings.ServerLocked {
				fmt.Fprintf(stderr, "Your organisation already sets the server to %s.\n", c.Settings.Server)
			}
			return nil
		}}
}
