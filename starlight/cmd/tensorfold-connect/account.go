package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"strings"
	"text/tabwriter"

	"github.com/ashhart/TensorFold/starlight/internal/cli"
	"github.com/ashhart/TensorFold/starlight/internal/connect/core"
	"github.com/ashhart/TensorFold/starlight/internal/connect/tokens"
	"github.com/ashhart/TensorFold/starlight/internal/discovery"
)

func loginCmd() *cli.Command {
	var server, project string
	var device bool
	return &cli.Command{
		Name: "login", Summary: "Sign in to Starlight in your browser, or with a code on another device",
		Usage: "[--server URL] [--device] [--project SLUG]",
		Flags: func(fs *flag.FlagSet) {
			fs.StringVar(&server, "server", "", "Starlight address, for example https://starlight.example")
			fs.BoolVar(&device, "device", false, "sign in by entering a code in a browser on any device")
			fs.StringVar(&project, "project", "", "project to act in by default")
		},
		Run: func(ctx context.Context, fs *flag.FlagSet) error {
			c, err := load()
			if err != nil {
				return err
			}
			if server != "" {
				if err := c.UseServer(server); err != nil {
					return err
				}
			}
			me, err := c.Login(ctx, core.LoginOptions{Device: device, Out: stderr})
			if err != nil {
				return err
			}
			if err := chooseProject(c, me, project); err != nil {
				return err
			}
			fmt.Fprintf(stdout, "Signed in to %s as %s.\n", me.OrgName, me.Email)
			if c.Settings.Project != "" {
				fmt.Fprintf(stdout, "Requests act in project %s.\n", c.Settings.Project)
			}
			return nil
		},
	}
}

func chooseProject(c *core.Core, me discovery.Me, want string) error {
	if want == "" {
		if c.Settings.Project != "" {
			return nil
		}
		for _, p := range me.Projects {
			if p.Default {
				return c.UseProject(p.Slug)
			}
		}
		return nil
	}
	for _, p := range me.Projects {
		if p.Slug == want {
			return c.UseProject(want)
		}
	}
	return fmt.Errorf("you are not a member of project %q; run `tensorfold-connect projects` to list yours", want)
}

func logoutCmd() *cli.Command {
	return &cli.Command{Name: "logout", Summary: "Sign out and revoke this device's Starlight sign-in", Usage: "",
		Run: func(ctx context.Context, fs *flag.FlagSet) error {
			c, err := load()
			if err != nil {
				return err
			}
			if err := c.Logout(ctx); err != nil {
				return err
			}
			fmt.Fprintln(stdout, "Signed out.")
			return nil
		}}
}

func statusCmd() *cli.Command {
	return &cli.Command{Name: "status", Summary: "Show the server, sign-in, project and local endpoint", Usage: "",
		Run: func(ctx context.Context, fs *flag.FlagSet) error {
			c, err := load()
			if err != nil {
				return err
			}
			w := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
			defer w.Flush()
			server, err := c.Server()
			if err != nil {
				fmt.Fprintln(w, "Server\tnot configured")
				return nil
			}
			note := ""
			if c.Settings.ServerLocked {
				note = " (set by your organisation)"
			}
			fmt.Fprintf(w, "Server\t%s%s\n", server, note)
			fmt.Fprintf(w, "Credentials\t%s\n", c.Store.Backend())
			api, err := c.API(ctx)
			if err != nil {
				fmt.Fprintf(w, "Starlight\tunreachable: %v\n", err)
			} else if me, err := api.Me(ctx); errors.Is(err, tokens.ErrSignedOut) {
				fmt.Fprintln(w, "Signed in as\tnobody; run tensorfold-connect login")
			} else if err != nil {
				fmt.Fprintf(w, "Signed in as\tunknown: %v\n", err)
			} else {
				fmt.Fprintf(w, "Organisation\t%s\nSigned in as\t%s\n", me.OrgName, me.Email)
			}
			if c.Settings.Project != "" {
				fmt.Fprintf(w, "Project\t%s\n", c.Settings.Project)
			}
			endpoint := "not running"
			switch {
			case c.Settings.BridgeDisabled:
				endpoint = "turned off by your organisation"
			case c.BridgeRunning(ctx, 0):
				endpoint = "running"
			}
			fmt.Fprintf(w, "Local endpoint\t%s (%s)\n", c.BridgeURL(0), endpoint)
			if p := c.Settings.Policy; len(p.Sources) > 0 {
				fmt.Fprintf(w, "Managed by\t%s\n", strings.Join(p.Sources, ", "))
				if len(p.Unknown) > 0 {
					fmt.Fprintf(w, "Ignored policy keys\t%s\n", strings.Join(p.Unknown, ", "))
				}
			}
			return nil
		}}
}

func modelsCmd() *cli.Command {
	var project string
	return &cli.Command{Name: "models", Summary: "List the models you may use", Usage: "[--project SLUG]",
		Flags: func(fs *flag.FlagSet) { fs.StringVar(&project, "project", "", "project to list models for") },
		Run: func(ctx context.Context, fs *flag.FlagSet) error {
			c, err := load()
			if err != nil {
				return err
			}
			api, err := c.API(ctx)
			if err != nil {
				return err
			}
			if project == "" {
				project = c.Settings.Project
			}
			models, err := api.Models(ctx, project)
			if err != nil {
				return err
			}
			if len(models) == 0 {
				fmt.Fprintln(stdout, "No models are available to you in this project.")
				return nil
			}
			w := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
			defer w.Flush()
			fmt.Fprintln(w, "MODEL\tOWNED BY")
			for _, m := range models {
				fmt.Fprintf(w, "%s\t%s\n", m.ID, m.OwnedBy)
			}
			return nil
		}}
}

func projectsCmd() *cli.Command {
	var use string
	return &cli.Command{Name: "projects", Summary: "List your projects, or choose the default one", Usage: "[--use SLUG]",
		Flags: func(fs *flag.FlagSet) { fs.StringVar(&use, "use", "", "make this project the default for requests") },
		Run: func(ctx context.Context, fs *flag.FlagSet) error {
			c, err := load()
			if err != nil {
				return err
			}
			api, err := c.API(ctx)
			if err != nil {
				return err
			}
			me, err := api.Me(ctx)
			if err != nil {
				return err
			}
			if use != "" {
				if err := chooseProject(c, me, use); err != nil {
					return err
				}
				fmt.Fprintf(stdout, "Requests now act in project %s.\n", use)
				return nil
			}
			w := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
			defer w.Flush()
			fmt.Fprintln(w, "\tPROJECT\tNAME\tROLE\tCLASSIFICATION")
			for _, p := range me.Projects {
				mark := ""
				if p.Slug == c.Settings.Project {
					mark = "*"
				}
				fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\n", mark, p.Slug, p.Name, p.Role, p.Classification)
			}
			return nil
		}}
}
