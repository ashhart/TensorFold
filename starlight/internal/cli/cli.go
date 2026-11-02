// Package cli is a small subcommand dispatcher on top of the standard flag package.
package cli

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"sort"
	"strings"
	"syscall"
)

// Command is one verb; a command either runs or holds subcommands.
type Command struct {
	Name    string
	Summary string
	Usage   string
	Flags   func(fs *flag.FlagSet)
	Run     func(ctx context.Context, fs *flag.FlagSet) error
	Sub     []*Command
}

// ErrUsage makes Main print usage and exit 2.
var ErrUsage = errors.New("usage")

// Main dispatches os.Args and exits with the command's status.
func Main(root *Command) {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	err := Dispatch(ctx, root, os.Args[1:], os.Stdout, os.Stderr)
	stop()
	switch {
	case err == nil:
	case errors.Is(err, ErrUsage), errors.Is(err, flag.ErrHelp):
		os.Exit(2)
	default:
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

// Dispatch walks args to a command and runs it.
func Dispatch(ctx context.Context, cmd *Command, args []string, stdout, stderr io.Writer) error {
	path := cmd.Name
	for len(cmd.Sub) > 0 {
		if len(args) == 0 || args[0] == "-h" || args[0] == "--help" || args[0] == "help" {
			help(stderr, path, cmd)
			return ErrUsage
		}
		next := find(cmd, args[0])
		if next == nil {
			fmt.Fprintf(stderr, "unknown command %q\n\n", args[0])
			help(stderr, path, cmd)
			return ErrUsage
		}
		cmd, args, path = next, args[1:], path+" "+next.Name
	}
	fs := flag.NewFlagSet(path, flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() {
		fmt.Fprintf(stderr, "usage: %s %s\n\n%s\n\n", path, cmd.Usage, cmd.Summary)
		fs.PrintDefaults()
	}
	if cmd.Flags != nil {
		cmd.Flags(fs)
	}
	if err := fs.Parse(args); err != nil {
		return err
	}
	if cmd.Run == nil {
		return ErrUsage
	}
	err := cmd.Run(ctx, fs)
	if errors.Is(err, ErrUsage) {
		fs.Usage()
	}
	return err
}

func find(cmd *Command, name string) *Command {
	for _, s := range cmd.Sub {
		if s.Name == name {
			return s
		}
	}
	return nil
}

func help(w io.Writer, path string, cmd *Command) {
	fmt.Fprintf(w, "usage: %s <command> [flags]\n\n", path)
	if cmd.Summary != "" {
		fmt.Fprintf(w, "%s\n\n", cmd.Summary)
	}
	subs := append([]*Command(nil), cmd.Sub...)
	sort.Slice(subs, func(i, j int) bool { return subs[i].Name < subs[j].Name })
	width := 0
	for _, s := range subs {
		width = max(width, len(s.Name))
	}
	for _, s := range subs {
		fmt.Fprintf(w, "  %s%s  %s\n", s.Name, strings.Repeat(" ", width-len(s.Name)), s.Summary)
	}
}
