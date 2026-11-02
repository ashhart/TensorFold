package declare

import (
	"context"
	"errors"
	"fmt"
	"io"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/ashhart/TensorFold/starlight/internal/admin/fleet"
	"github.com/ashhart/TensorFold/starlight/internal/audit"
)

// Options control an apply; Secrets resolves credential references and defaults to Reference.
type Options struct {
	DryRun  bool
	Actor   audit.Event
	Secrets func(ref string) (string, error)
}

// Change is one difference between the file and Starlight; Detail never holds a secret.
type Change struct {
	Op     string
	Kind   string
	Name   string
	Detail []string
}

func (c Change) String() string {
	sign := "+"
	if c.Op == "update" {
		sign = "~"
	}
	s := sign + " " + c.Kind
	if c.Name != "" {
		s += " " + c.Name
	}
	if len(c.Detail) > 0 {
		s += ": " + strings.Join(c.Detail, "; ")
	}
	return s
}

// Result lists what an apply changed, or would change on a dry run.
type Result struct {
	Changes []Change
	DryRun  bool
}

// Write prints the changes as a diff with a closing summary.
func (r Result) Write(w io.Writer) error {
	for _, c := range r.Changes {
		if _, err := fmt.Fprintln(w, c); err != nil {
			return err
		}
	}
	var err error
	switch {
	case len(r.Changes) == 0:
		_, err = fmt.Fprintln(w, "No changes: Starlight already matches the file.")
	case r.DryRun:
		_, err = fmt.Fprintf(w, "Dry run: %d changes would be applied; nothing was written.\n", len(r.Changes))
	default:
		_, err = fmt.Fprintf(w, "Applied %d changes in one transaction.\n", len(r.Changes))
	}
	return err
}

var errDryRun = errors.New("declare: dry run")

// Apply makes Starlight match the file in one transaction; a dry run computes the same changes and then rolls back.
func Apply(ctx context.Context, svc *fleet.Service, f *File, o Options) (Result, error) {
	if o.Secrets == nil {
		o.Secrets = Reference
	}
	secrets := map[string]string{}
	for _, p := range f.Providers {
		if p.Credential == "" {
			continue
		}
		v, err := o.Secrets(p.Credential)
		if err != nil {
			return Result{}, fmt.Errorf("provider %s: %w", p.Name, err)
		}
		secrets[p.Name] = v
	}
	if o.Actor.ActorType == "" {
		o.Actor = audit.Event{ActorType: "operator", ActorID: "starlight-apply"}
	}
	res := Result{DryRun: o.DryRun}
	err := svc.Do(ctx, o.Actor, func(tx *fleet.Tx) error {
		a := &applier{ctx: ctx, tx: tx, secrets: secrets}
		for _, step := range []func(*File) error{a.settings, a.providers, a.models, a.projects, a.orgLimits, a.orgRoles} {
			if err := step(f); err != nil {
				return err
			}
		}
		res.Changes = a.changes
		if o.DryRun {
			return errDryRun
		}
		return nil
	})
	if errors.Is(err, errDryRun) {
		err = nil
	}
	return res, err
}

// Reference reads a credential reference: env:NAME from the environment or file:/absolute/path from disk.
func Reference(ref string) (string, error) {
	kind, rest, _ := strings.Cut(ref, ":")
	switch kind {
	case "env":
		v := strings.TrimSpace(os.Getenv(rest))
		if rest == "" || v == "" {
			return "", fmt.Errorf("environment variable %q is not set", rest)
		}
		return v, nil
	case "file":
		if !filepath.IsAbs(rest) {
			return "", errors.New("credential files must be absolute paths")
		}
		b, err := os.ReadFile(rest)
		if err != nil {
			return "", err
		}
		if v := strings.TrimSpace(string(b)); v != "" {
			return v, nil
		}
		return "", fmt.Errorf("credential file %s is empty", rest)
	}
	return "", errors.New("credential must be a reference, env:NAME or file:/path; inline secrets are refused")
}

type applier struct {
	ctx     context.Context
	tx      *fleet.Tx
	secrets map[string]string
	changes []Change
}

func (a *applier) create(kind, name string, detail ...string) {
	a.changes = append(a.changes, Change{Op: "create", Kind: kind, Name: name, Detail: detail})
}

func (a *applier) update(kind, name string, detail ...string) {
	a.changes = append(a.changes, Change{Op: "update", Kind: kind, Name: name, Detail: detail})
}

// diff collects readable before and after descriptions of the fields that differ.
type diff []string

func field[T comparable](d *diff, name string, before, after T) {
	if before != after {
		*d = append(*d, fmt.Sprintf("%s %s → %s", name, show(before), show(after)))
	}
}

func fieldMap(d *diff, name string, before, after map[string]string) {
	if !maps.Equal(before, after) {
		*d = append(*d, fmt.Sprintf("%s %s → %s", name, pairs(before), pairs(after)))
	}
}

func fieldList(d *diff, name string, before, after []string) {
	if !slices.Equal(before, after) {
		*d = append(*d, fmt.Sprintf("%s [%s] → [%s]", name, strings.Join(before, ", "), strings.Join(after, ", ")))
	}
}

func fieldCap(d *diff, name string, before, after *int) {
	if (before == nil) != (after == nil) || before != nil && *before != *after {
		*d = append(*d, fmt.Sprintf("%s %s → %s", name, capText(before), capText(after)))
	}
}

func show(v any) string {
	if s, ok := v.(string); ok {
		return fmt.Sprintf("%q", s)
	}
	return fmt.Sprint(v)
}

func capText(v *int) string {
	if v == nil {
		return "none"
	}
	return fmt.Sprint(*v)
}

func pairs(m map[string]string) string {
	var out []string
	for _, k := range slices.Sorted(maps.Keys(m)) {
		out = append(out, k+"="+m[k])
	}
	return "{" + strings.Join(out, ", ") + "}"
}

func notFound(err error) bool { return errors.Is(err, fleet.ErrNotFound) }
