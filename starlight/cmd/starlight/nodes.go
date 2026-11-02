package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"os"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/admin/fleet"
	"github.com/ashhart/TensorFold/starlight/internal/audit"
	"github.com/ashhart/TensorFold/starlight/internal/cli"
	"github.com/ashhart/TensorFold/starlight/internal/control"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/pki"
)

type labels map[string]string

func (l labels) String() string { return fmt.Sprint(map[string]string(l)) }

func (l labels) Set(v string) error {
	k, val, ok := strings.Cut(v, "=")
	if !ok || k == "" {
		return errors.New("labels are key=value")
	}
	l[k] = val
	return nil
}

// operator attributes command-line changes to the shell user on this host.
func operator() audit.Event {
	return audit.Event{ActorType: "operator", ActorID: hostUser()}
}

func nodesCmd() *cli.Command {
	var spec control.TokenSpec
	lbl := labels{}
	return &cli.Command{Name: "nodes", Summary: "Enrol, approve, drain and revoke nodes", Sub: []*cli.Command{
		{Name: "token", Summary: "Create an enrolment token and print the enrol command",
			Usage: "[--pool default] [--label k=v] [--max-class confidential] [--auto-approve] [--uses 1] [--ttl 24h]",
			Flags: func(fs *flag.FlagSet) {
				withConfig(fs)
				fs.StringVar(&spec.Pool, "pool", "default", "pool the node joins")
				fs.Var(lbl, "label", "node label key=value (repeatable)")
				fs.StringVar(&spec.MaxClass, "max-class", "confidential", "highest data classification the node may process")
				fs.BoolVar(&spec.AutoApprove, "auto-approve", false, "activate nodes without manual approval")
				fs.IntVar(&spec.MaxUses, "uses", 1, "how many nodes may enrol with this token")
				fs.DurationVar(&spec.TTL, "ttl", 24*time.Hour, "token lifetime")
			},
			Run: func(ctx context.Context, _ *flag.FlagSet) error {
				spec.Labels = lbl
				return createToken(ctx, spec)
			}},
		{Name: "list", Summary: "List nodes with their state and load", Flags: withConfig, Run: listNodes},
		{Name: "approve", Summary: "Activate a pending node", Usage: "NAME", Flags: withConfig, Run: nodeAction(fleet.NodeApprove)},
		{Name: "online", Summary: "Bring a drained or disabled node back into scheduling", Usage: "NAME", Flags: withConfig,
			Run: nodeAction(fleet.NodeOnline)},
		{Name: "drain", Summary: "Stop scheduling on a node; running replicas move elsewhere", Usage: "NAME", Flags: withConfig,
			Run: nodeAction(fleet.NodeDrain)},
		{Name: "disable", Summary: "Take a node offline without revoking it", Usage: "NAME", Flags: withConfig, Run: nodeAction(fleet.NodeDisable)},
		{Name: "revoke", Summary: "Revoke a node and its certificates immediately", Usage: "NAME", Flags: withConfig, Run: nodeAction(fleet.NodeRevoke)},
	}}
}

func createToken(ctx context.Context, spec control.TokenSpec) error {
	cfg, pool, err := openDB(ctx)
	if err != nil {
		return err
	}
	defer pool.Close()
	var certPEM string
	if err := pool.QueryRow(ctx, "SELECT cert_pem FROM pki WHERE name = 'workload-ca'").Scan(&certPEM); err != nil {
		return err
	}
	ca, err := pki.ParseCertPEM([]byte(certPEM))
	if err != nil {
		return err
	}
	var tok string
	svc := &fleet.Service{Pool: pool}
	if err := svc.Do(ctx, operator(), func(tx *fleet.Tx) (err error) { tok, err = tx.CreateEnrolmentToken(ctx, spec); return err }); err != nil {
		return err
	}
	sum := sha256.Sum256(ca.Raw)
	nodeURL := cfg.NodeURL
	if nodeURL == "" {
		nodeURL = "https://<starlight-host>:9443"
	}
	fmt.Printf("starlight-node enroll --server %s --token %s --ca-sha256 %s\n", nodeURL, tok, hex.EncodeToString(sum[:]))
	return nil
}

func listNodes(ctx context.Context, _ *flag.FlagSet) error {
	_, pool, err := openDB(ctx)
	if err != nil {
		return err
	}
	defer pool.Close()
	nodes, err := fleet.Nodes(ctx, pool)
	if err != nil {
		return err
	}
	w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "NAME\tSTATE\tPOOL\tBACKEND\tACCELERATORS\tREPLICAS\tTOK/S\tLAST SEEN")
	for _, n := range nodes {
		r := n.Row()
		tps, seen := "-", "never"
		if r.Fresh {
			tps = fleet.Rate(r.TokensPerSecond)
		}
		if n.LastSeen != nil {
			seen = n.LastSeen.Local().Format("2006-01-02 15:04:05")
		}
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\t%d/%d\t%s\t%s\n", n.Name, r.State, n.Pool, r.Backend, r.Accelerators, r.Ready, r.Replicas, tps, seen)
	}
	return w.Flush()
}

func nodeAction(action string) func(context.Context, *flag.FlagSet) error {
	return func(ctx context.Context, fs *flag.FlagSet) error {
		if fs.NArg() != 1 {
			return cli.ErrUsage
		}
		_, pool, err := openDB(ctx)
		if err != nil {
			return err
		}
		defer pool.Close()
		svc := &fleet.Service{Pool: pool}
		err = svc.Do(ctx, operator(), func(tx *fleet.Tx) error { _, err := tx.SetNodeStatus(ctx, fs.Arg(0), action); return err })
		var inv fleet.Invalid
		switch {
		case errors.Is(err, fleet.ErrNotFound):
			return fmt.Errorf("no node named %s", fs.Arg(0))
		case errors.As(err, &inv):
			return errors.New(string(inv))
		case err != nil:
			return err
		}
		fmt.Printf("%s: %s done\n", fs.Arg(0), action)
		return nil
	}
}
