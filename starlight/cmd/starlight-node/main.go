// Command starlight-node runs TensorFold engines on one machine under the direction of TensorFold Starlight.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"strings"

	"github.com/ashhart/TensorFold/starlight/internal/buildinfo"
	"github.com/ashhart/TensorFold/starlight/internal/cli"
	"github.com/ashhart/TensorFold/starlight/internal/node/service"
	"github.com/ashhart/TensorFold/starlight/internal/node/state"
)

func main() {
	root := &cli.Command{Name: "starlight-node", Summary: "Run TensorFold engines for TensorFold Starlight.", Sub: []*cli.Command{
		enrollCommand(), runCommand(), statusCommand(), serviceCommand(),
		{Name: "version", Summary: "Print the agent version.", Run: func(context.Context, *flag.FlagSet) error {
			fmt.Println("starlight-node", buildinfo.String())
			return nil
		}},
	}}
	if service.IsService() {
		err := service.RunService(func(ctx context.Context) error {
			return cli.Dispatch(ctx, root, os.Args[1:], os.Stdout, os.Stderr)
		})
		if err != nil {
			os.Exit(1)
		}
		return
	}
	cli.Main(root)
}

func stateFlag(fs *flag.FlagSet, dst *string) {
	fs.StringVar(dst, "state-dir", envOr("STARLIGHT_NODE_STATE_DIR", state.DefaultRoot()), "private state directory")
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func newLogger(w io.Writer, level, format string) *slog.Logger {
	var lv slog.Level
	if err := lv.UnmarshalText([]byte(level)); err != nil {
		lv = slog.LevelInfo
	}
	opts := &slog.HandlerOptions{Level: lv}
	if strings.EqualFold(format, "text") {
		return slog.New(slog.NewTextHandler(w, opts))
	}
	return slog.New(slog.NewJSONHandler(w, opts))
}

func statusCommand() *cli.Command {
	var dir string
	var asJSON bool
	return &cli.Command{Name: "status", Summary: "Show this node's identity and the running agent's last report.",
		Flags: func(fs *flag.FlagSet) {
			stateFlag(fs, &dir)
			fs.BoolVar(&asJSON, "json", false, "print the raw status document")
		},
		Run: func(context.Context, *flag.FlagSet) error {
			d, err := state.Open(dir)
			if err != nil {
				return err
			}
			id, err := d.LoadIdentity()
			if err != nil {
				return err
			}
			raw, statusErr := d.ReadStatus()
			if asJSON {
				out := map[string]any{"identity": id}
				if statusErr == nil {
					out["status"] = json.RawMessage(raw)
				}
				enc := json.NewEncoder(os.Stdout)
				enc.SetIndent("", "  ")
				return enc.Encode(out)
			}
			fmt.Printf("node      %s (%s)\nserver    %s\ntrust     %s\nenrolled  %s\n", id.Name, id.NodeID, id.ServerURL, id.Trust,
				id.EnrolledAt.Format("2006-01-02 15:04 MST"))
			if statusErr != nil {
				fmt.Println("agent     no status yet; is `starlight-node run` running?")
				return nil
			}
			return printStatus(raw)
		},
	}
}

func printStatus(raw []byte) error {
	var st struct {
		NodeStatus   string `json:"node_status"`
		Generation   int64  `json:"generation"`
		Offline      bool   `json:"offline"`
		LastSync     string `json:"last_sync"`
		LastError    string `json:"last_error"`
		CertNotAfter string `json:"cert_not_after"`
		Replicas     []struct {
			ReplicaID string `json:"replica_id"`
			State     string `json:"state"`
			Error     string `json:"error"`
			InFlight  int    `json:"in_flight"`
		} `json:"replicas"`
	}
	if err := json.Unmarshal(raw, &st); err != nil {
		return err
	}
	fmt.Printf("status    %s (generation %d, offline %t)\nlast sync %s\ncert      valid until %s\n", st.NodeStatus, st.Generation, st.Offline,
		st.LastSync, st.CertNotAfter)
	if st.LastError != "" {
		fmt.Printf("error     %s\n", st.LastError)
	}
	for _, r := range st.Replicas {
		line := fmt.Sprintf("replica   %s  %-9s in-flight %d", r.ReplicaID, r.State, r.InFlight)
		if r.Error != "" {
			line += "  " + r.Error
		}
		fmt.Println(line)
	}
	return nil
}
