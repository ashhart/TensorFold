package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/cli"
	"github.com/ashhart/TensorFold/starlight/internal/node/agent"
	"github.com/ashhart/TensorFold/starlight/internal/node/enroll"
	"github.com/ashhart/TensorFold/starlight/internal/node/facts"
	"github.com/ashhart/TensorFold/starlight/internal/node/state"
)

type envList []string

func (e *envList) String() string { return strings.Join(*e, ",") }

var envKey = regexp.MustCompile(`^(TENSORFOLD_[A-Z0-9_]+|TF_[A-Z0-9_]+|NCCL_[A-Z0-9_]+|(CUDA|NVIDIA|HIP|ROCR)_VISIBLE_DEVICES)$`)

// reserved names are set per engine by the agent itself.
var reserved = map[string]bool{"TENSORFOLD_API_KEY": true, "TENSORFOLD_REQUEST_LOG": true, "NCCL_COMM_ID": true}

func (e *envList) Set(v string) error {
	k, _, ok := strings.Cut(v, "=")
	if !ok || !envKey.MatchString(k) || reserved[k] {
		return fmt.Errorf("--env takes KEY=VALUE for TENSORFOLD_*, TF_*, NCCL_* or a *_VISIBLE_DEVICES list; %q is not allowed", k)
	}
	*e = append(*e, v)
	return nil
}

// withholdGPUs applies an --env device list to the agent too, so it reports only the GPUs Starlight may assign.
func withholdGPUs(env []string) error {
	for _, kv := range env {
		k, v, _ := strings.Cut(kv, "=")
		if strings.HasSuffix(k, "_VISIBLE_DEVICES") && k != "NVIDIA_VISIBLE_DEVICES" {
			if err := os.Setenv(k, v); err != nil {
				return err
			}
		}
	}
	return nil
}

func runCommand() *cli.Command {
	var cfg agent.Config
	var env envList
	var level, format string
	return &cli.Command{Name: "run", Summary: "Serve assigned replicas until stopped.",
		Flags: func(fs *flag.FlagSet) {
			stateFlag(fs, &cfg.StateDir)
			fs.StringVar(&cfg.Listen, "listen", envOr("STARLIGHT_NODE_LISTEN", ":7443"), "gateway-facing mTLS listen address")
			fs.StringVar(&cfg.Advertise, "advertise", os.Getenv("STARLIGHT_NODE_ADVERTISE"), "host:port gateways dial (default: hostname and listen port)")
			fs.StringVar(&cfg.Engine, "engine", os.Getenv("STARLIGHT_NODE_ENGINE"), "TensorFold engine binary (default: tensorfold on PATH)")
			fs.StringVar(&cfg.ModelsDir, "models-dir", os.Getenv("STARLIGHT_NODE_MODELS_DIR"), "where URL artifacts are unpacked (default: state-dir/models)")
			fs.StringVar(&cfg.HFHome, "hf-home", os.Getenv("STARLIGHT_NODE_HF_HOME"), "HF_HOME for pulled Hugging Face artifacts")
			fs.StringVar(&cfg.HFEndpoint, "hf-endpoint", os.Getenv("STARLIGHT_NODE_HF_ENDPOINT"), "internal Hugging Face mirror; allowed when offline")
			fs.BoolVar(&cfg.PreventSleep, "prevent-sleep", false, "keep the machine awake while the agent runs (macOS, Windows)")
			fs.DurationVar(&cfg.StartupTimeout, "startup-timeout", 30*time.Minute, "longest an engine may take to load and become ready")
			fs.DurationVar(&cfg.DrainTimeout, "drain-timeout", 2*time.Minute, "longest to wait for in-flight requests before stopping an engine")
			fs.DurationVar(&cfg.StatsInterval, "stats-interval", 5*time.Second, "how often live replica stats are measured and reported")
			fs.Var(&env, "env", "extra engine environment KEY=VALUE (repeatable)")
			fs.StringVar(&cfg.RankHost, "rank-host", os.Getenv("STARLIGHT_NODE_RANK_HOST"), "address other nodes reach this one at for tensor-parallel ranks (default: the advertise host)")
			fs.StringVar(&level, "log-level", envOr("STARLIGHT_NODE_LOG_LEVEL", "info"), "debug, info, warn or error")
			fs.StringVar(&format, "log-format", envOr("STARLIGHT_NODE_LOG_FORMAT", "json"), "json or text")
		},
		Run: func(ctx context.Context, fs *flag.FlagSet) error {
			if fs.NArg() > 0 {
				return cli.ErrUsage
			}
			if err := withholdGPUs(env); err != nil {
				return err
			}
			cfg.Env = env
			cfg.Logger = newLogger(os.Stderr, level, format)
			return agent.Run(ctx, cfg)
		},
	}
}

func enrollCommand() *cli.Command {
	var o enroll.Options
	var dir, caFile, engine string
	return &cli.Command{Name: "enroll", Summary: "Exchange a one-time enrolment token for this node's identity.",
		Usage: "--server https://starlight.example:9443 --token sle_...",
		Flags: func(fs *flag.FlagSet) {
			stateFlag(fs, &dir)
			fs.StringVar(&o.Server, "server", os.Getenv("STARLIGHT_NODE_SERVER"), "Starlight node endpoint URL")
			fs.StringVar(&o.Token, "token", os.Getenv("STARLIGHT_NODE_TOKEN"), "one-time enrolment token")
			fs.StringVar(&o.Name, "name", "", "node name (default: hostname)")
			fs.StringVar(&o.Address, "advertise", os.Getenv("STARLIGHT_NODE_ADVERTISE"), "host:port gateways dial")
			fs.StringVar(&caFile, "ca-file", "", "PEM bundle holding Starlight's CA certificate (instead of --ca-sha256)")
			fs.StringVar(&o.CAPin, "ca-sha256", os.Getenv("STARLIGHT_NODE_CA_SHA256"), "SHA-256 of Starlight's CA certificate, as the console prints it")
			fs.StringVar(&engine, "engine", os.Getenv("STARLIGHT_NODE_ENGINE"), "engine binary, for reporting its version")
			fs.BoolVar(&o.Force, "force", false, "replace an existing identity")
		},
		Run: func(ctx context.Context, fs *flag.FlagSet) error {
			if o.Server == "" || o.Token == "" || fs.NArg() > 0 {
				return cli.ErrUsage
			}
			if caFile != "" {
				b, err := os.ReadFile(caFile)
				if err != nil {
					return err
				}
				o.CABundle = b
			}
			d, err := state.Open(dir)
			if err != nil {
				return err
			}
			o.Facts = facts.Collect(ctx, engine, d.Root)
			id, resp, err := enroll.Run(ctx, d, o)
			if err != nil {
				return err
			}
			fmt.Printf("Enrolled node %s (%s) with %s; status %s.\n", id.Name, id.NodeID, id.ServerURL, resp.Status)
			if resp.Status == "pending" {
				fmt.Println("An administrator must approve this node in the Starlight console before it receives work.")
			}
			return nil
		},
	}
}
