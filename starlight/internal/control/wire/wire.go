// Package wire is the JSON contract between Starlight and its node agents.
package wire

import (
	"errors"
	"fmt"
	"regexp"
	"strings"
)

// Paths served on the node listener; everything except enroll requires a node certificate.
const (
	PathEnroll = "/node/v1/enroll"
	PathSync   = "/node/v1/sync"
	PathRenew  = "/node/v1/renew"
	PathStats  = "/node/v1/stats"
)

// ReplicaPrefix is the node-side route a gateway calls: ReplicaPrefix + replicaID + engine path.
const ReplicaPrefix = "/starlight/v1/replicas/"

// PermitHeader carries the gateway's signed permit to the node.
const PermitHeader = "Starlight-Permit"

// EnginePaths are the only engine routes a node forwards, keyed by method and path.
var EnginePaths = map[string]bool{
	"POST /v1/chat/completions":      true,
	"POST /v1/completions":           true,
	"POST /v1/messages":              true,
	"POST /v1/messages/count_tokens": true,
	"POST /v1/responses":             true,
	"POST /v1/embeddings":            true,
	"POST /v1/tokenize":              true,
	"POST /v1/detokenize":            true,
}

var responseIDPath = regexp.MustCompile(`^/v1/responses/[A-Za-z0-9_-]{1,128}$`)

// AllowedEngineRoute reports whether a node may forward method and path to its engine.
func AllowedEngineRoute(method, path string) bool {
	if EnginePaths[method+" "+path] {
		return true
	}
	return (method == "GET" || method == "DELETE") && responseIDPath.MatchString(path)
}

// GPU is one accelerator as the node sees it; a UUID pins it across reboots and re-enumeration.
type GPU struct {
	Index       int    `json:"index"`
	Name        string `json:"name"`
	MemoryBytes uint64 `json:"memory_bytes"`
	UUID        string `json:"uuid,omitempty"`
}

// Facts describe a node's platform and capacity.
type Facts struct {
	Hostname      string `json:"hostname"`
	Platform      string `json:"platform"`
	Arch          string `json:"arch"`
	Backend       string `json:"backend"`
	OSVersion     string `json:"os_version"`
	AgentVersion  string `json:"agent_version"`
	EngineVersion string `json:"engine_version"`
	MemoryBytes   uint64 `json:"memory_bytes"`
	DiskFreeBytes uint64 `json:"disk_free_bytes"`
	GPUs          []GPU  `json:"gpus"`
	RankHost      string `json:"rank_host,omitempty"`
}

// EnrollRequest exchanges a one-time token and a CSR for a node identity.
type EnrollRequest struct {
	Token   string `json:"token"`
	CSR     string `json:"csr"`
	Name    string `json:"name"`
	Address string `json:"address"`
	Facts   Facts  `json:"facts"`
}

// PermitKey is a public key nodes use to verify gateway permits.
type PermitKey struct {
	KID       string `json:"kid"`
	PublicKey []byte `json:"public_key"`
}

// EnrollResponse returns the node's certificate and the trust anchors it must pin.
type EnrollResponse struct {
	NodeID      string      `json:"node_id"`
	Status      string      `json:"status"`
	Certificate string      `json:"certificate"`
	CA          string      `json:"ca"`
	TrustDomain string      `json:"trust_domain"`
	PermitKeys  []PermitKey `json:"permit_keys"`
}

// RenewRequest asks for a fresh certificate for a new key.
type RenewRequest struct {
	CSR string `json:"csr"`
}

// RenewResponse carries the renewed certificate.
type RenewResponse struct {
	Certificate string `json:"certificate"`
}

// Artifact names model files by source; nodes refuse anything they cannot verify.
type Artifact struct {
	Kind     string `json:"kind"`
	Repo     string `json:"repo,omitempty"`
	Revision string `json:"revision,omitempty"`
	Path     string `json:"path,omitempty"`
	URL      string `json:"url,omitempty"`
	SHA256   string `json:"sha256,omitempty"`
}

var (
	repoPattern     = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$`)
	revisionPattern = regexp.MustCompile(`^[0-9a-f]{40}$`)
	shaPattern      = regexp.MustCompile(`^[0-9a-f]{64}$`)
)

// Key is a stable identifier for caches and placement affinity.
func (a Artifact) Key() string {
	switch a.Kind {
	case "hf":
		return "hf:" + a.Repo + "@" + a.Revision
	case "path":
		return "path:" + a.Path
	default:
		return "url:" + a.SHA256
	}
}

// Validate refuses artifacts that are unpinned or could escape the models directory.
func (a Artifact) Validate() error {
	switch a.Kind {
	case "hf":
		if !repoPattern.MatchString(a.Repo) {
			return fmt.Errorf("artifact repo %q is not owner/name", a.Repo)
		}
		if !revisionPattern.MatchString(a.Revision) {
			return errors.New("hf artifacts must pin a 40-character commit revision")
		}
	case "path":
		if !strings.HasPrefix(a.Path, "/") || strings.Contains(a.Path, "..") || strings.ContainsAny(a.Path, "\x00\n") {
			return fmt.Errorf("artifact path %q must be absolute and clean", a.Path)
		}
	case "url":
		if !strings.HasPrefix(a.URL, "https://") {
			return errors.New("url artifacts must use https")
		}
		if !shaPattern.MatchString(a.SHA256) {
			return errors.New("url artifacts must pin a sha256 of the archive")
		}
	default:
		return fmt.Errorf("unknown artifact kind %q", a.Kind)
	}
	return nil
}

// Spec is the typed engine configuration; nodes build argv from it and never from free text.
type Spec struct {
	Model           Artifact  `json:"model"`
	Drafter         *Artifact `json:"drafter,omitempty"`
	Backend         string    `json:"backend,omitempty"`
	Context         int       `json:"context,omitempty"`
	Parallel        int       `json:"parallel,omitempty"`
	KVCache         string    `json:"kv_cache,omitempty"`
	ReasoningEffort string    `json:"reasoning_effort,omitempty"`
	ThinkingBudget  int       `json:"thinking_budget,omitempty"`
	MaxTokens       int       `json:"max_tokens,omitempty"`
	PromptCacheGiB  int       `json:"prompt_cache_gib,omitempty"`
	PromptCache     string    `json:"prompt_cache,omitempty"`
	Vision          bool      `json:"vision,omitempty"`
	NoDrafts        bool      `json:"no_drafts,omitempty"`
	MemoryGiB       int       `json:"memory_gib,omitempty"`
	GPUs            int       `json:"gpus,omitempty"`
	Nodes           int       `json:"nodes,omitempty"`
	TensorParallel  string    `json:"tensor_parallel,omitempty"`
}

// Validate bounds every field the node will turn into an engine flag.
func (s Spec) Validate() error {
	if err := s.Model.Validate(); err != nil {
		return err
	}
	if s.Drafter != nil {
		if err := s.Drafter.Validate(); err != nil {
			return fmt.Errorf("drafter: %w", err)
		}
	}
	switch s.Backend {
	case "", "metal", "cuda", "hip", "xpu":
	default:
		return fmt.Errorf("unknown backend %q", s.Backend)
	}
	switch s.KVCache {
	case "", "bf16", "fp8":
	default:
		return fmt.Errorf("kv_cache must be bf16 or fp8")
	}
	switch {
	case s.PromptCache != "" && s.PromptCache != "shared" && s.PromptCache != "off":
		return fmt.Errorf("prompt_cache must be shared or off")
	case s.PromptCache == "off" && s.PromptCacheGiB > 0:
		return fmt.Errorf("prompt_cache off keeps no prompt states, so prompt_cache_gib must be 0")
	}
	switch s.ReasoningEffort {
	case "", "low", "medium", "high", "xhigh":
	default:
		return fmt.Errorf("reasoning_effort must be low, medium, high or xhigh")
	}
	if err := s.validateRanks(); err != nil {
		return err
	}
	bounds := []struct {
		name      string
		v, lo, hi int
	}{
		{"context", s.Context, 0, 4 << 20}, {"parallel", s.Parallel, 0, 128}, {"thinking_budget", s.ThinkingBudget, 0, 1 << 20},
		{"max_tokens", s.MaxTokens, 0, 1 << 20}, {"prompt_cache_gib", s.PromptCacheGiB, 0, 4096}, {"memory_gib", s.MemoryGiB, 0, 1 << 16},
		{"gpus", s.GPUs, 0, 16}, {"nodes", s.Nodes, 0, 16},
	}
	for _, b := range bounds {
		if b.v < b.lo || b.v > b.hi {
			return fmt.Errorf("%s must be between %d and %d", b.name, b.lo, b.hi)
		}
	}
	return nil
}

// Assignment is one replica the control plane wants the node to run or stop.
type Assignment struct {
	ReplicaID    string   `json:"replica_id"`
	DeploymentID string   `json:"deployment_id"`
	Revision     int      `json:"revision"`
	Generation   int64    `json:"generation"`
	ServedName   string   `json:"served_name"`
	Desired      string   `json:"desired"`
	Slots        int      `json:"slots"`
	Spec         Spec     `json:"spec"`
	Devices      []string `json:"devices,omitempty"`
	Group        *Group   `json:"group,omitempty"`
}

// ReplicaStatus is the node's observed state of one assignment.
type ReplicaStatus struct {
	ReplicaID  string `json:"replica_id"`
	Generation int64  `json:"generation"`
	State      string `json:"state"`
	Error      string `json:"error,omitempty"`
	InFlight   int    `json:"in_flight"`
	Waiting    int    `json:"waiting"`
	MaxBatch   int    `json:"max_batch,omitempty"`
}

// SyncRequest reports node state; the response waits until desired state changes or a timeout passes.
type SyncRequest struct {
	Generation      int64           `json:"generation"`
	Address         string          `json:"address"`
	Facts           Facts           `json:"facts"`
	Replicas        []ReplicaStatus `json:"replicas"`
	CachedArtifacts []string        `json:"cached_artifacts"`
}

// SyncResponse is the full desired state; nodes stop anything it does not list.
type SyncResponse struct {
	Generation  int64        `json:"generation"`
	NodeStatus  string       `json:"node_status"`
	Offline     bool         `json:"offline"`
	Assignments []Assignment `json:"assignments"`
	PermitKeys  []PermitKey  `json:"permit_keys"`
	RenewAfter  int64        `json:"renew_after"`
}

// ReplicaStats is a replica's live load, measured by the node from its engine every few seconds.
type ReplicaStats struct {
	ReplicaID             string  `json:"replica_id"`
	TokensPerSecond       float64 `json:"tokens_per_second"`
	PromptTokensPerSecond float64 `json:"prompt_tokens_per_second"`
	Running               int     `json:"running"`
	Waiting               int     `json:"waiting"`
	KVUsage               float64 `json:"kv_usage"`
}

// StatsReport is posted to PathStats every five seconds while replicas run.
type StatsReport struct {
	Replicas []ReplicaStats `json:"replicas"`
}

// Replica states reported by nodes.
const (
	StateAssigned = "assigned"
	StateStaging  = "staging"
	StateStarting = "starting"
	StateReady    = "ready"
	StateDraining = "draining"
	StateStopped  = "stopped"
	StateFailed   = "failed"
)
