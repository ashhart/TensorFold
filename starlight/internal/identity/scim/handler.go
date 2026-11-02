package scim

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/audit"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/ratelimit"
)

// Prefix is where the SCIM service is mounted; IdPs are given BaseURL + Prefix as the tenant URL.
const Prefix = "/scim/v2"

const (
	maxBody      = 4 << 20
	defaultCount = 100
	maxCount     = 500
)

// Options configure the SCIM service.
type Options struct {
	Pool    *pgxpool.Pool
	BaseURL string
	Proxies *httpx.Proxies
	Rate    float64
	Burst   int
	RateKey []byte
}

// Handler serves SCIM 2.0 for every connection; each token sees only its own connection's resources.
type Handler struct {
	pool    *pgxpool.Pool
	base    string
	proxies *httpx.Proxies
	limit   *ratelimit.Limiter
	mux     *http.ServeMux
}

// New builds the handler; Rate and Burst bound requests per token on all instances together (defaults 25/s, burst 200).
func New(o Options) *Handler {
	h := &Handler{pool: o.Pool, base: strings.TrimRight(o.BaseURL, "/") + Prefix, proxies: o.Proxies, limit: newLimiter(o.Pool, o.RateKey, o.Rate, o.Burst)}
	m := http.NewServeMux()
	m.HandleFunc("GET "+Prefix+"/ServiceProviderConfig", h.serviceProviderConfig)
	m.HandleFunc("GET "+Prefix+"/ResourceTypes", h.resourceTypes)
	m.HandleFunc("GET "+Prefix+"/ResourceTypes/{name}", h.resourceType)
	m.HandleFunc("GET "+Prefix+"/Schemas", h.schemas)
	m.HandleFunc("GET "+Prefix+"/Schemas/{id}", h.schema)
	m.HandleFunc("GET "+Prefix+"/Users", h.listUsers)
	m.HandleFunc("POST "+Prefix+"/Users", h.createUser)
	m.HandleFunc("GET "+Prefix+"/Users/{id}", h.getUser)
	m.HandleFunc("PUT "+Prefix+"/Users/{id}", h.replaceUser)
	m.HandleFunc("PATCH "+Prefix+"/Users/{id}", h.patchUser)
	m.HandleFunc("DELETE "+Prefix+"/Users/{id}", h.deleteUser)
	m.HandleFunc("GET "+Prefix+"/Groups", h.listGroups)
	m.HandleFunc("POST "+Prefix+"/Groups", h.createGroup)
	m.HandleFunc("GET "+Prefix+"/Groups/{id}", h.getGroup)
	m.HandleFunc("PUT "+Prefix+"/Groups/{id}", h.replaceGroup)
	m.HandleFunc("PATCH "+Prefix+"/Groups/{id}", h.patchGroup)
	m.HandleFunc("DELETE "+Prefix+"/Groups/{id}", h.deleteGroup)
	m.HandleFunc("/", h.fallback)
	h.mux = m
	return h
}

// Register mounts the service on mux under Prefix.
func (h *Handler) Register(mux *http.ServeMux) {
	mux.Handle(Prefix+"/", h)
}

// ServeHTTP authenticates, rate-limits and dispatches one SCIM request.
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	c, err := h.authenticate(r)
	if err != nil {
		writeError(w, err)
		return
	}
	if ok, wait := h.limit.Take(r.Context(), c.TokenID); !ok {
		w.Header().Set("Retry-After", strconv.Itoa(int(wait.Seconds())+1))
		writeError(w, errf(http.StatusTooManyRequests, "", "too many requests for this SCIM token"))
		return
	}
	if r.Body != nil {
		r.Body = http.MaxBytesReader(w, r.Body, maxBody)
	}
	h.mux.ServeHTTP(w, r.WithContext(context.WithValue(r.Context(), callerKey{}, c)))
}

func (h *Handler) fallback(w http.ResponseWriter, r *http.Request) {
	rest := strings.TrimPrefix(r.URL.Path, Prefix+"/")
	head, _, _ := strings.Cut(rest, "/")
	switch head {
	case "Users", "Groups", "ServiceProviderConfig", "ResourceTypes", "Schemas":
		writeError(w, errf(http.StatusMethodNotAllowed, "", "%s is not supported on %s", r.Method, r.URL.Path))
	case "Bulk":
		writeError(w, errf(http.StatusNotImplemented, "", "bulk operations are not supported"))
	default:
		writeError(w, errNotFound)
	}
}

// listQuery is a parsed list request.
type listQuery struct {
	start    int
	count    int
	filter   Filter
	attrs    []string
	excluded []string
}

func parseList(r *http.Request) (listQuery, error) {
	q := r.URL.Query()
	lq := listQuery{start: 1, count: defaultCount, attrs: splitList(q.Get("attributes")), excluded: splitList(q.Get("excludedAttributes"))}
	if s := q.Get("startIndex"); s != "" {
		n, err := strconv.Atoi(s)
		if err != nil {
			return lq, badRequest(TypeInvalidValue, "startIndex must be an integer")
		}
		lq.start = max(n, 1)
	}
	if s := q.Get("count"); s != "" {
		n, err := strconv.Atoi(s)
		if err != nil {
			return lq, badRequest(TypeInvalidValue, "count must be an integer")
		}
		lq.count = min(max(n, 0), maxCount)
	}
	if f := strings.TrimSpace(q.Get("filter")); f != "" {
		parsed, err := ParseFilter(f)
		if err != nil {
			return lq, err
		}
		lq.filter = parsed
	}
	return lq, nil
}

func (h *Handler) writeList(w http.ResponseWriter, lq listQuery, total int, items []map[string]any) {
	if items == nil {
		items = []map[string]any{}
	}
	writeJSON(w, http.StatusOK, map[string]any{"schemas": []string{URNListResponse}, "totalResults": total,
		"startIndex": lq.start, "itemsPerPage": len(items), "Resources": items})
}

// readObject decodes a JSON object body, keeping numbers exact.
func readObject(r *http.Request) (map[string]any, []byte, error) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		var mbe *http.MaxBytesError
		if errors.As(err, &mbe) {
			return nil, nil, errf(http.StatusRequestEntityTooLarge, "", "request body exceeds %d bytes", maxBody)
		}
		return nil, nil, badRequest(TypeInvalidSyntax, "could not read the request body")
	}
	var obj map[string]any
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.UseNumber()
	if err := dec.Decode(&obj); err != nil || obj == nil {
		return nil, nil, badRequest(TypeInvalidSyntax, "request body must be a JSON object")
	}
	return obj, body, nil
}

func pathID(r *http.Request) (string, error) {
	id := strings.ToLower(r.PathValue("id"))
	if !ids.Valid(id) {
		return "", errNotFound
	}
	return id, nil
}

func version(t time.Time) string {
	return `W/"` + strconv.FormatInt(t.UnixMicro(), 10) + `"`
}

// checkIfMatch enforces an If-Match precondition when the client sends one.
func checkIfMatch(r *http.Request, updated time.Time) error {
	want := strings.TrimSpace(r.Header.Get("If-Match"))
	if want == "" || want == "*" {
		return nil
	}
	for _, v := range strings.Split(want, ",") {
		if strings.TrimSpace(v) == version(updated) {
			return nil
		}
	}
	return errPrecondition
}

func (h *Handler) meta(kind, id string, created, updated time.Time) map[string]any {
	return map[string]any{"resourceType": kind, "created": created.UTC().Format(time.RFC3339Nano),
		"lastModified": updated.UTC().Format(time.RFC3339Nano), "location": h.base + "/" + kind + "s/" + id, "version": version(updated)}
}

// event starts an audit event attributed to the SCIM token behind r.
func (h *Handler) event(r *http.Request, action, targetType, targetID string, details map[string]any) audit.Event {
	e := audit.FromRequest(r, h.proxies, nil)
	c := callerFrom(r.Context())
	e.ActorType, e.ActorID, e.ActorLabel = "scim", c.TokenID, "scim:"+c.ConnectionID
	e.Action, e.TargetType, e.TargetID, e.Details = action, targetType, targetID, details
	return e
}

// tx runs fn in a transaction, retrying transient conflicts.
func (h *Handler) tx(ctx context.Context, fn func(pgx.Tx) error) error {
	return db.Tx(ctx, h.pool, fn)
}

func writeResource(w http.ResponseWriter, status int, rep map[string]any) {
	if m, ok := rep["meta"].(map[string]any); ok {
		if v, ok := m["version"].(string); ok {
			w.Header().Set("ETag", v)
		}
		if loc, ok := m["location"].(string); ok && status == http.StatusCreated {
			w.Header().Set("Location", loc)
		}
	}
	writeJSON(w, status, rep)
}
