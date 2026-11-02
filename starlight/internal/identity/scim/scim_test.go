package scim

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

func newUser(email string) map[string]any {
	return map[string]any{"schemas": []string{URNUser}, "userName": email, "emails": []any{map[string]any{"value": email, "primary": true}}}
}

func TestAuthenticationFailures(t *testing.T) {
	e := setup(t)
	if r := e.do("", "GET", "/scim/v2/Users", nil); r.status != 401 || r.body["status"] != "401" || r.header.Get("WWW-Authenticate") == "" {
		t.Fatalf("no token: %d %v", r.status, r.body)
	}
	if r := e.do("not-a-token", "GET", "/scim/v2/Users", nil); r.status != 401 {
		t.Fatalf("malformed token: %d", r.status)
	}
	tok, info, err := IssueToken(context.Background(), e.pool, e.connA, "rotating", nil)
	if err != nil {
		t.Fatal(err)
	}
	e.must(e.do(tok, "GET", "/scim/v2/Users", nil), 200)
	if err := RevokeToken(context.Background(), e.pool, e.connB, info.ID); err != ErrTokenNotFound {
		t.Fatalf("revoked another connection's token: %v", err)
	}
	if err := RevokeToken(context.Background(), e.pool, e.connA, info.ID); err != nil {
		t.Fatal(err)
	}
	if r := e.do(tok, "GET", "/scim/v2/Users", nil); r.status != 401 {
		t.Fatalf("revoked token: %d", r.status)
	}
	past := time.Now().Add(-time.Hour)
	if _, _, err := IssueToken(context.Background(), e.pool, e.connA, "old", &past); err == nil {
		t.Fatal("issued an already expired token")
	}
	toks, err := ListTokens(context.Background(), e.pool, e.connA)
	if err != nil || len(toks) != 2 || toks[0].RevokedAt == nil {
		t.Fatalf("list tokens = %+v %v", toks, err)
	}
	if _, err := e.pool.Exec(context.Background(), "UPDATE idp_connections SET enabled = false WHERE id = $1", e.connB); err != nil {
		t.Fatal(err)
	}
	if r := e.do(e.tokenB, "GET", "/scim/v2/Users", nil); r.status != 401 {
		t.Fatalf("disabled connection: %d", r.status)
	}
}

func TestConnectionsAreIsolated(t *testing.T) {
	e := setup(t)
	u := e.must(e.do(e.tokenA, "POST", "/scim/v2/Users", newUser("iso@example.com")), 201)
	g := e.must(e.do(e.tokenA, "POST", "/scim/v2/Groups", map[string]any{"displayName": "Isolated", "members": []any{map[string]any{"value": u["id"]}}}), 201)
	uid, gid := u["id"].(string), g["id"].(string)
	for _, path := range []string{"/scim/v2/Users/" + uid, "/scim/v2/Groups/" + gid} {
		if r := e.do(e.tokenB, "GET", path, nil); r.status != 404 {
			t.Fatalf("other connection read %s: %d", path, r.status)
		}
		if r := e.do(e.tokenB, "DELETE", path, nil); r.status != 404 {
			t.Fatalf("other connection deleted %s: %d", path, r.status)
		}
	}
	if r := e.do(e.tokenB, "PATCH", "/scim/v2/Users/"+uid, map[string]any{"Operations": []any{map[string]any{"op": "replace", "path": "active", "value": false}}}); r.status != 404 {
		t.Fatalf("other connection patched: %d", r.status)
	}
	list := e.must(e.do(e.tokenB, "GET", "/scim/v2/Users?filter="+q(`userName eq "iso@example.com"`), nil), 200)
	if list["totalResults"].(float64) != 0 {
		t.Fatal("other connection listed the user")
	}
	foreign := e.do(e.tokenB, "POST", "/scim/v2/Groups", map[string]any{"displayName": "Mine", "members": []any{map[string]any{"value": uid}}})
	if foreign.status != 400 || foreign.body["scimType"] != TypeInvalidValue {
		t.Fatalf("added another connection's user: %d %v", foreign.status, foreign.body)
	}
	same := e.do(e.tokenB, "POST", "/scim/v2/Users", newUser("iso@example.com"))
	if same.status != 409 || same.body["scimType"] != TypeUniqueness {
		t.Fatalf("second connection took a live email: %d %v", same.status, same.body)
	}
}

func TestUniquenessAndPreconditions(t *testing.T) {
	e := setup(t)
	u := e.must(e.do(e.tokenA, "POST", "/scim/v2/Users", newUser("uniq@example.com")), 201)
	dup := e.do(e.tokenA, "POST", "/scim/v2/Users", newUser("UNIQ@example.com"))
	if dup.status != 409 || dup.body["scimType"] != TypeUniqueness {
		t.Fatalf("duplicate userName: %d %v", dup.status, dup.body)
	}
	e.must(e.do(e.tokenA, "POST", "/scim/v2/Groups", map[string]any{"displayName": "Ops"}), 201)
	if r := e.do(e.tokenA, "POST", "/scim/v2/Groups", map[string]any{"displayName": "ops"}); r.status != 409 || r.body["scimType"] != TypeUniqueness {
		t.Fatalf("duplicate group: %d %v", r.status, r.body)
	}
	id := u["id"].(string)
	etag := e.do(e.tokenA, "GET", "/scim/v2/Users/"+id, nil).header.Get("ETag")
	if !strings.HasPrefix(etag, `W/"`) {
		t.Fatalf("etag %q", etag)
	}
	body := map[string]any{"Operations": []any{map[string]any{"op": "replace", "path": "displayName", "value": "Renamed"}}}
	req := func(match string) int {
		r := e.doHeader(e.tokenA, "PATCH", "/scim/v2/Users/"+id, body, "If-Match", match)
		return r.status
	}
	if s := req(`W/"1"`); s != 412 {
		t.Fatalf("stale If-Match = %d", s)
	}
	if s := req(etag); s != 200 {
		t.Fatalf("current If-Match = %d", s)
	}
	bad := e.do(e.tokenA, "PATCH", "/scim/v2/Users/"+id, map[string]any{"Operations": []any{map[string]any{"op": "replace", "path": "id", "value": "x"}}})
	if bad.status != 400 || bad.body["scimType"] != TypeMutability {
		t.Fatalf("id write: %d %v", bad.status, bad.body)
	}
	noTarget := e.do(e.tokenA, "PATCH", "/scim/v2/Users/"+id, map[string]any{"Operations": []any{map[string]any{"op": "remove"}}})
	if noTarget.body["scimType"] != TypeNoTarget {
		t.Fatalf("remove without path: %v", noTarget.body)
	}
	if r := e.do(e.tokenA, "POST", "/scim/v2/Users", map[string]any{"userName": "no-email"}); r.status != 400 || r.body["scimType"] != TypeInvalidValue {
		t.Fatalf("user without email: %d %v", r.status, r.body)
	}
	if r := e.do(e.tokenA, "POST", "/scim/v2/Users", "{not json"); r.body["scimType"] != TypeInvalidSyntax {
		t.Fatalf("bad json: %v", r.body)
	}
	if r := e.do(e.tokenA, "GET", "/scim/v2/Users?filter="+q(`password eq "x"`), nil); r.body["scimType"] != TypeInvalidFilter {
		t.Fatalf("unsupported filter: %v", r.body)
	}
}

func TestPaginationAndProjection(t *testing.T) {
	e := setup(t)
	for i := 0; i < 7; i++ {
		e.must(e.do(e.tokenA, "POST", "/scim/v2/Users", newUser(fmt.Sprintf("page%d@example.com", i))), 201)
	}
	page := e.must(e.do(e.tokenA, "GET", "/scim/v2/Users?startIndex=1&count=3", nil), 200)
	if page["totalResults"].(float64) != 7 || page["itemsPerPage"].(float64) != 3 || page["startIndex"].(float64) != 1 {
		t.Fatalf("first page = %v", page)
	}
	last := e.must(e.do(e.tokenA, "GET", "/scim/v2/Users?startIndex=7&count=3", nil), 200)
	if len(resources(t, last)) != 1 || resources(t, last)[0]["userName"] != "page6@example.com" {
		t.Fatalf("last page = %v", last)
	}
	none := e.must(e.do(e.tokenA, "GET", "/scim/v2/Users?count=0", nil), 200)
	if none["totalResults"].(float64) != 7 || len(resources(t, none)) != 0 {
		t.Fatalf("count=0 = %v", none)
	}
	capped := e.must(e.do(e.tokenA, "GET", "/scim/v2/Users?count=100000&startIndex=-4", nil), 200)
	if capped["startIndex"].(float64) != 1 || len(resources(t, capped)) != 7 {
		t.Fatalf("clamped = %v", capped)
	}
	proj := e.must(e.do(e.tokenA, "GET", "/scim/v2/Users?attributes=userName,emails.value&filter="+q(`emails co "page3"`), nil), 200)
	items := resources(t, proj)
	if len(items) != 1 || items[0]["id"] == nil || items[0]["meta"] != nil || items[0]["emails"].([]any)[0].(map[string]any)["primary"] != nil {
		t.Fatalf("projection = %v", items)
	}
	sw := e.must(e.do(e.tokenA, "GET", "/scim/v2/Users?filter="+q(`userName sw "page" and not (userName ew "6@example.com")`), nil), 200)
	if sw["totalResults"].(float64) != 6 {
		t.Fatalf("sw/not filter = %v", sw["totalResults"])
	}
}

func TestAdoptsUsersWhoSignedInFirst(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	jit, err := directory.CreateUser(ctx, e.pool, directory.NewUser{Email: "early@example.com", DisplayName: "Early Bird", Source: "jit"})
	if err != nil {
		t.Fatal(err)
	}
	if err := directory.LinkIdentity(ctx, e.pool, jit.ID, e.connA, "sub-early", "early@example.com"); err != nil {
		t.Fatal(err)
	}
	seen := e.must(e.do(e.tokenA, "GET", "/scim/v2/Users?filter="+q(`userName eq "early@example.com"`), nil), 200)
	if seen["totalResults"].(float64) != 1 {
		t.Fatalf("owning connection cannot see its signed-in user: %v", seen)
	}
	if other := e.must(e.do(e.tokenB, "GET", "/scim/v2/Users?filter="+q(`userName eq "early@example.com"`), nil), 200); other["totalResults"].(float64) != 0 {
		t.Fatal("unrelated connection sees the user")
	}
	adopted := e.must(e.do(e.tokenA, "POST", "/scim/v2/Users", newUser("early@example.com")), 201)
	if adopted["id"] != jit.ID {
		t.Fatalf("POST created a duplicate instead of adopting: %v", adopted["id"])
	}
	var owner *string
	if err := e.pool.QueryRow(ctx, "SELECT scim_connection_id FROM users WHERE id = $1", jit.ID).Scan(&owner); err != nil || owner == nil || *owner != e.connA {
		t.Fatalf("owner = %v %v", owner, err)
	}
	assertAudit(t, e, "scim.user.adopt")
}

func TestRateLimitAndRouting(t *testing.T) {
	e := setup(t, func(o *Options) { o.Rate, o.Burst = 0.01, 3 })
	e.must(e.do(e.tokenA, "GET", "/scim/v2/ServiceProviderConfig", nil), 200)
	e.must(e.do(e.tokenA, "GET", "/scim/v2/Schemas", nil), 200)
	e.must(e.do(e.tokenA, "GET", "/scim/v2/Schemas/"+URNUser, nil), 200)
	if r := e.do(e.tokenA, "GET", "/scim/v2/Users", nil); r.status != 429 || r.header.Get("Retry-After") == "" {
		t.Fatalf("burst exceeded: %d", r.status)
	}
	e.must(e.do(e.tokenB, "GET", "/scim/v2/ResourceTypes", nil), 200)
	e.must(e.do(e.tokenB, "GET", "/scim/v2/ResourceTypes/User", nil), 200)
	if r := e.do(e.tokenB, "POST", "/scim/v2/Users/"+strings.Repeat("0", 8), nil); r.status != 405 {
		t.Fatalf("unsupported method: %d", r.status)
	}
}

func TestRateLimitIsSharedAcrossInstances(t *testing.T) {
	key := []byte("both instances derive this")
	e := setup(t, func(o *Options) { o.Rate, o.Burst, o.RateKey = 0.01, 3, key })
	mux := http.NewServeMux()
	New(Options{Pool: e.pool, BaseURL: "https://starlight.example.com", Rate: 0.01, Burst: 3, RateKey: key}).Register(mux)
	other := *e
	other.srv = httptest.NewServer(mux)
	t.Cleanup(other.srv.Close)
	var got []reply
	for _, x := range []*env{e, &other, e, &other} {
		got = append(got, x.do(x.tokenA, "GET", "/scim/v2/ServiceProviderConfig", nil))
	}
	rows, err := e.pool.Query(context.Background(), "SELECT key, window_start FROM counters")
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	windows := map[time.Time]bool{}
	for rows.Next() {
		var key string
		var start time.Time
		if err := rows.Scan(&key, &start); err != nil {
			t.Fatal(err)
		}
		if !regexp.MustCompile(`^rl:scim:[0-9a-f]{32}$`).MatchString(key) {
			t.Fatalf("counter key %q is not a hash", key)
		}
		windows[start] = true
	}
	if len(windows) > 1 {
		t.Skip("the requests straddled a rate-limit window")
	}
	for i, r := range got {
		if want := []int{200, 200, 200, 429}[i]; r.status != want {
			t.Fatalf("request %d across two instances: %d, want %d", i+1, r.status, want)
		}
	}
	if got[3].header.Get("Retry-After") == "" {
		t.Fatal("429 without Retry-After")
	}
}

func (e *env) doHeader(token, method, path string, body any, key, value string) reply {
	e.t.Helper()
	return e.doWith(token, method, path, body, map[string]string{key: value})
}

func TestGroupReplaceAndCleanup(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	u1 := e.must(e.do(e.tokenA, "POST", "/scim/v2/Users", newUser("one@example.com")), 201)["id"].(string)
	u2 := e.must(e.do(e.tokenA, "POST", "/scim/v2/Users", newUser("two@example.com")), 201)["id"].(string)
	gid := e.must(e.do(e.tokenA, "POST", "/scim/v2/Groups", map[string]any{"displayName": "Team", "members": []any{map[string]any{"value": u1}}}), 201)["id"].(string)
	put := e.must(e.do(e.tokenA, "PUT", "/scim/v2/Groups/"+gid, map[string]any{"displayName": "Team", "members": []any{map[string]any{"value": u2}}}), 200)
	if ms := put["members"].([]any); len(ms) != 1 || ms[0].(map[string]any)["value"] != u2 {
		t.Fatalf("put members = %v", put["members"])
	}
	e.must(e.do(e.tokenA, "PATCH", "/scim/v2/Groups/"+gid, map[string]any{"Operations": []any{map[string]any{"op": "replace", "path": "members",
		"value": []any{map[string]any{"value": u1}, map[string]any{"value": u2}}}}}), 204)
	if g := e.must(e.do(e.tokenA, "GET", "/scim/v2/Groups/"+gid+"?attributes=members", nil), 200); len(g["members"].([]any)) != 2 || g["displayName"] != nil {
		t.Fatalf("replace members = %v", g)
	}
	nested := e.do(e.tokenA, "PATCH", "/scim/v2/Groups/"+gid, map[string]any{"Operations": []any{map[string]any{"op": "add", "path": "members",
		"value": []any{map[string]any{"value": gid, "type": "Group"}}}}})
	if nested.status != 400 || nested.body["scimType"] != TypeInvalidValue {
		t.Fatalf("nested group accepted: %d %v", nested.status, nested.body)
	}
	if _, err := e.pool.Exec(ctx, "INSERT INTO role_bindings (id, role, subject_type, subject_id) VALUES ($1, 'auditor', 'group', $2)", ids.New(), gid); err != nil {
		t.Fatal(err)
	}
	if err := directoryLink(ctx, e, u2); err != nil {
		t.Fatal(err)
	}
	e.must(e.do(e.tokenA, "DELETE", "/scim/v2/Users/"+u2, nil), 204)
	var links, members, bindings int
	_ = e.pool.QueryRow(ctx, "SELECT count(*) FROM identities WHERE user_id = $1", u2).Scan(&links)
	_ = e.pool.QueryRow(ctx, "SELECT count(*) FROM group_members WHERE user_id = $1", u2).Scan(&members)
	if links != 0 || members != 0 {
		t.Fatalf("deleted user kept %d identity links and %d memberships", links, members)
	}
	e.must(e.do(e.tokenA, "DELETE", "/scim/v2/Groups/"+gid, nil), 204)
	_ = e.pool.QueryRow(ctx, "SELECT count(*) FROM role_bindings WHERE subject_id = $1", gid).Scan(&bindings)
	if bindings != 0 {
		t.Fatal("group deletion left its role bindings")
	}
}

func directoryLink(ctx context.Context, e *env, userID string) error {
	return directory.LinkIdentity(ctx, e.pool, userID, e.connA, "sub-"+userID, "two@example.com")
}
