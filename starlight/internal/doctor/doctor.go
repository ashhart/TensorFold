// Package doctor checks a Starlight installation end to end and explains each problem it finds.
package doctor

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/config"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/keyring"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/pki"
	"github.com/ashhart/TensorFold/starlight/internal/db"
)

// Status of one check.
type Status string

const (
	OK   Status = "ok"
	Warn Status = "warn"
	Fail Status = "fail"
)

// Check is one finding with what to do about it.
type Check struct {
	Area   string
	Status Status
	Detail string
	Hint   string
}

// Doctor runs checks with the given configuration and connections.
type Doctor struct {
	Config config.Config
	Pool   *pgxpool.Pool
	HTTP   *http.Client
	Now    func() time.Time
	out    []Check
}

func (d *Doctor) add(area string, s Status, detail, hint string) {
	d.out = append(d.out, Check{Area: area, Status: s, Detail: detail, Hint: hint})
}

// Run executes every check in dependency order and returns the findings.
func (d *Doctor) Run(ctx context.Context) []Check {
	if d.Now == nil {
		d.Now = time.Now
	}
	if d.HTTP == nil {
		d.HTTP = &http.Client{Timeout: 5 * time.Second}
	}
	d.config()
	if !d.database(ctx) {
		return d.out
	}
	d.keys(ctx)
	d.installation(ctx)
	d.trust(ctx)
	d.leader(ctx)
	d.fleet(ctx)
	d.identity(ctx)
	d.providers(ctx)
	return d.out
}

func (d *Doctor) config() {
	c := d.Config
	d.add("config", OK, "public_url "+c.PublicURL+", roles "+strings.Join(c.Roles, ","), "")
	if c.NodeURL == "" && c.HasRole("control") {
		d.add("config", Warn, "node_url is not set", "set node_url so enrolment commands name the node listener nodes can reach")
	}
	if c.TLS.CertFile != "" {
		pair, err := tls.LoadX509KeyPair(c.TLS.CertFile, c.TLS.KeyFile)
		if err != nil {
			d.add("tls", Fail, err.Error(), "check tls.cert_file and tls.key_file")
			return
		}
		if leaf, err := x509.ParseCertificate(pair.Certificate[0]); err == nil {
			d.expiry("tls", "public certificate", leaf.NotAfter, 30*24*time.Hour)
		}
	} else if c.Secure() && len(c.TrustedProxies) == 0 {
		d.add("tls", Warn, "TLS terminates elsewhere but no trusted_proxies are set", "list your load balancer CIDRs so client IPs and audit records are right")
	}
}

func (d *Doctor) database(ctx context.Context) bool {
	var version int
	var dbNow time.Time
	if err := d.Pool.QueryRow(ctx, "SELECT current_setting('server_version_num')::int, now()").Scan(&version, &dbNow); err != nil {
		d.add("database", Fail, err.Error(), "check the database URL and that Postgres accepts connections from this host")
		return false
	}
	if version < 150000 {
		d.add("database", Fail, fmt.Sprintf("Postgres %d is too old", version/10000), "Starlight needs Postgres 15 or newer")
		return false
	}
	d.add("database", OK, fmt.Sprintf("Postgres %d.%d reachable", version/10000, version%10000), "")
	skew := d.Now().Sub(dbNow)
	if skew < 0 {
		skew = -skew
	}
	if skew > 2*time.Second {
		d.add("clock", Fail, fmt.Sprintf("this host and Postgres differ by %s", skew.Round(time.Millisecond)), "run NTP everywhere; permits allow 5 s of skew")
	} else {
		d.add("clock", OK, fmt.Sprintf("skew %s against Postgres", skew.Round(time.Millisecond)), "")
	}
	n, err := db.Pending(ctx, d.Pool)
	switch {
	case err != nil:
		d.add("schema", Fail, err.Error(), "")
	case n > 0:
		d.add("schema", Fail, fmt.Sprintf("%d migrations pending", n), "run `starlight migrate`")
	default:
		d.add("schema", OK, "schema matches this build", "")
	}
	return n == 0 && err == nil
}

func (d *Doctor) keys(ctx context.Context) {
	k := d.Config.Keys
	master, err := keyring.LoadMaster(k.MasterKey, k.MasterKeyFile)
	if err != nil {
		d.add("keys", Fail, err.Error(), "set STARLIGHT_MASTER_KEY or keys.master_key_file")
		return
	}
	var prev []*keyring.Master
	for _, f := range k.PreviousKeyFiles {
		if m, err := keyring.LoadMaster("", f); err == nil {
			prev = append(prev, m)
		}
	}
	if err := keyring.New(master, prev...).Load(ctx, d.Pool); err != nil {
		d.add("keys", Fail, err.Error(), "the master key must be the one used at `starlight init`, or list the old one in keys.previous_key_files")
		return
	}
	d.add("keys", OK, "master key "+master.ID+" unwraps every data key", "")
}

func (d *Doctor) installation(ctx context.Context) {
	var org string
	var offline bool
	if err := d.Pool.QueryRow(ctx, "SELECT org_name, offline FROM settings WHERE id").Scan(&org, &offline); err != nil {
		d.add("installation", Fail, "not initialised", "run `starlight init`")
		return
	}
	mode := "cloud providers allowed"
	if offline {
		mode = "offline: no cloud egress"
	}
	d.add("installation", OK, org+", "+mode, "")
	var owners int
	_ = d.Pool.QueryRow(ctx, `SELECT count(*) FROM role_bindings rb JOIN users u ON u.id = rb.subject_id
		WHERE rb.role = 'owner' AND rb.subject_type = 'user' AND rb.project_id IS NULL AND u.status = 'active'`).Scan(&owners)
	if owners == 0 {
		d.add("owners", Warn, "no active owner", "open the setup link from `starlight init`, or `starlight admin setup-code` for a new one")
	} else {
		d.add("owners", OK, fmt.Sprintf("%d active owner(s)", owners), "")
	}
}

func (d *Doctor) trust(ctx context.Context) {
	var certPEM string
	if err := d.Pool.QueryRow(ctx, "SELECT cert_pem FROM pki WHERE name = 'workload-ca'").Scan(&certPEM); err != nil {
		d.add("ca", Fail, "workload CA missing", "run `starlight init`")
		return
	}
	if ca, err := pki.ParseCertPEM([]byte(certPEM)); err == nil {
		d.expiry("ca", "workload CA", ca.NotAfter, 365*24*time.Hour)
	}
	var active int
	_ = d.Pool.QueryRow(ctx, "SELECT count(*) FROM permit_keys WHERE status = 'active'").Scan(&active)
	if active != 1 {
		d.add("permits", Fail, fmt.Sprintf("%d active permit keys", active), "run `starlight keys rotate-permits`")
	} else {
		d.add("permits", OK, "one active permit key", "")
	}
}

func (d *Doctor) expiry(area, what string, notAfter time.Time, warn time.Duration) {
	left := notAfter.Sub(d.Now())
	switch {
	case left <= 0:
		d.add(area, Fail, what+" expired "+notAfter.Format(time.DateOnly), "replace it now")
	case left < warn:
		d.add(area, Warn, what+" expires "+notAfter.Format(time.DateOnly), "plan its replacement")
	default:
		d.add(area, OK, what+" valid until "+notAfter.Format(time.DateOnly), "")
	}
}

func (d *Doctor) leader(ctx context.Context) {
	var expires time.Time
	var epoch int64
	if err := d.Pool.QueryRow(ctx, "SELECT expires_at, epoch FROM controller_leader WHERE id").Scan(&expires, &epoch); err != nil {
		d.add("controller", Fail, err.Error(), "")
		return
	}
	if expires.Before(d.Now()) {
		d.add("controller", Warn, "no instance holds the controller lease", "run at least one instance with the control role")
		return
	}
	d.add("controller", OK, fmt.Sprintf("leader at epoch %d", epoch), "")
}

func hostOf(raw string) string {
	u, err := url.Parse(raw)
	if err != nil {
		return ""
	}
	return u.Hostname()
}

func resolves(ctx context.Context, host string) error {
	if net.ParseIP(host) != nil {
		return nil
	}
	ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	_, err := net.DefaultResolver.LookupHost(ctx, host)
	return err
}
