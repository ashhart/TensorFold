package samlfake

import (
	"bytes"
	"compress/flate"
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/hex"
	"encoding/xml"
	"errors"
	"fmt"
	"html/template"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/beevik/etree"
	crewsaml "github.com/crewjam/saml"
	dsig "github.com/russellhaering/goxmldsig"
)

// Logout shapes a logout message the IdP sends; empty fields take working defaults, the rest forge bad messages.
type Logout struct {
	ID             string
	Issuer         string
	Destination    string
	InResponseTo   string
	Status         string
	NameID         string
	NameIDFormat   string
	SessionIndexes []string
	IssueInstant   time.Time
	RelayState     string
	POST           bool
	Key            *rsa.PrivateKey
	Unsigned       bool
}

// Delivery is how a browser carries a message to the SP: a GET of URL when Form is nil, else a POST of Form to URL.
type Delivery struct {
	URL  string
	Form url.Values
}

// Seen is a logout message the IdP received from the SP; Err says why it would refuse it.
type Seen struct {
	Binding      string
	ID           string
	InResponseTo string
	NameID       crewsaml.NameID
	SessionIndex string
	Status       string
	RelayState   string
	Err          error
}

const redirectSigAlg = dsig.RSASHA256SignatureMethod

var postForm = template.Must(template.New("post").Parse(`<form method="post" action="{{.URL}}">` +
	`{{range $k, $v := .Form}}<input type="hidden" name="{{$k}}" value="{{index $v 0}}">{{end}}</form>`))

func (i *IdP) sloURL() string { return i.Server.URL + "/slo" }

// SetSLOBindings chooses the logout endpoints the metadata lists; none means the IdP does not support single logout.
func (i *IdP) SetSLOBindings(bindings ...string) {
	i.mu.Lock()
	i.sloBindings = bindings
	i.mu.Unlock()
}

// SetLogoutReply shapes how the IdP answers the SP's LogoutRequests from now on.
func (i *IdP) SetLogoutReply(l Logout) {
	i.mu.Lock()
	i.reply = l
	i.mu.Unlock()
}

// LogoutRequests lists the LogoutRequests the SP sent, oldest first.
func (i *IdP) LogoutRequests() []Seen {
	i.mu.Lock()
	defer i.mu.Unlock()
	return append([]Seen(nil), i.requests...)
}

// LogoutResponses lists the SP's answers to IdP-initiated LogoutRequests, oldest first.
func (i *IdP) LogoutResponses() []Seen {
	i.mu.Lock()
	defer i.mu.Unlock()
	return append([]Seen(nil), i.responses...)
}

// LogoutRequest builds a signed IdP-initiated LogoutRequest for the registered SP's logout endpoint.
func (i *IdP) LogoutRequest(l Logout) (Delivery, error) {
	i.mu.Lock()
	if l.NameID == "" {
		l.NameID, l.NameIDFormat = i.session.NameID, i.session.NameIDFormat
	}
	i.mu.Unlock()
	if l.Destination == "" {
		dest, err := i.spEndpoint(l.POST, false)
		if err != nil {
			return Delivery{}, err
		}
		l.Destination = dest
	}
	el := i.header("samlp:LogoutRequest", &l)
	n := el.CreateElement("saml:NameID")
	if l.NameIDFormat != "" {
		n.CreateAttr("Format", l.NameIDFormat)
	}
	n.SetText(l.NameID)
	for _, idx := range l.SessionIndexes {
		el.CreateElement("samlp:SessionIndex").SetText(idx)
	}
	return i.deliver(el, "SAMLRequest", l)
}

// logoutResponse answers a LogoutRequest from the SP, shaped by SetLogoutReply.
func (i *IdP) logoutResponse(requestID, relay string) (Delivery, error) {
	i.mu.Lock()
	l := i.reply
	i.mu.Unlock()
	if l.InResponseTo == "" {
		l.InResponseTo = requestID
	}
	if l.RelayState == "" {
		l.RelayState = relay
	}
	if l.Status == "" {
		l.Status = crewsaml.StatusSuccess
	}
	if l.Destination == "" {
		dest, err := i.spEndpoint(l.POST, true)
		if err != nil {
			return Delivery{}, err
		}
		l.Destination = dest
	}
	el := i.header("samlp:LogoutResponse", &l)
	el.CreateAttr("InResponseTo", l.InResponseTo)
	el.CreateElement("samlp:Status").CreateElement("samlp:StatusCode").CreateAttr("Value", l.Status)
	return i.deliver(el, "SAMLResponse", l)
}

// header starts a protocol message with its ID, instant, destination and issuer, filling defaults into l.
func (i *IdP) header(tag string, l *Logout) *etree.Element {
	if l.ID == "" {
		b := make([]byte, 16)
		_, _ = rand.Read(b)
		l.ID = "_" + hex.EncodeToString(b)
	}
	if l.IssueInstant.IsZero() {
		l.IssueInstant = time.Now()
	}
	if l.Issuer == "" {
		l.Issuer = i.IDP.MetadataURL.String()
	}
	el := etree.NewElement(tag)
	el.CreateAttr("xmlns:samlp", "urn:oasis:names:tc:SAML:2.0:protocol")
	el.CreateAttr("xmlns:saml", "urn:oasis:names:tc:SAML:2.0:assertion")
	el.CreateAttr("ID", l.ID)
	el.CreateAttr("Version", "2.0")
	el.CreateAttr("IssueInstant", l.IssueInstant.UTC().Format(time.RFC3339))
	el.CreateAttr("Destination", l.Destination)
	el.CreateElement("saml:Issuer").SetText(l.Issuer)
	return el
}

// deliver signs el for the binding l asks for, with the IdP's key unless l forges another or none.
func (i *IdP) deliver(el *etree.Element, param string, l Logout) (Delivery, error) {
	key := l.Key
	if key == nil {
		key = i.IDP.Key.(*rsa.PrivateKey)
	}
	if l.POST {
		if !l.Unsigned {
			ctx, err := dsig.NewSigningContext(key, [][]byte{i.IDP.Certificate.Raw})
			if err != nil {
				return Delivery{}, err
			}
			ctx.Canonicalizer = dsig.MakeC14N10ExclusiveCanonicalizerWithPrefixList("")
			signed, err := ctx.SignEnveloped(el)
			if err != nil {
				return Delivery{}, err
			}
			sig := signed.Child[len(signed.Child)-1]
			signed.Child = signed.Child[:len(signed.Child)-1]
			signed.InsertChildAt(signed.SelectElement("Issuer").Index()+1, sig)
			el = signed
		}
		doc := etree.NewDocument()
		doc.SetRoot(el)
		b, err := doc.WriteToBytes()
		if err != nil {
			return Delivery{}, err
		}
		form := url.Values{param: {base64.StdEncoding.EncodeToString(b)}}
		if l.RelayState != "" {
			form.Set("RelayState", l.RelayState)
		}
		return Delivery{URL: l.Destination, Form: form}, nil
	}
	doc := etree.NewDocument()
	doc.SetRoot(el)
	var buf bytes.Buffer
	fw, _ := flate.NewWriter(&buf, flate.BestCompression)
	if _, err := doc.WriteTo(fw); err != nil {
		return Delivery{}, err
	}
	fw.Close()
	q := param + "=" + url.QueryEscape(base64.StdEncoding.EncodeToString(buf.Bytes()))
	if l.RelayState != "" {
		q += "&RelayState=" + url.QueryEscape(l.RelayState)
	}
	if !l.Unsigned {
		q += "&SigAlg=" + url.QueryEscape(redirectSigAlg)
		sum := sha256.Sum256([]byte(q))
		sig, err := rsa.SignPKCS1v15(rand.Reader, key, crypto.SHA256, sum[:])
		if err != nil {
			return Delivery{}, err
		}
		q += "&Signature=" + url.QueryEscape(base64.StdEncoding.EncodeToString(sig))
	}
	return Delivery{URL: l.Destination + "?" + q}, nil
}

// spEndpoint finds the registered SP's logout endpoint for a binding; responses prefer ResponseLocation.
func (i *IdP) spEndpoint(post, response bool) (string, error) {
	i.mu.Lock()
	defer i.mu.Unlock()
	want := crewsaml.HTTPRedirectBinding
	if post {
		want = crewsaml.HTTPPostBinding
	}
	if i.sp == nil {
		return "", errors.New("samlfake: no service provider registered")
	}
	for _, d := range i.sp.SPSSODescriptors {
		for _, e := range d.SingleLogoutServices {
			if e.Binding == want && response && e.ResponseLocation != "" {
				return e.ResponseLocation, nil
			}
			if e.Binding == want {
				return e.Location, nil
			}
		}
	}
	return "", fmt.Errorf("samlfake: the service provider lists no %s logout endpoint", want)
}

// spCert is the registered SP's signing certificate from its metadata.
func (i *IdP) spCert() (*x509.Certificate, error) {
	i.mu.Lock()
	defer i.mu.Unlock()
	if i.sp == nil {
		return nil, errors.New("samlfake: no service provider registered")
	}
	for _, d := range i.sp.SPSSODescriptors {
		for _, k := range d.KeyDescriptors {
			if (k.Use == "signing" || k.Use == "") && len(k.KeyInfo.X509Data.X509Certificates) > 0 {
				der, err := base64.StdEncoding.DecodeString(strings.Join(strings.Fields(k.KeyInfo.X509Data.X509Certificates[0].Data), ""))
				if err != nil {
					return nil, err
				}
				return x509.ParseCertificate(der)
			}
		}
	}
	return nil, errors.New("samlfake: the service provider publishes no signing certificate")
}

// serveSLO receives the SP's LogoutRequests, which it answers, and the SP's answers to the IdP's own requests.
func (i *IdP) serveSLO(w http.ResponseWriter, r *http.Request) {
	param, xmlBody, relay, binding, err := i.readSigned(r)
	if param == "" {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	seen := Seen{Binding: binding, RelayState: relay, Err: err}
	if param == "SAMLResponse" {
		var resp crewsaml.LogoutResponse
		if xerr := xml.Unmarshal(xmlBody, &resp); xerr != nil && seen.Err == nil {
			seen.Err = xerr
		}
		seen.ID, seen.InResponseTo, seen.Status = resp.ID, resp.InResponseTo, resp.Status.StatusCode.Value
		seen.Err = firstErr(seen.Err, i.checkFromSP(resp.Issuer, resp.Destination))
		i.mu.Lock()
		i.responses = append(i.responses, seen)
		i.mu.Unlock()
		fmt.Fprint(w, "signed out at the IdP")
		return
	}
	var req crewsaml.LogoutRequest
	if xerr := xml.Unmarshal(xmlBody, &req); xerr != nil && seen.Err == nil {
		seen.Err = xerr
	}
	seen.ID = req.ID
	if req.NameID != nil {
		seen.NameID = *req.NameID
	}
	if req.SessionIndex != nil {
		seen.SessionIndex = req.SessionIndex.Value
	}
	seen.Err = firstErr(seen.Err, i.checkFromSP(req.Issuer, req.Destination))
	i.mu.Lock()
	i.requests = append(i.requests, seen)
	i.mu.Unlock()
	if seen.Err != nil {
		http.Error(w, seen.Err.Error(), http.StatusBadRequest)
		return
	}
	d, err := i.logoutResponse(req.ID, relay)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if d.Form == nil {
		http.Redirect(w, r, d.URL, http.StatusFound)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_ = postForm.Execute(w, d)
}

func (i *IdP) checkFromSP(issuer *crewsaml.Issuer, destination string) error {
	i.mu.Lock()
	defer i.mu.Unlock()
	switch {
	case issuer == nil || i.sp == nil || issuer.Value != i.sp.EntityID:
		return errors.New("samlfake: message is not from the registered SP")
	case destination != i.sloURL():
		return errors.New("samlfake: message is not addressed to the IdP's logout endpoint")
	}
	return nil
}

// readSigned decodes a message from either binding and verifies the SP's signature on it.
func (i *IdP) readSigned(r *http.Request) (param string, body []byte, relay, binding string, err error) {
	cert, err := i.spCert()
	if err != nil {
		return "", nil, "", "", err
	}
	if r.Method == http.MethodPost {
		if err := r.ParseForm(); err != nil {
			return "", nil, "", "", err
		}
		param = "SAMLRequest"
		if r.PostForm.Get("SAMLResponse") != "" {
			param = "SAMLResponse"
		}
		raw, err := base64.StdEncoding.DecodeString(r.PostForm.Get(param))
		if err != nil {
			return "", nil, "", "", err
		}
		doc := etree.NewDocument()
		if err := doc.ReadFromBytes(raw); err != nil {
			return "", nil, "", "", err
		}
		vc := dsig.NewDefaultValidationContext(&dsig.MemoryX509CertificateStore{Roots: []*x509.Certificate{cert}})
		_, verr := vc.Validate(doc.Root())
		return param, raw, r.PostForm.Get("RelayState"), "post", verr
	}
	raw := map[string]string{}
	var order []string
	for _, part := range strings.Split(r.URL.RawQuery, "&") {
		k, v, _ := strings.Cut(part, "=")
		raw[k] = v
		order = append(order, k)
	}
	param = "SAMLRequest"
	if _, ok := raw["SAMLResponse"]; ok {
		param = "SAMLResponse"
	}
	value, _ := url.QueryUnescape(raw[param])
	relay, _ = url.QueryUnescape(raw["RelayState"])
	compressed, err := base64.StdEncoding.DecodeString(value)
	if err != nil {
		return "", nil, "", "", err
	}
	body, err = io.ReadAll(flate.NewReader(bytes.NewReader(compressed)))
	if err != nil {
		return "", nil, "", "", err
	}
	signed := param + "=" + raw[param]
	if _, ok := raw["RelayState"]; ok {
		signed += "&RelayState=" + raw["RelayState"]
	}
	signed += "&SigAlg=" + raw["SigAlg"]
	alg, _ := url.QueryUnescape(raw["SigAlg"])
	sigB64, _ := url.QueryUnescape(raw["Signature"])
	sig, _ := base64.StdEncoding.DecodeString(sigB64)
	sum := sha256.Sum256([]byte(signed))
	pub, _ := cert.PublicKey.(*rsa.PublicKey)
	switch {
	case alg != redirectSigAlg || pub == nil:
		err = fmt.Errorf("samlfake: unexpected SigAlg %q", alg)
	case rsa.VerifyPKCS1v15(pub, crypto.SHA256, sum[:], sig) != nil:
		err = errors.New("samlfake: query signature does not verify")
	case bytes.Contains(body, []byte("Signature>")):
		err = errors.New("samlfake: redirect-binding XML must not carry a signature")
	case strings.Join(order, ",") != expectedOrder(raw):
		err = errors.New("samlfake: query parameters are out of order")
	}
	return param, body, relay, "redirect", err
}

func expectedOrder(raw map[string]string) string {
	var keys []string
	for _, k := range []string{"SAMLRequest", "SAMLResponse", "RelayState", "SigAlg", "Signature"} {
		if _, ok := raw[k]; ok {
			keys = append(keys, k)
		}
	}
	return strings.Join(keys, ",")
}

func firstErr(errs ...error) error {
	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	return nil
}
