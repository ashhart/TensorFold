package saml

import (
	"encoding/base64"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/beevik/etree"
	crewsaml "github.com/crewjam/saml"
)

// maxResponse bounds a decoded SAMLResponse.
const maxResponse = 1 << 20

const bearerMethod = "urn:oasis:names:tc:SAML:2.0:cm:bearer"

// ErrInvalid wraps every response rejection; the detail is for logs, not users.
var ErrInvalid = errors.New("saml: response rejected")

// Assertion is the verified content Starlight uses from a SAML response; the NameID parts are kept for single logout.
type Assertion struct {
	ID              string
	NameID          string
	NameIDFormat    string
	NameQualifier   string
	SPNameQualifier string
	Attributes      map[string][]string
	SessionIndex    string
	AuthInstant     time.Time
	NotOnOrAfter    time.Time
}

// Parse verifies a POST-binding SAMLResponse against the expected request IDs and returns its assertion.
func (s *SP) Parse(samlResponse string, possibleIDs []string) (a *Assertion, err error) {
	if len(samlResponse) > maxResponse*4/3+4 {
		return nil, fmt.Errorf("%w: response too large", ErrInvalid)
	}
	raw, err := base64.StdEncoding.DecodeString(strings.TrimSpace(samlResponse))
	if err != nil {
		return nil, fmt.Errorf("%w: not base64", ErrInvalid)
	}
	defer func() {
		if r := recover(); r != nil {
			a, err = nil, fmt.Errorf("%w: malformed assertion", ErrInvalid)
		}
	}()
	doc := etree.NewDocument()
	if err := doc.ReadFromBytes(raw); err != nil || doc.Root() == nil {
		return nil, fmt.Errorf("%w: malformed XML", ErrInvalid)
	}
	root := doc.Root()
	if _, err := signatures(root); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalid, err)
	}
	// crewjam decrypts every assertion before checking any signature, and Starlight reads only the first.
	if n := len(root.SelectElements("Assertion")) + len(root.SelectElements("EncryptedAssertion")); n > 1 {
		return nil, fmt.Errorf("%w: response carries %d assertions, not one", ErrInvalid, n)
	}
	parsed, err := s.sp.ParseXMLResponse(raw, possibleIDs, s.sp.AcsURL)
	if err != nil {
		return nil, fmt.Errorf("%w: %s", ErrInvalid, detail(err))
	}
	if len(possibleIDs) > 0 && !answers(parsed, possibleIDs) {
		return nil, fmt.Errorf("%w: assertion does not answer this sign-in", ErrInvalid)
	}
	return extract(parsed)
}

// answers re-checks InResponseTo, which crewjam skips whenever IdP-initiated sign-in is enabled.
func answers(a *crewsaml.Assertion, ids []string) bool {
	if a.Subject == nil {
		return false
	}
	for _, sc := range a.Subject.SubjectConfirmations {
		if sc.SubjectConfirmationData == nil {
			continue
		}
		for _, id := range ids {
			if id != "" && sc.SubjectConfirmationData.InResponseTo == id {
				return true
			}
		}
	}
	return false
}

// detail unwraps crewjam's deliberately vague error into its private cause for logs.
func detail(err error) string {
	var ire *crewsaml.InvalidResponseError
	if errors.As(err, &ire) && ire.PrivateErr != nil {
		return ire.PrivateErr.Error()
	}
	return err.Error()
}

func extract(a *crewsaml.Assertion) (*Assertion, error) {
	if a.Subject == nil || a.Subject.NameID == nil {
		return nil, fmt.Errorf("%w: assertion has no subject", ErrInvalid)
	}
	if a.Conditions == nil {
		return nil, fmt.Errorf("%w: assertion has no conditions", ErrInvalid)
	}
	bearer := false
	expires := a.Conditions.NotOnOrAfter
	for _, sc := range a.Subject.SubjectConfirmations {
		if sc.Method != bearerMethod || sc.SubjectConfirmationData == nil {
			continue
		}
		bearer = true
		if t := sc.SubjectConfirmationData.NotOnOrAfter; !t.IsZero() && (expires.IsZero() || t.Before(expires)) {
			expires = t
		}
	}
	if !bearer {
		return nil, fmt.Errorf("%w: assertion has no bearer subject confirmation", ErrInvalid)
	}
	if a.ID == "" {
		return nil, fmt.Errorf("%w: assertion has no ID", ErrInvalid)
	}
	n := a.Subject.NameID
	out := &Assertion{ID: a.ID, NameID: strings.TrimSpace(n.Value), NameIDFormat: n.Format, NameQualifier: n.NameQualifier,
		SPNameQualifier: n.SPNameQualifier, Attributes: map[string][]string{}, NotOnOrAfter: expires}
	if len(a.AuthnStatements) > 0 {
		out.SessionIndex = a.AuthnStatements[0].SessionIndex
		out.AuthInstant = a.AuthnStatements[0].AuthnInstant
	}
	for _, st := range a.AttributeStatements {
		for _, attr := range st.Attributes {
			for _, v := range attr.Values {
				val := v.Value
				if v.NameID != nil {
					val = v.NameID.Value
				}
				out.Attributes[attr.Name] = append(out.Attributes[attr.Name], val)
				if attr.FriendlyName != "" && attr.FriendlyName != attr.Name {
					out.Attributes[attr.FriendlyName] = append(out.Attributes[attr.FriendlyName], val)
				}
			}
		}
	}
	return out, nil
}
