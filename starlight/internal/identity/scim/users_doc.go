package scim

import (
	"bytes"
	"encoding/json"
	"strings"

	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
)

// userDoc is the working SCIM document: stored attributes with the authoritative columns laid over them.
func userDoc(u userRow) map[string]any {
	doc := deepCopy(u.Attrs)
	if len(doc) == 0 {
		doc = map[string]any{"emails": []any{map[string]any{"value": u.Email, "type": "work", "primary": true}}}
		if u.DisplayName != "" {
			doc["displayName"] = u.DisplayName
		}
	}
	doc["userName"] = u.Email
	if u.SCIMUserName != nil {
		doc["userName"] = *u.SCIMUserName
	}
	delete(doc, "externalId")
	if u.ExternalID != nil {
		doc["externalId"] = *u.ExternalID
	}
	doc["active"] = u.Status == directory.Active
	return doc
}

func deepCopy(m map[string]any) map[string]any {
	out := map[string]any{}
	if len(m) == 0 {
		return out
	}
	b, err := json.Marshal(m)
	if err != nil {
		return out
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	_ = dec.Decode(&out)
	return out
}

var multiValued = []string{"emails", "phoneNumbers", "ims", "photos", "addresses", "entitlements", "roles", "x509Certificates"}

// fieldsFrom validates a user document and derives the columns Starlight relies on.
func fieldsFrom(doc map[string]any) (userFields, *bool, error) {
	var f userFields
	userName, ok := doc["userName"].(string)
	f.UserName = strings.TrimSpace(userName)
	if !ok || f.UserName == "" {
		return f, nil, badRequest(TypeInvalidValue, "userName is required")
	}
	if len(f.UserName) > 256 {
		return f, nil, badRequest(TypeInvalidValue, "userName is longer than 256 characters")
	}
	if v, present := doc["externalId"]; present && v != nil {
		s, ok := v.(string)
		if !ok || len(s) > 256 {
			return f, nil, badRequest(TypeInvalidValue, "externalId must be a string of at most 256 characters")
		}
		if s = strings.TrimSpace(s); s != "" {
			f.ExternalID = &s
		}
	}
	for _, name := range multiValued {
		if arr, ok := doc[name].([]any); ok {
			onePrimary(arr)
		}
	}
	f.Email = primaryEmail(doc, f.UserName)
	if !validEmail(f.Email) {
		return f, nil, badRequest(TypeInvalidValue, "a user needs an email address: send emails or an email-shaped userName")
	}
	f.DisplayName = displayNameOf(doc)
	if len(f.DisplayName) > 256 {
		return f, nil, badRequest(TypeInvalidValue, "displayName is longer than 256 characters")
	}
	var active *bool
	if v, present := doc["active"]; present && v != nil {
		b, ok := asBool(v)
		if !ok {
			return f, nil, badRequest(TypeInvalidValue, "active must be a boolean")
		}
		active = &b
	}
	f.Attrs = map[string]any{}
	for k, v := range doc {
		switch k {
		case "active", "id", "meta", "groups", "schemas", "password":
		default:
			f.Attrs[k] = v
		}
	}
	if err := checkSize(f.Attrs); err != nil {
		return f, nil, err
	}
	return f, active, nil
}

// onePrimary keeps at most one primary value, the last one marked, as RFC 7643 section 2.4 requires.
func onePrimary(arr []any) {
	last := -1
	for i, el := range arr {
		if obj, ok := el.(map[string]any); ok {
			if b, _ := asBool(obj["primary"]); b {
				last = i
			}
		}
	}
	for i, el := range arr {
		if obj, ok := el.(map[string]any); ok && i != last {
			if _, has := obj["primary"]; has {
				obj["primary"] = false
			}
		}
	}
}

// primaryEmail picks the primary email, then a work email, then the first, then an email-shaped userName.
func primaryEmail(doc map[string]any, userName string) string {
	arr, _ := doc["emails"].([]any)
	var first, work string
	for _, el := range arr {
		obj, ok := el.(map[string]any)
		if !ok {
			continue
		}
		v, _ := obj["value"].(string)
		v = strings.TrimSpace(v)
		if v == "" {
			continue
		}
		if b, _ := asBool(obj["primary"]); b {
			return v
		}
		if t, _ := obj["type"].(string); work == "" && strings.EqualFold(t, "work") {
			work = v
		}
		if first == "" {
			first = v
		}
	}
	switch {
	case work != "":
		return work
	case first != "":
		return first
	}
	return userName
}

func validEmail(s string) bool {
	at := strings.LastIndex(s, "@")
	return at > 0 && at < len(s)-1 && len(s) <= 320 && !strings.ContainsAny(s, " \t\r\n")
}

// displayNameOf prefers displayName, then name.formatted, then given and family names.
func displayNameOf(doc map[string]any) string {
	if s, _ := doc["displayName"].(string); strings.TrimSpace(s) != "" {
		return strings.TrimSpace(s)
	}
	name, _ := doc["name"].(map[string]any)
	if s, _ := name["formatted"].(string); strings.TrimSpace(s) != "" {
		return strings.TrimSpace(s)
	}
	given, _ := name["givenName"].(string)
	family, _ := name["familyName"].(string)
	return strings.TrimSpace(strings.TrimSpace(given) + " " + strings.TrimSpace(family))
}
