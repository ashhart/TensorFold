package scim

import "strings"

// attr is one RFC 7643 attribute definition; it drives key canonicalisation, PATCH semantics and /Schemas.
type attr struct {
	Name       string
	Type       string
	Multi      bool
	Required   bool
	CaseExact  bool
	Mutability string
	Returned   string
	Uniqueness string
	Sub        []attr
}

func str(name string) attr { return attr{Name: name, Type: "string"} }

func boolean(name string) attr { return attr{Name: name, Type: "boolean"} }

func complexAttr(name string, sub ...attr) attr { return attr{Name: name, Type: "complex", Sub: sub} }

func multi(name string, sub ...attr) attr {
	return attr{Name: name, Type: "complex", Multi: true, Sub: sub}
}

func readOnly(a attr) attr {
	a.Mutability = "readOnly"
	return a
}

func plural(name string) attr {
	return multi(name, str("value"), str("display"), str("type"), boolean("primary"))
}

var userAttrs = []attr{
	{Name: "id", Type: "string", CaseExact: true, Mutability: "readOnly", Returned: "always", Uniqueness: "server"},
	{Name: "externalId", Type: "string", CaseExact: true},
	{Name: "userName", Type: "string", Required: true, Uniqueness: "server"},
	complexAttr("name", str("formatted"), str("familyName"), str("givenName"), str("middleName"), str("honorificPrefix"), str("honorificSuffix")),
	str("displayName"), str("nickName"), {Name: "profileUrl", Type: "reference"}, str("title"), str("userType"),
	str("preferredLanguage"), str("locale"), str("timezone"), boolean("active"),
	{Name: "password", Type: "string", Mutability: "writeOnly", Returned: "never"},
	plural("emails"), plural("phoneNumbers"), plural("ims"), plural("photos"),
	multi("addresses", str("formatted"), str("streetAddress"), str("locality"), str("region"), str("postalCode"), str("country"),
		str("type"), boolean("primary")),
	readOnly(multi("groups", str("value"), attr{Name: "$ref", Type: "reference"}, str("display"), str("type"))),
	plural("entitlements"), plural("roles"), plural("x509Certificates"),
}

var enterpriseAttrs = []attr{
	str("employeeNumber"), str("costCenter"), str("organization"), str("division"), str("department"),
	complexAttr("manager", str("value"), attr{Name: "$ref", Type: "reference"}, readOnly(str("displayName"))),
}

var groupAttrs = []attr{
	{Name: "id", Type: "string", CaseExact: true, Mutability: "readOnly", Returned: "always", Uniqueness: "server"},
	{Name: "externalId", Type: "string", CaseExact: true},
	{Name: "displayName", Type: "string", Required: true, Uniqueness: "server"},
	multi("members", attr{Name: "value", Type: "string", Mutability: "immutable"}, attr{Name: "$ref", Type: "reference"},
		str("display"), str("type")),
}

// findAttr looks a name up case-insensitively.
func findAttr(defs []attr, name string) (attr, bool) {
	for _, a := range defs {
		if strings.EqualFold(a.Name, name) {
			return a, true
		}
	}
	return attr{}, false
}

// resource identifies which schema a document follows.
type resource int

const (
	resUser resource = iota
	resGroup
)

func (r resource) attrs() []attr {
	if r == resGroup {
		return groupAttrs
	}
	return userAttrs
}

func (r resource) urn() string {
	if r == resGroup {
		return URNGroup
	}
	return URNUser
}

func (r resource) name() string {
	if r == resGroup {
		return "Group"
	}
	return "User"
}

// defsFor returns the attribute table and container key for a parsed path's schema.
func (r resource) defsFor(p AttrPath) ([]attr, string, bool) {
	switch {
	case p.URN == "":
		return r.attrs(), "", true
	case p.URN == URNEnterprise && r == resUser:
		return enterpriseAttrs, URNEnterprise, true
	}
	return nil, p.URN, false
}
