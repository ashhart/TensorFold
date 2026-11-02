package scim

import (
	"net/http"
	"strings"
)

func (h *Handler) serviceProviderConfig(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"schemas":        []string{URNSPConfig},
		"patch":          map[string]any{"supported": true},
		"bulk":           map[string]any{"supported": false, "maxOperations": 0, "maxPayloadSize": 0},
		"filter":         map[string]any{"supported": true, "maxResults": maxCount},
		"changePassword": map[string]any{"supported": false},
		"sort":           map[string]any{"supported": false},
		"etag":           map[string]any{"supported": true},
		"authenticationSchemes": []any{map[string]any{"type": "oauthbearertoken", "name": "Bearer token",
			"description": "A SCIM token issued by Starlight for one SSO connection", "primary": true}},
		"meta": map[string]any{"resourceType": "ServiceProviderConfig", "location": h.base + "/ServiceProviderConfig"},
	})
}

func (h *Handler) resourceTypeDoc(r resource) map[string]any {
	doc := map[string]any{
		"schemas": []string{URNResourceType}, "id": r.name(), "name": r.name(), "endpoint": "/" + r.name() + "s",
		"description": r.name() + " account", "schema": r.urn(),
		"meta": map[string]any{"resourceType": "ResourceType", "location": h.base + "/ResourceTypes/" + r.name()},
	}
	if r == resUser {
		doc["schemaExtensions"] = []any{map[string]any{"schema": URNEnterprise, "required": false}}
	}
	return doc
}

func (h *Handler) resourceTypes(w http.ResponseWriter, r *http.Request) {
	items := []map[string]any{h.resourceTypeDoc(resUser), h.resourceTypeDoc(resGroup)}
	h.writeList(w, listQuery{start: 1}, len(items), items)
}

func (h *Handler) resourceType(w http.ResponseWriter, r *http.Request) {
	switch strings.ToLower(r.PathValue("name")) {
	case "user":
		writeJSON(w, http.StatusOK, h.resourceTypeDoc(resUser))
	case "group":
		writeJSON(w, http.StatusOK, h.resourceTypeDoc(resGroup))
	default:
		writeError(w, errNotFound)
	}
}

func (h *Handler) schemaDocs() []map[string]any {
	return []map[string]any{
		h.schemaDoc(URNUser, "User", "User account", userAttrs),
		h.schemaDoc(URNGroup, "Group", "Group", groupAttrs),
		h.schemaDoc(URNEnterprise, "EnterpriseUser", "Enterprise user", enterpriseAttrs),
	}
}

func (h *Handler) schemaDoc(id, name, desc string, defs []attr) map[string]any {
	return map[string]any{"schemas": []string{URNSchema}, "id": id, "name": name, "description": desc, "attributes": describeAttrs(defs),
		"meta": map[string]any{"resourceType": "Schema", "location": h.base + "/Schemas/" + id}}
}

func describeAttrs(defs []attr) []any {
	out := make([]any, 0, len(defs))
	for _, a := range defs {
		d := map[string]any{"name": a.Name, "type": a.Type, "multiValued": a.Multi, "required": a.Required, "caseExact": a.CaseExact,
			"mutability": orDefault(a.Mutability, "readWrite"), "returned": orDefault(a.Returned, "default"), "uniqueness": orDefault(a.Uniqueness, "none")}
		if a.Type == "reference" {
			d["referenceTypes"] = []string{"external"}
		}
		if len(a.Sub) > 0 {
			d["subAttributes"] = describeAttrs(a.Sub)
		}
		out = append(out, d)
	}
	return out
}

func orDefault(s, def string) string {
	if s == "" {
		return def
	}
	return s
}

func (h *Handler) schemas(w http.ResponseWriter, r *http.Request) {
	docs := h.schemaDocs()
	h.writeList(w, listQuery{start: 1}, len(docs), docs)
}

func (h *Handler) schema(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	for _, d := range h.schemaDocs() {
		if strings.EqualFold(d["id"].(string), id) {
			writeJSON(w, http.StatusOK, d)
			return
		}
	}
	writeError(w, errNotFound)
}
