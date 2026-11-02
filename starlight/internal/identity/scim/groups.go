package scim

import (
	"net/http"
	"strings"

	"github.com/jackc/pgx/v5"

	"github.com/ashhart/TensorFold/starlight/internal/audit"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

const maxMembers = 20000

func (h *Handler) listGroups(w http.ResponseWriter, r *http.Request) {
	lq, err := parseList(r)
	if err != nil {
		writeError(w, err)
		return
	}
	groups, total, err := queryGroups(r.Context(), h.pool, callerFrom(r.Context()).ConnectionID, lq)
	if err != nil {
		writeError(w, err)
		return
	}
	withMembers := wants(lq, "members")
	var members map[string][]member
	if withMembers {
		gids := make([]string, len(groups))
		for i, g := range groups {
			gids[i] = g.ID
		}
		if members, err = groupMembers(r.Context(), h.pool, gids); err != nil {
			writeError(w, err)
			return
		}
	}
	items := make([]map[string]any, 0, len(groups))
	for _, g := range groups {
		items = append(items, project(h.groupRep(g, members[g.ID], withMembers), lq.attrs, lq.excluded))
	}
	h.writeList(w, lq, total, items)
}

func (h *Handler) getGroup(w http.ResponseWriter, r *http.Request) {
	id, err := pathID(r)
	if err != nil {
		writeError(w, err)
		return
	}
	q := r.URL.Query()
	lq := listQuery{attrs: splitList(q.Get("attributes")), excluded: splitList(q.Get("excludedAttributes"))}
	rep, err := h.loadGroupRep(r, id, wants(lq, "members"))
	if err != nil {
		writeError(w, err)
		return
	}
	writeResource(w, http.StatusOK, project(rep, lq.attrs, lq.excluded))
}

func (h *Handler) loadGroupRep(r *http.Request, id string, withMembers bool) (map[string]any, error) {
	g, err := loadGroup(r.Context(), h.pool, callerFrom(r.Context()).ConnectionID, id, false)
	if err != nil {
		return nil, err
	}
	var ms []member
	if withMembers {
		all, err := groupMembers(r.Context(), h.pool, []string{g.ID})
		if err != nil {
			return nil, err
		}
		ms = all[g.ID]
	}
	return h.groupRep(g, ms, withMembers), nil
}

// groupRep renders a group; members are omitted entirely when they were not loaded.
func (h *Handler) groupRep(g groupRow, ms []member, withMembers bool) map[string]any {
	rep := groupDoc(g, ms)
	if !withMembers {
		delete(rep, "members")
	} else {
		for _, el := range rep["members"].([]any) {
			m := el.(map[string]any)
			m["$ref"], m["type"] = h.base+"/Users/"+m["value"].(string), "User"
		}
	}
	rep["id"] = g.ID
	rep["schemas"] = []string{URNGroup}
	rep["meta"] = h.meta("Group", g.ID, g.CreatedAt, g.UpdatedAt)
	return rep
}

// groupDoc is the patchable document of a group.
func groupDoc(g groupRow, ms []member) map[string]any {
	doc := map[string]any{"displayName": g.DisplayName}
	if g.ExternalID != g.ID {
		doc["externalId"] = g.ExternalID
	}
	els := make([]any, 0, len(ms))
	for _, m := range ms {
		els = append(els, map[string]any{"value": m.ID, "display": m.Label})
	}
	doc["members"] = els
	return doc
}

// groupFields validates a group document and returns its name, externalId and member ids.
func groupFields(doc map[string]any) (string, *string, []string, error) {
	name, _ := doc["displayName"].(string)
	name = strings.TrimSpace(name)
	if name == "" || len(name) > 256 {
		return "", nil, nil, badRequest(TypeInvalidValue, "displayName is required and at most 256 characters")
	}
	var ext *string
	if v, ok := doc["externalId"]; ok && v != nil {
		s, isStr := v.(string)
		if !isStr || len(s) > 256 {
			return "", nil, nil, badRequest(TypeInvalidValue, "externalId must be a string of at most 256 characters")
		}
		if s = strings.TrimSpace(s); s != "" {
			ext = &s
		}
	}
	els, _ := doc["members"].([]any)
	if len(els) > maxMembers {
		return "", nil, nil, badRequest(TypeInvalidValue, "a group holds at most %d members", maxMembers)
	}
	seen := map[string]bool{}
	var out []string
	for _, el := range els {
		obj, ok := el.(map[string]any)
		if !ok {
			return "", nil, nil, badRequest(TypeInvalidValue, "members must be objects with a value")
		}
		if t, _ := lookupFold(obj, "type").(string); t != "" && !strings.EqualFold(t, "User") {
			return "", nil, nil, badRequest(TypeInvalidValue, "nested groups are not supported; members must be users")
		}
		v, _ := lookupFold(obj, "value").(string)
		id := strings.ToLower(strings.TrimSpace(v))
		if !ids.Valid(id) {
			return "", nil, nil, badRequest(TypeInvalidValue, "member value %q is not a user id", v)
		}
		if !seen[id] {
			seen[id] = true
			out = append(out, id)
		}
	}
	return name, ext, out, nil
}

func (h *Handler) createGroup(w http.ResponseWriter, r *http.Request) {
	obj, _, err := readObject(r)
	if err != nil {
		writeError(w, err)
		return
	}
	doc, err := normalize(resGroup, obj)
	if err != nil {
		writeError(w, err)
		return
	}
	name, ext, want, err := groupFields(doc)
	if err != nil {
		writeError(w, err)
		return
	}
	conn := callerFrom(r.Context()).ConnectionID
	var id string
	err = h.tx(r.Context(), func(tx pgx.Tx) error {
		if err := checkMembers(r.Context(), tx, conn, want); err != nil {
			return err
		}
		if id, err = insertGroup(r.Context(), tx, conn, name, ext); err != nil {
			return err
		}
		if _, _, err := syncMembers(r.Context(), tx, id, nil, want); err != nil {
			return err
		}
		return audit.Append(r.Context(), tx, h.event(r, "scim.group.create", "group", id, map[string]any{"display_name": name, "members": len(want)}))
	})
	if err != nil {
		writeError(w, err)
		return
	}
	rep, err := h.loadGroupRep(r, id, true)
	if err != nil {
		writeError(w, err)
		return
	}
	writeResource(w, http.StatusCreated, rep)
}

func (h *Handler) replaceGroup(w http.ResponseWriter, r *http.Request) {
	id, err := pathID(r)
	if err != nil {
		writeError(w, err)
		return
	}
	obj, _, err := readObject(r)
	if err != nil {
		writeError(w, err)
		return
	}
	doc, err := normalize(resGroup, obj)
	if err != nil {
		writeError(w, err)
		return
	}
	h.updateGroup(w, r, id, true, func(groupRow, []member) (map[string]any, error) { return doc, nil })
}

func (h *Handler) patchGroup(w http.ResponseWriter, r *http.Request) {
	id, err := pathID(r)
	if err != nil {
		writeError(w, err)
		return
	}
	_, body, err := readObject(r)
	if err != nil {
		writeError(w, err)
		return
	}
	ops, err := decodePatch(body)
	if err != nil {
		writeError(w, err)
		return
	}
	q := r.URL.Query()
	respond := q.Has("attributes") || q.Has("excludedAttributes")
	h.updateGroup(w, r, id, respond, func(g groupRow, ms []member) (map[string]any, error) {
		doc := groupDoc(g, ms)
		return doc, applyPatch(resGroup, doc, ops, g.ID)
	})
}

// updateGroup applies a replacement or patch under a row lock and records name and membership changes.
func (h *Handler) updateGroup(w http.ResponseWriter, r *http.Request, id string, respond bool, next func(groupRow, []member) (map[string]any, error)) {
	conn := callerFrom(r.Context()).ConnectionID
	err := h.tx(r.Context(), func(tx pgx.Tx) error {
		g, err := loadGroup(r.Context(), tx, conn, id, true)
		if err != nil {
			return err
		}
		if err := checkIfMatch(r, g.UpdatedAt); err != nil {
			return err
		}
		current, err := groupMembers(r.Context(), tx, []string{g.ID})
		if err != nil {
			return err
		}
		doc, err := next(g, current[g.ID])
		if err != nil {
			return err
		}
		name, ext, want, err := groupFields(doc)
		if err != nil {
			return err
		}
		have := map[string]bool{}
		for _, m := range current[g.ID] {
			have[m.ID] = true
		}
		var fresh []string
		for _, uid := range want {
			if !have[uid] {
				fresh = append(fresh, uid)
			}
		}
		if err := checkMembers(r.Context(), tx, conn, fresh); err != nil {
			return err
		}
		renamed := name != g.DisplayName || (ext == nil) != (g.ExternalID == g.ID) || (ext != nil && *ext != g.ExternalID)
		if renamed {
			if err := renameGroup(r.Context(), tx, g.ID, name, ext); err != nil {
				return err
			}
			if err := audit.Append(r.Context(), tx, h.event(r, "scim.group.update", "group", g.ID,
				map[string]any{"display_name": name, "previous_display_name": g.DisplayName})); err != nil {
				return err
			}
		}
		added, removed, err := syncMembers(r.Context(), tx, g.ID, current[g.ID], want)
		if err != nil || len(added)+len(removed) == 0 {
			return err
		}
		return audit.Append(r.Context(), tx, h.event(r, "scim.group.members", "group", g.ID,
			map[string]any{"display_name": name, "added": capList(added), "removed": capList(removed), "added_count": len(added), "removed_count": len(removed)}))
	})
	if err != nil {
		writeError(w, err)
		return
	}
	if !respond {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	q := r.URL.Query()
	lq := listQuery{attrs: splitList(q.Get("attributes")), excluded: splitList(q.Get("excludedAttributes"))}
	rep, err := h.loadGroupRep(r, id, wants(lq, "members"))
	if err != nil {
		writeError(w, err)
		return
	}
	writeResource(w, http.StatusOK, project(rep, lq.attrs, lq.excluded))
}

func (h *Handler) deleteGroup(w http.ResponseWriter, r *http.Request) {
	id, err := pathID(r)
	if err != nil {
		writeError(w, err)
		return
	}
	conn := callerFrom(r.Context()).ConnectionID
	err = h.tx(r.Context(), func(tx pgx.Tx) error {
		g, err := loadGroup(r.Context(), tx, conn, id, true)
		if err != nil {
			return err
		}
		if err := checkIfMatch(r, g.UpdatedAt); err != nil {
			return err
		}
		if err := dropGroup(r.Context(), tx, g.ID); err != nil {
			return err
		}
		return audit.Append(r.Context(), tx, h.event(r, "scim.group.delete", "group", g.ID, map[string]any{"display_name": g.DisplayName}))
	})
	if err != nil {
		writeError(w, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// capList keeps audit details bounded for very large membership changes.
func capList(xs []string) []string {
	if len(xs) > 50 {
		return xs[:50]
	}
	return xs
}
