package scim

import (
	"errors"
	"net/http"
	"sort"
	"strings"

	"github.com/jackc/pgx/v5"

	"github.com/ashhart/TensorFold/starlight/internal/audit"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
)

func (h *Handler) listUsers(w http.ResponseWriter, r *http.Request) {
	lq, err := parseList(r)
	if err != nil {
		writeError(w, err)
		return
	}
	conn := callerFrom(r.Context()).ConnectionID
	users, total, err := queryUsers(r.Context(), h.pool, conn, lq)
	if err != nil {
		writeError(w, err)
		return
	}
	var groups map[string][]groupRef
	if wants(lq, "groups") {
		userIDs := make([]string, len(users))
		for i, u := range users {
			userIDs[i] = u.ID
		}
		if groups, err = userGroups(r.Context(), h.pool, conn, userIDs); err != nil {
			writeError(w, err)
			return
		}
	}
	items := make([]map[string]any, 0, len(users))
	for _, u := range users {
		items = append(items, project(h.userRep(u, groups[u.ID]), lq.attrs, lq.excluded))
	}
	h.writeList(w, lq, total, items)
}

// wants reports whether a list or get request will return the attribute, so it is worth loading.
func wants(lq listQuery, name string) bool {
	for _, e := range lq.excluded {
		if strings.EqualFold(e, name) {
			return false
		}
	}
	if len(lq.attrs) == 0 {
		return true
	}
	for _, a := range lq.attrs {
		if strings.EqualFold(a, name) || strings.HasPrefix(strings.ToLower(a), strings.ToLower(name)+".") {
			return true
		}
	}
	return false
}

func (h *Handler) getUser(w http.ResponseWriter, r *http.Request) {
	id, err := pathID(r)
	if err != nil {
		writeError(w, err)
		return
	}
	q := r.URL.Query()
	lq := listQuery{attrs: splitList(q.Get("attributes")), excluded: splitList(q.Get("excludedAttributes"))}
	rep, err := h.loadUserRep(r, id)
	if err != nil {
		writeError(w, err)
		return
	}
	writeResource(w, http.StatusOK, project(rep, lq.attrs, lq.excluded))
}

func (h *Handler) loadUserRep(r *http.Request, id string) (map[string]any, error) {
	conn := callerFrom(r.Context()).ConnectionID
	u, err := loadUser(r.Context(), h.pool, conn, id, false)
	if err != nil {
		return nil, err
	}
	groups, err := userGroups(r.Context(), h.pool, conn, []string{u.ID})
	if err != nil {
		return nil, err
	}
	return h.userRep(u, groups[u.ID]), nil
}

func (h *Handler) createUser(w http.ResponseWriter, r *http.Request) {
	obj, _, err := readObject(r)
	if err != nil {
		writeError(w, err)
		return
	}
	doc, err := normalize(resUser, obj)
	if err != nil {
		writeError(w, err)
		return
	}
	f, active, err := fieldsFrom(doc)
	if err != nil {
		writeError(w, err)
		return
	}
	isActive := active == nil || *active
	conn := callerFrom(r.Context()).ConnectionID
	var id string
	err = h.tx(r.Context(), func(tx pgx.Tx) error {
		adoptable, err := conflictCheck(r.Context(), tx, conn, "", f)
		if err != nil {
			return err
		}
		action := "scim.user.create"
		if adoptable != "" {
			id, action = adoptable, "scim.user.adopt"
			u, err := loadUser(r.Context(), tx, conn, adoptable, true)
			if err != nil {
				return err
			}
			if err := writeUser(r.Context(), tx, conn, u.ID, f); err != nil {
				return err
			}
			if _, err := setActive(r.Context(), tx, u, isActive); err != nil {
				return err
			}
		} else if id, err = insertUser(r.Context(), tx, conn, f, isActive); err != nil {
			return err
		}
		return audit.Append(r.Context(), tx, h.event(r, action, "user", id, userDetails(f, isActive)))
	})
	if errors.Is(err, errLastOwner) {
		h.refused(r, "scim.user.suspend", id, f.UserName)
	}
	if err != nil {
		writeError(w, err)
		return
	}
	rep, err := h.loadUserRep(r, id)
	if err != nil {
		writeError(w, err)
		return
	}
	writeResource(w, http.StatusCreated, rep)
}

func (h *Handler) replaceUser(w http.ResponseWriter, r *http.Request) {
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
	doc, err := normalize(resUser, obj)
	if err != nil {
		writeError(w, err)
		return
	}
	h.updateUser(w, r, id, func(userRow) (map[string]any, error) { return doc, nil })
}

func (h *Handler) patchUser(w http.ResponseWriter, r *http.Request) {
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
	h.updateUser(w, r, id, func(u userRow) (map[string]any, error) {
		doc := userDoc(u)
		return doc, applyPatch(resUser, doc, ops, u.ID)
	})
}

// updateUser applies a replacement or patch under a row lock, then records status and attribute changes.
func (h *Handler) updateUser(w http.ResponseWriter, r *http.Request, id string, next func(userRow) (map[string]any, error)) {
	conn := callerFrom(r.Context()).ConnectionID
	var name string
	err := h.tx(r.Context(), func(tx pgx.Tx) error {
		u, err := loadUser(r.Context(), tx, conn, id, true)
		if err != nil {
			return err
		}
		name = u.name()
		if err := checkIfMatch(r, u.UpdatedAt); err != nil {
			return err
		}
		doc, err := next(u)
		if err != nil {
			return err
		}
		f, active, err := fieldsFrom(doc)
		if err != nil {
			return err
		}
		if _, err := conflictCheck(r.Context(), tx, conn, u.ID, f); err != nil {
			return err
		}
		if err := writeUser(r.Context(), tx, conn, u.ID, f); err != nil {
			return err
		}
		changed := false
		// Status changes take the owner lock, so they run before audit appends take the chain lock, as in the console.
		if active != nil {
			if changed, err = setActive(r.Context(), tx, u, *active); err != nil {
				return err
			}
		}
		isActive := u.Status == directory.Active
		action := "scim.user.update"
		if !u.owned() {
			action = "scim.user.adopt"
		}
		if err := audit.Append(r.Context(), tx, h.event(r, action, "user", u.ID, userDetails(f, isActive))); err != nil {
			return err
		}
		if !changed {
			return nil
		}
		statusAction := "scim.user.suspend"
		if *active {
			statusAction = "scim.user.reactivate"
		}
		return audit.Append(r.Context(), tx, h.event(r, statusAction, "user", u.ID, map[string]any{"user_name": f.UserName}))
	})
	if errors.Is(err, errLastOwner) {
		h.refused(r, "scim.user.suspend", id, name)
	}
	if err != nil {
		writeError(w, err)
		return
	}
	rep, err := h.loadUserRep(r, id)
	if err != nil {
		writeError(w, err)
		return
	}
	writeResource(w, http.StatusOK, rep)
}

func (h *Handler) deleteUser(w http.ResponseWriter, r *http.Request) {
	id, err := pathID(r)
	if err != nil {
		writeError(w, err)
		return
	}
	conn := callerFrom(r.Context()).ConnectionID
	var name string
	err = h.tx(r.Context(), func(tx pgx.Tx) error {
		u, err := loadUser(r.Context(), tx, conn, id, true)
		if err != nil {
			return err
		}
		name = u.name()
		if err := checkIfMatch(r, u.UpdatedAt); err != nil {
			return err
		}
		if err := removeUser(r.Context(), tx, conn, u); err != nil {
			return err
		}
		return audit.Append(r.Context(), tx, h.event(r, "scim.user.delete", "user", u.ID, map[string]any{"user_name": name}))
	})
	if errors.Is(err, errLastOwner) {
		h.refused(r, "scim.user.delete", id, name)
	}
	if err != nil {
		writeError(w, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func userDetails(f userFields, active bool) map[string]any {
	d := map[string]any{"user_name": f.UserName, "email": f.Email, "active": active}
	if f.ExternalID != nil {
		d["external_id"] = *f.ExternalID
	}
	return d
}

// userRep renders the full SCIM representation, before attribute projection.
func (h *Handler) userRep(u userRow, groups []groupRef) map[string]any {
	doc := userDoc(u)
	doc["id"] = u.ID
	schemas := []string{URNUser}
	var extra []string
	for k := range doc {
		if len(k) > 4 && strings.EqualFold(k[:4], "urn:") {
			extra = append(extra, k)
		}
	}
	sort.Strings(extra)
	doc["schemas"] = append(schemas, extra...)
	gs := make([]any, 0, len(groups))
	for _, g := range groups {
		gs = append(gs, map[string]any{"value": g.ID, "display": g.DisplayName, "$ref": h.base + "/Groups/" + g.ID, "type": "direct"})
	}
	doc["groups"] = gs
	doc["meta"] = h.meta("User", u.ID, u.CreatedAt, u.UpdatedAt)
	return doc
}
