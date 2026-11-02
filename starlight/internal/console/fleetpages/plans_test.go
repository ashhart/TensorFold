package fleetpages_test

import (
	"net/http"
	"net/url"
	"strings"
	"testing"
)

func TestPlanApprovalUsesTheReviewedHash(t *testing.T) {
	r := newRig(t)
	owner := r.user("owner@example.com", "owner", "")
	author := r.user("pat@example.com", "", "member")
	outsider := r.user("eve@example.com", "", "")
	id, sum := r.plan(author.user, "1. Read the parser.\n2. Port it to the new tokenizer.")
	if _, page := author.get("/console/plans"); !strings.Contains(page, "Port the parser") || strings.Contains(page, "Waiting for approval") {
		t.Fatal("the author's plan list is wrong")
	}
	code, page := author.get("/console/plans/" + id)
	if code != http.StatusOK || !strings.Contains(page, "Port it to the new tokenizer.") || strings.Contains(page, "Approve this plan") ||
		!strings.Contains(page, "GET /v1/starlight/plans/"+id) {
		t.Fatalf("author view %d", code)
	}
	if code, _ := outsider.get("/console/plans/" + id); code != http.StatusNotFound {
		t.Fatalf("an outsider opening the plan answered %d", code)
	}
	if code, _ := author.post("/console/plans/"+id+"/approve", url.Values{"plan_sha256": {sum}}); code != http.StatusForbidden {
		t.Fatalf("self-approval without the approver role answered %d", code)
	}
	if _, page := owner.get("/console/plans"); !strings.Contains(page, "Waiting for approval") || !strings.Contains(page, "/console/plans/"+id) {
		t.Fatal("the approver does not see the pending plan")
	}
	if _, page := owner.get("/console/plans/" + id); !strings.Contains(page, `name="plan_sha256" value="`+sum+`"`) {
		t.Fatal("the approve form does not carry the hash of the text shown")
	}
	if code, _ := owner.post("/console/plans/"+id+"/approve", url.Values{"plan_sha256": {strings.Repeat("0", 64)}}); code != http.StatusConflict {
		t.Fatalf("approving another hash answered %d", code)
	}
	if code, page := owner.follow("/console/plans/"+id+"/approve", url.Values{"plan_sha256": {sum}}); code != http.StatusSeeOther ||
		!strings.Contains(page, "Plan approved.") || !strings.Contains(page, "by owner@example.com") {
		t.Fatalf("approve %d", code)
	}
	if r.scalar("SELECT status FROM plans WHERE id = $1", id) != "approved" || len(r.plans.calls) != 2 || r.plans.calls[1] != "approve:"+owner.user.ID {
		t.Fatalf("decider calls %v", r.plans.calls)
	}
	tampered, _ := r.plan(author.user, "original text")
	r.exec("UPDATE plans SET plan_sha256 = decode(repeat('ab', 32), 'hex') WHERE id = $1", tampered)
	if _, page := owner.get("/console/plans/" + tampered); !strings.Contains(page, "does not match its recorded hash") || strings.Contains(page, "Approve this plan") {
		t.Fatal("a plan whose text does not match its hash can still be approved")
	}
	rejected, _ := r.plan(author.user, "reject me")
	if code, _ := owner.post("/console/plans/"+rejected+"/reject", nil); code != http.StatusSeeOther || r.scalar("SELECT status FROM plans WHERE id = $1", rejected) != "rejected" {
		t.Fatalf("reject %d", code)
	}
}
