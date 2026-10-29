//! Actual CUDA family facts and loaded drafter identities, without opening the CUDA driver.
const std = @import("std");
const check = @import("draft_sources_check.zig").check;
const present = @import("draft_sources_check.zig").present;

test "draft sources: CUDA Nemotron and Bolt copy switch and loaded MTP head" {
    const n = @import("nemotron");
    var e: n.Engine = undefined;
    e.c = undefined;
    e.c.latent = 0;
    e.c.bolt = false;
    var b: n.Lanes = undefined;
    b.e = &e;
    b.head = null;
    b.copy_drafts = true;
    b.mtp_drafts = null;
    b.mtp_confidence = null;
    b.cost_count = 0;
    b.mtp_ms = 0;
    try check(b.facts(), null, true, false);
    b.head = present(@TypeOf(b.head.?));
    try check(b.facts(), null, true, true);
    b.mtp_drafts = 0;
    try check(b.facts(), null, true, false);
    b.head = null;
    b.copy_drafts = false;
    try check(b.facts(), null, false, false);
}

test "draft sources: CUDA GLM distinguishes its checkpoint head from loaded DFlash2" {
    const g = @import("glm_cuda");
    var e: g.engine.Engine = undefined;
    var b: g.lanes.Lanes = undefined;
    b.x.e = &e;
    b.alone = false;
    b.mtp_confidence = null;
    e.mtp = false;
    e.drafter = null;
    try check(b.facts(), null, false, false);
    e.mtp = true;
    try check(b.facts(), null, true, true);
    b.mtp_confidence = 0.3; // --mtp-confidence: the head's chances size its windows, still a checkpoint head
    try check(b.facts(), null, true, true);
    b.mtp_confidence = null;
    e.drafter = present(@TypeOf(e.drafter.?));
    try check(b.facts(), .{ .kind = "DFlash2", .name = "example/draft" }, true, false);
}

test "draft sources: CUDA Gemma, Spark, Kolibri and Dory have copy windows and no checkpoint head" {
    const g = @import("gemma4_cuda");
    var gb: g.backend.Gemma = undefined;
    var gm: @typeInfo(@TypeOf(gb.m)).pointer.child = undefined;
    gm.slots = 4;
    gb.m = &gm;
    try check(gb.facts(), null, true, false);
    const s = @import("spark25");
    var sb: s.backend.Spark = undefined;
    var sm: @typeInfo(@TypeOf(sb.m)).pointer.child = undefined;
    sm.slots = 4;
    sb.m = &sm;
    try check(sb.facts(), null, true, false);
    const k = @import("kolibri1");
    var kb: k.backend.Kolibri = undefined;
    var km: @typeInfo(@TypeOf(kb.m)).pointer.child = undefined;
    km.slots = 4;
    kb.m = &km;
    try check(kb.facts(), null, true, false);
    const d = @import("dory");
    var dm: d.cuda_model.Model = undefined;
    dm.slots = 4;
    dm.paged = false;
    var db: d.cuda_backend.Dory = undefined;
    db.m = &dm;
    try check(db.facts(), null, true, false);
}

test "draft sources: CUDA loaders identify their actual external draft implementations" {
    const q = @import("qwen35");
    var qd: q.draft.Draft = undefined;
    var qm: q.mtp.Mtp = undefined;
    const g = @import("gemma4_cuda");
    var gd: g.dflash.Drafter = undefined;
    var ga: g.assistant.Drafter = undefined;
    try std.testing.expectEqualStrings("DFlash2", qd.drafter().kind);
    try std.testing.expectEqualStrings("MTP", qm.drafter().kind);
    try std.testing.expectEqualStrings("DFlash", gd.drafter().kind);
    try std.testing.expectEqualStrings("MTP assistant", ga.drafter().kind);
}

test "draft sources: CUDA Qwen dense and MoE share copy-only target facts" {
    try check(@import("qwen35").native.laneFacts(16, 64), null, true, false);
    try check(@import("qwen35").native.laneFacts(1, 64), null, false, false);
}

test "draft sources: CUDA Flash Next follows the loaded checkpoint head" {
    var b: @import("flashnext").native.Cuda = undefined;
    b.head = null;
    b.depth = 8;
    try check(b.facts(), null, true, false);
    b.head = present(@TypeOf(b.head.?));
    try check(b.facts(), null, true, true);
}
