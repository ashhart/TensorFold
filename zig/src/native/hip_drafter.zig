//! --drafter on the HIP host: its refusals before any load, and the family's DFlash2 drafter behind Drafted.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const fam = @import("hip_families.zig");
const Allocator = std.mem.Allocator;

/// Why --drafter cannot run under tensor parallelism: the drafter and its taps live on one rank.
const tp_refusal = "--drafter with --tp {d}: the HIP engine runs its DFlash2 drafter on one rank; " ++
    "serve with --tp 1, or without --drafter";
/// Why --drafter-bits 0 is refused: the HIP drafter quantizes its weights to 4 bits.
const bits_refusal = "--drafter-bits {d}: the HIP engine runs the DFlash2 drafter at 4 bits; " ++
    "serve with --drafter-bits 4";

/// Why the group cannot take --drafter, on every rank before any load: the drafter runs on one rank at 4 bits.
pub fn groupRefusal(a: Allocator, o: api.Open) ?[]const u8 {
    if (o.drafter == null) return null;
    if (o.tp > 1) return std.fmt.allocPrint(a, tp_refusal, .{o.tp}) catch tp_refusal;
    if (o.drafter_bits != 4) return std.fmt.allocPrint(a, bits_refusal, .{o.drafter_bits}) catch bits_refusal;
    return null;
}

/// Why a HIP family without a DFlash2 drafter refuses --drafter.
fn familyRefusal(comptime F: type) []const u8 {
    const what = if (@hasDecl(F, "model_type")) F.model_type ++ " checkpoints load" else "this family loads";
    return "--drafter: " ++ what ++ " no DFlash2 drafter on HIP; serve without --drafter";
}

/// The family's refusal of --drafter before it loads: every family but one that declares `takes_drafter`.
fn refused(comptime F: type, o: api.Open) ?[]const u8 {
    if (o.drafter != null and !(@hasDecl(F, "takes_drafter") and F.takes_drafter)) return familyRefusal(F);
    return null;
}

/// The family's refusals before it loads or joins a group (hip_families.zig), then its refusal of --drafter.
pub fn flagRefused(comptime F: type, o: api.Open) ?[]const u8 {
    return fam.refused(F, o) orelse refused(F, o);
}

/// The drafter the family loads: --drafter's, none under --no-drafts, which serves the target alone (as on CUDA).
pub fn drafterOf(o: api.Open) ?[]const u8 {
    return if (o.drafts) o.drafter else null;
}

/// The serving line's drafter: the DFlash2 drafter --drafter attached, by the name it was given (as on CUDA).
pub fn draftSources(o: api.Open, drafted: ?*const lanes.drafted.Drafted) api.DraftSources {
    if (drafted == null) return .{};
    return .{ .drafter = .{ .kind = "DFlash2", .name = o.drafter_name orelse o.drafter orelse "" } };
}

/// What the host serves: the family's backend and facts as it loaded them, or behind Drafted over its drafter.
pub const Served = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    /// The lane core's wrapper over the family's backend and drafter (null: served as it loaded).
    drafted: ?*lanes.drafted.Drafted = null,
};

/// `loaded` behind Drafted when it loaded a drafter; null with `problem`, the model freed, when it cannot attach.
pub fn attach(a: Allocator, gpa: Allocator, loaded: anytype, o: api.Open, problem: *[]const u8) Allocator.Error!?Served {
    const as_loaded: Served = .{ .backend = loaded.backend, .facts = loaded.facts };
    const d = if (comptime @hasField(@TypeOf(loaded), "drafter")) loaded.drafter orelse return as_loaded else return as_loaded;
    // the drafter's taps were set before the streams were sized; the wrapper's prepare keeps them
    const x = try gpa.create(lanes.drafted.Drafted);
    x.* = lanes.drafted.Drafted.init(gpa, loaded.backend, d) catch |err| {
        gpa.destroy(x);
        // the message first: a failed print returns through the caller's errdefer, which frees the model once
        problem.* = try std.fmt.allocPrint(a, "the drafter {s} cannot attach to this model ({s})", .{ o.drafter orelse "", @errorName(err) });
        loaded.deinit(loaded.ctx);
        return null;
    };
    return .{ .backend = x.backend(), .facts = x.facts(loaded.facts), .drafted = x };
}

/// Frees the wrapper attach made, after the lane core that ran over it; the family frees its drafter.
pub fn release(gpa: Allocator, drafted: ?*lanes.drafted.Drafted) void {
    const x = drafted orelse return;
    x.deinit();
    gpa.destroy(x);
}

const qwen35 = @import("qwen3_5");
const flashnext = @import("flashnext");
const nemotron = @import("nemotron");

/// A family's open as attach takes it: its drafter, if it loaded one, and the free of the whole model.
const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    drafter: ?lanes.drafter.Drafter = null,
    ctx: *anyopaque = undefined,
    deinit: *const fn (*anyopaque) void = undefined,
};

/// A drafter attach only wraps: it gives its taps and facts, and no stream ever reaches the rest.
const Idle = struct {
    depth: u32 = 3,

    fn drafter(x: *Idle) lanes.drafter.Drafter {
        return .{ .ptr = x, .vtable = &.{ .taps = taps, .facts = facts, .open = open, .absorb = absorb, .hold = hold, .held = held, .release = close } };
    }
    fn taps(_: *anyopaque) []const u32 {
        return &.{ 5, 19 };
    }
    fn facts(ptr: *anyopaque) lanes.drafter.Facts {
        return .{ .depth = @as(*Idle, @ptrCast(@alignCast(ptr))).depth, .step_ms = 1 };
    }
    fn open(_: *anyopaque, _: *lanes.Stream) anyerror!void {
        return error.Unused;
    }
    fn absorb(_: *anyopaque, _: []const lanes.drafter.Absorb) anyerror!void {
        return error.Unused;
    }
    fn hold(_: *anyopaque, _: []const lanes.drafter.Hold) anyerror!void {
        return error.Unused;
    }
    fn held(_: *anyopaque, _: []const *lanes.Stream, _: []const []u32) anyerror!void {
        return error.Unused;
    }
    fn close(_: *anyopaque, _: *lanes.Stream) void {}
    /// A target that refuses the drafter's taps, as Drafted.init's prepare_features may.
    fn refuseTaps(_: *anyopaque, _: []const u32) anyerror!void {
        return error.TapsMismatch;
    }
    /// The model's free, counted in the u32 behind `ctx`.
    fn free(ctx: *anyopaque) void {
        @as(*u32, @ptrCast(@alignCast(ctx))).* += 1;
    }
};

test "--drafter with --tp > 1 is refused by name, on rank 0 and on the ranks that follow it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const group: api.Open = .{ .dir = "d", .model_type = "qwen3_5", .drafter = "/models/drafter", .tp = 2, .master = "node0" };
    const tp = "--drafter with --tp 2: the HIP engine runs its DFlash2 drafter on one rank; serve with --tp 1, or without --drafter";
    try std.testing.expectEqualStrings(tp, groupRefusal(a, group) orelse "");
    var follower = group;
    follower.rank = 1;
    try std.testing.expectEqualStrings(tp, groupRefusal(a, follower) orelse "");
}

test "--drafter passes the group check on one rank at 4 bits; --drafter-bits 0 is refused; no --drafter, nothing" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const drafted: api.Open = .{ .dir = "d", .model_type = "qwen3_5", .drafter = "/models/drafter" };
    try std.testing.expect(groupRefusal(a, drafted) == null);
    var bf16 = drafted;
    bf16.drafter_bits = 0;
    const bits = "--drafter-bits 0: the HIP engine runs the DFlash2 drafter at 4 bits; serve with --drafter-bits 4";
    try std.testing.expectEqualStrings(bits, groupRefusal(a, bf16) orelse "");
    try std.testing.expect(groupRefusal(a, .{ .dir = "d", .model_type = "qwen3_5", .tp = 2, .drafter_bits = 0 }) == null);
}

test "--drafter loads on the dense qwen3_5 family; every other HIP family refuses it by name before it loads" {
    const drafted: api.Open = .{ .dir = "d", .model_type = "qwen3_5", .drafter = "/models/drafter" };
    try std.testing.expect(refused(qwen35.native, drafted) == null);
    const by_name = .{
        .{ qwen35.native_moe, "--drafter: qwen3_5_moe checkpoints load no DFlash2 drafter on HIP; serve without --drafter" },
        .{ flashnext.native, "--drafter: qwen4_exp checkpoints load no DFlash2 drafter on HIP; serve without --drafter" },
        .{ nemotron.native, "--drafter: nemotron_h checkpoints load no DFlash2 drafter on HIP; serve without --drafter" },
    };
    inline for (by_name) |f| try std.testing.expectEqualStrings(f[1], refused(f[0], drafted) orelse "");
    const plain: api.Open = .{ .dir = "d", .model_type = "m" };
    inline for (.{ qwen35.native, qwen35.native_moe, flashnext.native, nemotron.native }) |F| try std.testing.expect(refused(F, plain) == null);
}

test "a plain serve and --no-drafts with --drafter load no drafter and serve as loaded; --drafter loads its own" {
    const plain: api.Open = .{ .dir = "d", .model_type = "qwen3_5" };
    try std.testing.expect(drafterOf(plain) == null);
    var serial = plain;
    serial.drafter = "/models/drafter";
    serial.drafts = false;
    try std.testing.expect(drafterOf(serial) == null);
    try std.testing.expect(refused(qwen35.native, serial) == null);
    const drafted: api.Open = .{ .dir = "d", .model_type = "qwen3_5", .drafter = "/models/drafter" };
    try std.testing.expectEqualStrings("/models/drafter", drafterOf(drafted) orelse "");
    var fake: lanes.fake.Fake = .{ .gpa = std.testing.allocator };
    const Bare = struct { backend: lanes.backend.Backend, facts: lanes.Model };
    var problem: []const u8 = "";
    inline for (.{ Bare{ .backend = fake.backend(), .facts = undefined }, Loaded{ .backend = fake.backend(), .facts = undefined } }) |loaded| {
        const served = (try attach(std.testing.allocator, std.testing.allocator, loaded, plain, &problem)) orelse return error.TestUnexpectedResult;
        try std.testing.expect(served.drafted == null and served.backend.ptr == loaded.backend.ptr);
        release(std.testing.allocator, served.drafted);
    }
}

test "a loaded drafter is served behind Drafted over the family's backend, the model kept; release frees the wrapper" {
    var fake: lanes.fake.Fake = .{ .gpa = std.testing.allocator };
    var idle: Idle = .{};
    var frees: u32 = 0;
    const loaded: Loaded = .{ .backend = fake.backend(), .facts = .{ .exact_width = 8 }, .drafter = idle.drafter(), .ctx = &frees, .deinit = Idle.free };
    const drafted: api.Open = .{ .dir = "d", .model_type = "qwen3_5", .drafter = "/models/drafter" };
    var problem: []const u8 = "";
    const served = (try attach(std.testing.allocator, std.testing.allocator, loaded, drafted, &problem)) orelse return error.TestUnexpectedResult;
    defer release(std.testing.allocator, served.drafted); // std.testing.allocator fails the test if the wrapper leaks
    const x = served.drafted orelse return error.TestUnexpectedResult;
    try std.testing.expect(served.backend.ptr == @as(*anyopaque, x) and x.target.ptr == loaded.backend.ptr);
    try std.testing.expect(served.facts.mtp and served.facts.drafts == idle.depth);
    try std.testing.expect(frees == 0 and problem.len == 0);
}

test "a drafter that cannot attach frees the model once and says why; a failed message leaves the free to the caller" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var fake: lanes.fake.Fake = .{ .gpa = std.testing.allocator };
    var target = fake.backend();
    var refusing = target.vtable.*;
    refusing.prepare_features = Idle.refuseTaps;
    target.vtable = &refusing;
    var idle: Idle = .{};
    var frees: u32 = 0;
    const loaded: Loaded = .{ .backend = target, .facts = .{}, .drafter = idle.drafter(), .ctx = &frees, .deinit = Idle.free };
    const drafted: api.Open = .{ .dir = "d", .model_type = "qwen3_5", .drafter = "/models/drafter" };
    var problem: []const u8 = "";
    try std.testing.expect(try attach(arena.allocator(), std.testing.allocator, loaded, drafted, &problem) == null);
    try std.testing.expectEqualStrings("the drafter /models/drafter cannot attach to this model (TapsMismatch)", problem);
    try std.testing.expectEqual(@as(u32, 1), frees);
    frees = 0;
    var failing: std.testing.FailingAllocator = .init(arena.allocator(), .{ .fail_index = 0 });
    try std.testing.expectError(error.OutOfMemory, attach(failing.allocator(), std.testing.allocator, loaded, drafted, &problem));
    try std.testing.expectEqual(@as(u32, 0), frees); // the caller's errdefer frees it, once
}

test "draft sources: HIP names the DFlash2 drafter --drafter attached; without one the line is unchanged" {
    const check = @import("draft_sources_check.zig").check;
    const present = @import("draft_sources_check.zig").present;
    var e: qwen35.engine.Engine = undefined;
    var b: qwen35.hip_lanes.Hip = undefined;
    b.e = &e;
    b.costs = .{};
    b.head = null;
    e.o.batch_rows = 16;
    e.o.policy.mtp.drafts = 3;
    const plain: api.Open = .{ .dir = "d", .model_type = "qwen3_5" };
    var named = plain;
    named.drafter = "/cache/drafter";
    try check(b.facts(), draftSources(plain, null).drafter, true, false);
    try check(b.facts(), draftSources(named, null).drafter, true, false);
    var facts = b.facts();
    facts.mtp = true; // as Drafted.facts marks the target it wraps
    facts.speculate = true;
    const x = present(*lanes.drafted.Drafted);
    const shown = draftSources(named, x).drafter orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("DFlash2", shown.kind);
    try std.testing.expectEqualStrings("/cache/drafter", shown.name);
    try check(facts, shown, true, false);
    named.drafter_name = "z-lab/Qwen3.8-27B-DFlash2";
    const given = draftSources(named, x).drafter orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("z-lab/Qwen3.8-27B-DFlash2", given.name);
}
