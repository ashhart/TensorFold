//! --drafter on the CUDA host: a family's draft model behind the lane core's Drafted wrapper over its backend.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const Allocator = std.mem.Allocator;
const startup = @import("cuda_startup_costs.zig");

/// What the host serves: the family's backend as is, or wrapped with a drafter in front.
pub const Served = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    stream_bytes: usize,
    held: ?*Held = null,
    source: ?api.DraftSources.Named = null,
    prepare: ?startup.Prepare = null,
    /// The family's own per-stream buffers, when its Loaded pre-allocates them.
    family: ?struct { ctx: *anyopaque, reserve: *const fn (*anyopaque, u32) anyerror!void } = null,

    /// The family's and the drafter's per-stream buffers for `streams`, once the server knows the count.
    pub fn reserve(s: *Served, streams: u32) !void {
        if (s.family) |f| try f.reserve(f.ctx, streams);
        if (s.held) |h| try h.reserve(h.ctx, streams);
    }

    pub fn calibrate(s: *Served, gpa: Allocator, cfg: *lanes.Config) !void {
        try startup.calibrate(s, gpa, cfg);
    }

    pub fn close(s: *Served, gpa: Allocator) void {
        if (s.held) |h| {
            h.drafted.deinit();
            h.close(h.ctx);
            gpa.destroy(h);
        }
        s.held = null;
    }
};

/// A family that times its windows once its pool attaches: prices into `s.facts`; false with `problem` set.
pub fn price(comptime F: type, a: Allocator, io: std.Io, ctx: anytype, dir: []const u8, family: *anyopaque, s: *Served, problem: *[]const u8) !bool {
    if (comptime !@hasDecl(F, "priceWindows")) return true;
    s.facts = F.priceWindows(family, io, ctx, dir, s.facts) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "the native CUDA engine cannot time its windows ({s})", .{@errorName(e)});
        return false;
    };
    return true;
}

const Held = struct { drafted: lanes.drafted.Drafted, ctx: *anyopaque, close: *const fn (*anyopaque) void, reserve: *const fn (*anyopaque, u32) anyerror!void };

/// `loaded`'s backend, wrapped with the drafter `o.drafter` names when it does; null with `problem` set on refusal.
pub fn wrap(comptime F: type, a: Allocator, gpa: Allocator, io: std.Io, loaded: anytype, o: api.Open, problem: *[]const u8) !?Served {
    var s: Served = .{ .backend = loaded.backend, .facts = loaded.facts, .stream_bytes = loaded.stream_bytes };
    defer if (@hasDecl(F, "finishDrafterOpen")) F.finishDrafterOpen(loaded.ctx);
    if (@hasField(@TypeOf(loaded), "draft_kind")) if (loaded.draft_kind) |kind| {
        s.source = .{ .kind = kind, .name = o.drafter_name orelse o.drafter orelse "" };
    };
    if (@hasField(@TypeOf(loaded), "reserve")) s.family = .{ .ctx = loaded.ctx, .reserve = loaded.reserve };
    const dir = o.drafter orelse return s;
    if (!o.drafts) return s;
    if (!@hasDecl(F, "openDrafter")) {
        // a family that loads its drafter itself from its options (GLM's DFlash2) serves as it loaded
        if (comptime @hasDecl(F, "Options") and @hasField(F.Options, "drafter")) return s;
        problem.* = try std.fmt.allocPrint(a, "--drafter: {s} checkpoints load no draft model on CUDA", .{F.model_type});
        return null;
    }
    const opened = F.openDrafter(gpa, io, loaded.ctx, dir, o.drafter_bits) catch |e| {
        if (e == error.MtpDepthZero) return s;
        if (@hasDecl(F, "explain")) if (F.explain(loaded.ctx, e)) |why| {
            problem.* = why;
            return null;
        };
        problem.* = try std.fmt.allocPrint(a, "the native CUDA engine cannot load the drafter {s} ({s})", .{ dir, @errorName(e) });
        return null;
    };
    var held: ?*Held = null;
    var drafted_initialized = false;
    var transferred = false;
    defer if (!transferred) {
        if (held) |h| {
            if (drafted_initialized) h.drafted.deinit();
            gpa.destroy(h);
        }
        opened.close(opened.ctx);
    };
    const h = try gpa.create(Held);
    held = h;
    h.* = .{ .drafted = undefined, .ctx = opened.ctx, .close = opened.close, .reserve = opened.reserve };
    h.drafted = lanes.drafted.Drafted.init(gpa, loaded.backend, opened.drafter) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "the drafter {s} cannot attach to this model ({s})", .{ dir, @errorName(e) });
        return null;
    };
    drafted_initialized = true;
    // a family whose Opened asks for it serves logprobs' first rows through the wrapper; others keep main's table
    if (@hasField(@TypeOf(opened), "first_rows")) h.drafted.first_rows = opened.first_rows;
    s.source = .{ .kind = opened.drafter.kind, .name = o.drafter_name orelse dir };
    s.backend = h.drafted.backend();
    s.facts = h.drafted.facts(loaded.facts);
    s.stream_bytes = try h.drafted.streamBytes(loaded.stream_bytes);
    s.held = h;
    transferred = true;
    if (@hasField(@TypeOf(opened), "prepare")) {
        // a family's prepare, or an optional one when only some drafters time windows after the pool (not Qwen's MTP)
        const P = @TypeOf(opened.prepare);
        const maybe: ?(if (@typeInfo(P) == .optional) @typeInfo(P).optional.child else P) = opened.prepare;
        if (maybe) |p| s.prepare = .{ .ctx = p.ctx, .run = p.run, .bytes = if (@hasField(@TypeOf(p), "bytes")) p.bytes else 0 };
    }
    return s;
}

test "--drafter: a family that loads its own drafter (GLM's DFlash2) serves as it loaded, and one with neither refuses" {
    const F = @import("glm_cuda").native;
    try std.testing.expect(@hasField(F.Options, "drafter") and !@hasDecl(F, "openDrafter"));
    const Loaded = struct { backend: lanes.backend.Backend, facts: lanes.Model, stream_bytes: usize, draft_kind: ?[]const u8 = null };
    var loaded: Loaded = undefined;
    loaded.stream_bytes = 7;
    loaded.draft_kind = "DFlash2";
    const o: api.Open = .{ .dir = "glm", .model_type = F.model_type, .drafter = "cache/dflash2", .drafter_name = "example/DFlash2" };
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var problem: []const u8 = "";
    const served = (try wrap(F, arena.allocator(), std.testing.allocator, std.testing.io, loaded, o, &problem)) orelse return error.TestUnexpectedResult;
    try std.testing.expect(served.held == null and served.stream_bytes == 7);
    try std.testing.expectEqualStrings("DFlash2", served.source.?.kind);
    try std.testing.expectEqualStrings("example/DFlash2", served.source.?.name);
    const Plain = struct {
        pub const model_type = "plain";
        pub const Options = struct { context: usize };
    };
    try std.testing.expect((try wrap(Plain, arena.allocator(), std.testing.allocator, std.testing.io, loaded, o, &problem)) == null);
    try std.testing.expectEqualStrings("--drafter: plain checkpoints load no draft model on CUDA", problem);
}

test "--mtp-drafts 0 reaches drafter selection so an MTP refusal does not suppress other drafter kinds" {
    var called = false;
    const Opened = struct {
        drafter: lanes.drafter.Drafter,
        ctx: *anyopaque,
        close: *const fn (*anyopaque) void,
        reserve: *const fn (*anyopaque, u32) anyerror!void,
    };
    const F = struct {
        pub const model_type = "test";
        pub fn openDrafter(_: Allocator, _: std.Io, p: *anyopaque, _: []const u8, _: u8) error{MtpDepthZero}!Opened {
            @as(*bool, @ptrCast(@alignCast(p))).* = true;
            return error.MtpDepthZero;
        }
    };
    const Loaded = struct { backend: lanes.backend.Backend, facts: lanes.Model, stream_bytes: usize, ctx: *anyopaque };
    const loaded: Loaded = .{ .backend = undefined, .facts = undefined, .stream_bytes = 0, .ctx = &called };
    const o: api.Open = .{ .dir = "target", .model_type = F.model_type, .drafter = "target", .mtp_drafts = 0 };
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var problem: []const u8 = "";
    const served = (try wrap(F, arena.allocator(), std.testing.allocator, std.testing.io, loaded, o, &problem)) orelse return error.TestUnexpectedResult;
    try std.testing.expect(called);
    try std.testing.expect(served.held == null);
}

/// A drafter with facts and nothing else: wrap reads its facts and kind before any stream.
const FactsOnly = struct {
    fn taps(_: *anyopaque) []const u32 {
        return &.{};
    }
    fn facts(_: *anyopaque) lanes.drafter.Facts {
        return .{ .depth = 1, .step_ms = 1 };
    }
    fn close(_: *anyopaque) void {}
    fn reserve(_: *anyopaque, _: u32) anyerror!void {}
};

fn firstRowServed(comptime rows: ?bool, drafts: bool) !bool {
    var vt: lanes.drafter.Drafter.VTable = undefined;
    vt.taps = FactsOnly.taps;
    vt.facts = FactsOnly.facts;
    var drafter: lanes.drafter.Drafter = .{ .ptr = &vt, .vtable = &vt };
    const Opened = if (rows) |r| struct { drafter: lanes.drafter.Drafter, ctx: *anyopaque, close: *const fn (*anyopaque) void, reserve: *const fn (*anyopaque, u32) anyerror!void, first_rows: bool = r } else struct { drafter: lanes.drafter.Drafter, ctx: *anyopaque, close: *const fn (*anyopaque) void, reserve: *const fn (*anyopaque, u32) anyerror!void };
    const F = struct {
        pub const model_type = "test";
        pub fn openDrafter(_: Allocator, _: std.Io, p: *anyopaque, _: []const u8, _: u8) !Opened {
            return .{ .drafter = @as(*lanes.drafter.Drafter, @ptrCast(@alignCast(p))).*, .ctx = p, .close = FactsOnly.close, .reserve = FactsOnly.reserve };
        }
    };
    var target: lanes.fake.Fake = .{ .gpa = std.testing.allocator };
    defer target.deinit();
    const Loaded = struct { backend: lanes.backend.Backend, facts: lanes.Model, stream_bytes: usize, ctx: *anyopaque };
    const loaded: Loaded = .{ .backend = target.backend(), .facts = .{}, .stream_bytes = 0, .ctx = &drafter };
    const o: api.Open = .{ .dir = "target", .model_type = F.model_type, .drafter = "drafter", .drafts = drafts };
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var problem: []const u8 = "";
    var served = (try wrap(F, arena.allocator(), std.testing.allocator, std.testing.io, loaded, o, &problem)) orelse return error.TestUnexpectedResult;
    defer served.close(std.testing.allocator);
    try std.testing.expectEqual(drafts, served.held != null);
    return served.backend.vtable.first_row != null;
}

test "--drafter: a family whose Opened asks for first rows serves logprobs' first row drafted and plain" {
    try std.testing.expect(try firstRowServed(true, false)); // plain: the family's own table
    try std.testing.expect(try firstRowServed(true, true)); // drafted: the wrapper forwards first_row
    // a family that does not ask keeps main's wrapper table, without first_row
    try std.testing.expect(!try firstRowServed(null, true));
    try std.testing.expect(try firstRowServed(null, false));
}
