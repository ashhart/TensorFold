//! --drafter on the CUDA host: a family's draft model behind the lane core's Drafted wrapper over its backend.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const Allocator = std.mem.Allocator;

/// What the host serves: the family's backend as is, or wrapped with a drafter in front.
pub const Served = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    stream_bytes: usize,
    held: ?*Held = null,
    /// The family's own per-stream buffers, when its Loaded pre-allocates them.
    family: ?struct { ctx: *anyopaque, reserve: *const fn (*anyopaque, u32) anyerror!void } = null,

    /// The family's and the drafter's per-stream buffers for `streams`, once the server knows the count.
    pub fn reserve(s: *Served, streams: u32) !void {
        if (s.family) |f| try f.reserve(f.ctx, streams);
        if (s.held) |h| try h.reserve(h.ctx, streams);
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

const Held = struct { drafted: lanes.drafted.Drafted, ctx: *anyopaque, close: *const fn (*anyopaque) void, reserve: *const fn (*anyopaque, u32) anyerror!void };

/// `loaded`'s backend, wrapped with the drafter `o.drafter` names when it does; null with `problem` set on refusal.
pub fn wrap(comptime F: type, a: Allocator, gpa: Allocator, io: std.Io, loaded: anytype, o: api.Open, problem: *[]const u8) !?Served {
    var s: Served = .{ .backend = loaded.backend, .facts = loaded.facts, .stream_bytes = loaded.stream_bytes };
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
        problem.* = try std.fmt.allocPrint(a, "the native CUDA engine cannot load the drafter {s} ({s})", .{ dir, @errorName(e) });
        return null;
    };
    const h = try gpa.create(Held);
    errdefer gpa.destroy(h);
    h.* = .{ .drafted = undefined, .ctx = opened.ctx, .close = opened.close, .reserve = opened.reserve };
    h.drafted = lanes.drafted.Drafted.init(gpa, loaded.backend, opened.drafter) catch |e| {
        opened.close(opened.ctx);
        problem.* = try std.fmt.allocPrint(a, "the drafter {s} cannot attach to this model ({s})", .{ dir, @errorName(e) });
        return null;
    };
    s.backend = h.drafted.backend();
    s.facts = h.drafted.facts(loaded.facts);
    s.stream_bytes = h.drafted.streamBytes(loaded.stream_bytes);
    s.held = h;
    return s;
}

test "--drafter: a family that loads its own drafter (GLM's DFlash2) serves as it loaded, and one with neither refuses" {
    const F = @import("glm_cuda").native;
    try std.testing.expect(@hasField(F.Options, "drafter") and !@hasDecl(F, "openDrafter"));
    const Loaded = struct { backend: lanes.backend.Backend, facts: lanes.Model, stream_bytes: usize };
    var loaded: Loaded = undefined;
    loaded.stream_bytes = 7;
    const o: api.Open = .{ .dir = "glm", .model_type = F.model_type, .drafter = "dflash2" };
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var problem: []const u8 = "";
    const served = (try wrap(F, arena.allocator(), std.testing.allocator, std.testing.io, loaded, o, &problem)) orelse return error.TestUnexpectedResult;
    try std.testing.expect(served.held == null and served.stream_bytes == 7);
    const Plain = struct {
        pub const model_type = "plain";
        pub const Options = struct { context: usize };
    };
    try std.testing.expect((try wrap(Plain, arena.allocator(), std.testing.allocator, std.testing.io, loaded, o, &problem)) == null);
    try std.testing.expectEqualStrings("--drafter: plain checkpoints load no draft model on CUDA", problem);
}
