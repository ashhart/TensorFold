//! Per-family Metal learner registration, admission and adapter lifetime.
const std = @import("std");
const api = @import("engine_api");
const living = @import("living");
const adapter = @import("slide_adapter.zig");
const Allocator = std.mem.Allocator;

/// A learner supplies all three declarations, just as the CUDA family registration does.
pub fn supported(comptime F: type) bool {
    if (!@hasDecl(F, "slide")) return false;
    if (!@hasDecl(F, "learnModel") or !@hasDecl(F, "linked")) @compileError("a Metal slide family needs learnModel and linked");
    return true;
}

pub fn Registry(comptime modules: anytype) type {
    return struct {
        pub fn supports(model_type: []const u8) bool {
            inline for (modules) |F| if (std.mem.eql(u8, model_type, F.model_type)) return supported(F);
            return false;
        }
    };
}

/// Family-neutral ownership: the lane host stops before this adapter and the family's model are freed.
pub const Registration = struct {
    ctx: *anyopaque,
    hook: api.Learner,
    conversations: bool,
    free: *const fn (*anyopaque, Allocator) void,

    pub fn deinit(r: Registration, a: Allocator) void {
        r.free(r.ctx, a);
    }
};

pub fn init(comptime F: type, a: Allocator, io: std.Io, model: anytype, dir: []const u8) !Registration {
    if (comptime !supported(F)) @compileError("cannot attach a Metal family without a learner");
    const Slide = adapter.Adapter(F.slide);
    const s = try a.create(Slide);
    s.* = Slide.init(a, io, F.learnModel(model, dir));
    return .{ .ctx = s, .hook = s.hook(), .conversations = s.learner.readsConversations(), .free = struct {
        fn drop(ptr: *anyopaque, gpa: Allocator) void {
            const x: *Slide = @ptrCast(@alignCast(ptr));
            x.deinit();
            gpa.destroy(x);
        }
    }.drop };
}

pub fn refusal(comptime F: type, a: Allocator, io: std.Io, o: api.Open) !?[]const u8 {
    if (!o.slide) return null;
    if (comptime !supported(F)) return o.slideRefusal(a, "Macs");
    if (try adapter.refusal(F, a, io, o.dir, "`cp -cRL` (an APFS clone that follows links)")) |why| return why;
    const f = living.families.byModelType(F.model_type) orelse return error.NoLearningFamily;
    return slideWidths(a, io, o.dir, f);
}

/// Writable bits and groups and the matrix names come from Living Weights' family descriptor.
pub fn slideWidths(a: Allocator, io: std.Io, dir: []const u8, f: living.Family) !?[]const u8 {
    const path = try std.fs.path.join(a, &.{ dir, "config.json" });
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20)) catch return null;
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, text, .{}) catch return null;
    if (doc != .object) return null;
    const mismatch = living.widths.other(a, f, doc.object) catch |e| switch (e) {
        error.NoWidthDescription => return e,
        else => return null,
    };
    const m = mismatch orelse return null;
    return try std.fmt.allocPrint(a, "--slide learns into checkpoints at {d}-bit in groups of {d}, and {s} in {s} is {d}-bit in groups of {d}: serve it without --slide", .{ f.quant.bits, f.quant.group, m.name, dir, m.quant.bits, m.quant.group });
}

test "Metal learner: a family's declarations register it, without a model-name allowlist" {
    const Missing = struct {
        pub const model_type = "missing";
    };
    const Learns = struct {
        pub const model_type = "future";
        pub const slide = struct {};
        pub const learnModel = void;
        pub const linked = void;
    };
    const R = Registry(.{ Missing, Learns });
    try std.testing.expect(!R.supports("missing"));
    try std.testing.expect(!R.supports("unknown"));
    try std.testing.expect(R.supports("future"));
}
