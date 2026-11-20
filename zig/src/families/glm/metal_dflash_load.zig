//! GLM-5.3-Flash's DFlash2 drafter on Metal (--drafter): its refusals, and the drafter loaded beside the engine.
const std = @import("std");
const Engine = @import("engine.zig").Engine;
const model_mod = @import("metal_dflash_model.zig");
const Taps = @import("metal_dflash_taps.zig").Taps;
const Target = @import("metal_dflash_backend.zig").Target;
const rounds = @import("metal_dflash_rounds.zig");
const st = @import("state.zig");
const vision = @import("vision.zig");
const mtl = @import("metal");

/// The flags a GLM drafter is refused beside, before any weight loads (null: none of them).
pub const Flags = struct { speed_up: bool, drafter_bits: u8 };

/// Why the drafter cannot open with `f`, or null when it can.
pub fn refusal(f: Flags) ?[]const u8 {
    if (f.speed_up) return "GLM-5.3-Flash's DFlash2 drafter runs on one Mac: serve the --speed-up pair without --drafter";
    if (f.drafter_bits != 0 and f.drafter_bits != 4) return "--drafter-bits takes 0 (GLM-5.3-Flash's DFlash2 drafter in bf16, as published) or 4 (quantized at load)";
    return null;
}

/// The drafter, its tap capture and the target it borrows, freed with the engine before what it borrows.
pub const Drafter = struct {
    gpa: std.mem.Allocator,
    model: *model_mod.Model,
    taps: *Taps,
    target: *Target,

    pub fn deinit(d: *Drafter) void {
        const gpa = d.gpa;
        d.taps.deinit(gpa);
        gpa.destroy(d.taps);
        d.model.deinit();
        gpa.destroy(d.target);
        gpa.destroy(d);
    }
};

/// Device bytes the drafter holds beside the weights (none without one).
pub fn heldBytes(e: *const Engine) u64 {
    const d = e.dflash orelse return 0;
    return d.model.bytes + d.taps.stage.length();
}

/// Device bytes a stream's own drafter context takes (none without a drafter).
pub fn streamBytes(e: *const Engine) u64 {
    const d = e.dflash orelse return 0;
    return rounds.contextBytes(d.model.graph.config);
}

/// The DFlash2 drafter of `dir` over the loaded engine, its bytes checked against the load limit beside the weights.
pub fn load(gpa: std.mem.Allocator, io: std.Io, e: *Engine, dir: []const u8, bits: u8) !void {
    const target = try gpa.create(Target);
    errdefer gpa.destroy(target);
    target.* = .{ .device = e.device, .queue = e.queue, .k = e.k, .w = e.w, .c = e.c };
    const m = model_mod.Model.load(gpa, io, target, dir, if (bits == 4) .q4 else .bf16) catch |err| {
        std.log.err("glm drafter: the --drafter checkpoint failed to load ({s})", .{@errorName(err)});
        return err;
    };
    errdefer m.deinit();
    const taps = try gpa.create(Taps);
    errdefer gpa.destroy(taps);
    taps.* = try Taps.init(e.device, m.pending, m.graph.config.window, m.graph.config.hidden, gpa, st.max_rows);
    errdefer taps.deinit(gpa);
    const drafter_bytes = m.bytes + taps.stage.length();
    const used = e.w.bytes + e.arena.bytes + drafter_bytes + vision.heldBytes(e.vision);
    if (used > e.limit.bytes) {
        const gb = 1e9;
        std.log.err("glm drafter: the --drafter checkpoint's {d:.1} GB pass the {d:.1} GB load limit beside the weights (lower --context, or unset --drafter)", .{ @as(f64, @floatFromInt(drafter_bytes)) / gb, @as(f64, @floatFromInt(e.limit.bytes)) / gb });
        return error.OverMemoryLimit;
    }
    const d = try gpa.create(Drafter);
    d.* = .{ .gpa = gpa, .model = m, .taps = taps, .target = target };
    e.dflash = d;
}

/// Add all loaded drafter weights, caches and scratch buffers to the target's residency set.
pub fn joinResidency(gpa: std.mem.Allocator, e: *Engine) !void {
    const d = e.dflash orelse return;
    const r = &(e.residency orelse return);
    var list: std.ArrayList(mtl.Buffer) = .empty;
    defer list.deinit(gpa);
    for (d.model.checkpoint.shards.items) |s| try list.append(gpa, s.buffer);
    if (d.model.backend.copies) |*c| try c.appendBuffers(&list);
    for (d.model.backend.caches) |c| try list.appendSlice(gpa, &.{ c.keys, c.values });
    for (d.model.backend.frame.pool.items) |b| try list.append(gpa, b);
    try list.appendSlice(gpa, &.{ d.model.pending, d.taps.stage });
    @import("residency.zig").joinDrafter(e, list.items);
    d.model.backend.frame.setResident(r.follow());
}

test "heldBytes and load accounting include the tap-stage buffer beside drafter model bytes" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const ring = try device.buffer(64 * 5 * 512 * 2, mtl.ResourceOptions.shared);
    defer ring.deinit();
    var taps = try Taps.init(device, ring, 64, 512, std.testing.allocator, 1);
    defer taps.deinit(std.testing.allocator);
    var m: model_mod.Model = undefined;
    m.bytes = 1024;
    var d: Drafter = .{ .gpa = std.testing.allocator, .model = &m, .taps = &taps, .target = undefined };
    var e: Engine = undefined;
    e.dflash = &d;
    try std.testing.expectEqual(@as(u64, 1024 + taps.stage.length()), heldBytes(&e));
}

test "the drafter is refused beside the pair and with bits other than 0 or 4, and opens without them" {
    try std.testing.expect(refusal(.{ .speed_up = false, .drafter_bits = 0 }) == null);
    try std.testing.expect(std.mem.indexOf(u8, refusal(.{ .speed_up = true, .drafter_bits = 0 }).?, "--speed-up") != null);
    try std.testing.expect(refusal(.{ .speed_up = false, .drafter_bits = 4 }) == null);
    try std.testing.expect(std.mem.indexOf(u8, refusal(.{ .speed_up = false, .drafter_bits = 8 }).?, "--drafter-bits") != null);
}
