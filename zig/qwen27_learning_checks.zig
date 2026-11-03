//! Supplemental local FP64 learning checks, with real source faults counted in the contract's control tally.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const source = @import("learning_sources");
const ref = @import("tests/qwen27/learning_reference.zig");
pub const Tally = contract.gates.Tally;
const Case = struct { symbol: []const u8, text: []const u8, from: []const u8, to: []const u8 };
fn case(which: ref.Which) Case {
    return switch (which) {
        .pocket => .{ .symbol = "tf_learn_pocket_back", .text = source.learn_rows, .from = "const float scale = j >= p.u.x ? p.f.x : 1.0f;", .to = "const float scale = 1.0f;" },
        .affine => .{ .symbol = "tf_learn_q4_back", .text = source.learn_linear, .from = "4 * (column & 7)", .to = "4 * ((column + 1) & 7)" },
        .prep => .{ .symbol = "tf_attn_prep_back", .text = source.attn_train, .from = "go[d] * c + go[d + rope_half] * s", .to = "go[d] * c - go[d + rope_half] * s" },
        .scan => .{ .symbol = "tf_gdn_scan_back", .text = source.gdn_train, .from = "const float dkv = -bb * dd;", .to = "const float dkv = bb * dd;" },
    };
}
fn edit(a: std.mem.Allocator, c: Case) ![]u8 {
    if (std.mem.count(u8, c.text, c.from) != 1) return error.LearningControlEditMissing;
    return std.mem.replaceOwned(u8, a, c.text, c.from, c.to);
}
fn execute(comptime rig_: type, a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, which: ref.Which, control: bool) !bool {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const mem = arena.allocator();
    const c = case(which);
    const text = if (control) try edit(mem, c) else c.text;
    const pipeline = try rig_.pipeline(device, text, c.symbol);
    defer pipeline.deinit();
    const f = try ref.make(mem, which);
    if (pipeline.simdWidth() != 32 or pipeline.maxThreads() < f.threads) return error.LearningLocalLaunchLimit;
    var rig: rig_.Rig = .{ .a = mem, .device = device, .queue = queue };
    defer rig.deinit();
    const buffers = try mem.alloc(rig_.Ref, f.buffers.len);
    for (buffers, f.buffers) |*slot, b| {
        slot.* = if (b.input) try rig.input(b.bytes) else if (b.want != null) try rig.output(b.bytes.len) else try rig.scratch(b.bytes.len);
    }
    const inputs = try mem.dupe(u8, try rig.inputBytes());
    var first: ?[]u8 = null;
    var matches = true;
    var worst: f64 = 0;
    for (0..5) |_| {
        for (buffers, f.buffers) |slot, b| if (!b.input) @memcpy(rig.bytes(slot, b.bytes.len), b.bytes);
        const pass = rig.begin();
        pass.e.setPipeline(pipeline);
        for (buffers, 0..) |slot, i| pass.e.setBuffer(slot.buf, slot.off, i);
        pass.e.setBytes(f.params, buffers.len);
        pass.e.dispatchGroups(mtl.Size.of(f.groups[0], f.groups[1], f.groups[2]), mtl.Size.of(f.threads, 1, 1));
        try rig.finish(pass);
        for (buffers, f.buffers) |slot, b| if (b.want) |want| {
            const got = std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(rig.bytes(slot, b.bytes.len))));
            for (got, want) |x, y| {
                const distance = @abs(x - y);
                worst = @max(worst, distance);
                if (!std.math.isFinite(x) or distance > f.bound) matches = false;
            }
        };
        const out = try rig.outputBytes();
        if (first) |bytes| {
            if (!std.mem.eql(u8, bytes, out)) return error.LearningLocalRepeat;
        } else first = try mem.dupe(u8, out);
        if (!try rig.intact() or rig.overlap() or !std.mem.eql(u8, inputs, try rig.inputBytes())) return error.LearningLocalOwnership;
    }
    std.debug.print("Q27_LOCAL {s} control={} match={} worst_abs={e:.9} bound={e:.9} repeats=5 guards=true inputs_unchanged=true\n", .{ c.symbol, control, matches, worst, f.bound });
    return matches;
}

/// This supplements gates 1-5; passing four local finite fixtures is not a whole-model gradient qualification.
pub fn run(comptime rig_: type, a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, only: ?[]const u8, tally: *Tally) !void {
    for (std.enums.values(ref.Which)) |which| {
        const name = try std.fmt.allocPrint(a, "qwen27 learning local {s}", .{case(which).symbol});
        defer a.free(name);
        if (only) |filter| if (std.mem.indexOf(u8, name, filter) == null) continue;
        if (!try execute(rig_, a, device, queue, which, false)) {
            tally.failed += 1;
            std.debug.print("FAIL local shapes {s}\n", .{name});
        }
        tally.controls += 1;
        // Compile and launch errors are errors, never evidence that a numerical control was caught.
        if (!try execute(rig_, a, device, queue, which, true)) {
            tally.caught += 1;
            std.debug.print("CONTROL {s}: red at local shapes, as planted\n", .{name});
        } else {
            tally.missed += 1;
            std.debug.print("CONTROL MISSED {s}\n", .{name});
        }
    }
}

test "Qwen27 local adjoints: every real source control has exactly one witness" {
    for (std.enums.values(ref.Which)) |which| {
        const c = case(which);
        const changed = try edit(std.testing.allocator, c);
        defer std.testing.allocator.free(changed);
        try std.testing.expect(!std.mem.eql(u8, c.text, changed));
        try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, changed, c.from));
    }
}
test {
    _ = ref;
}
