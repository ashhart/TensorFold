//! GLM's final norm and lm_head launch policy without a CUDA device.

const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const fw = @import("cuda_forward.zig");
const weights = @import("cuda_weights.zig");
const shard = @import("cuda_shard.zig");
const cfg = @import("config.zig");
const t = std.testing;

const Seen = struct { id: usize, args: [11]u64 };
const ids = struct {
    const norm = fieldId("rmsnorm_4096");
    const rows = fieldId("bmm_rows");
    const tile = fieldId("bmm_tile");
    const tile1 = fieldId("bmm_tile1");
    const decode = fieldId("bmm_decode");
    const d32s1 = fieldId("bmm_d32s1");
    const d32s2 = fieldId("bmm_d32s2");
    const d32s3 = fieldId("bmm_d32s3");
    const d64s2 = fieldId("bmm_d64s2");
};

fn fieldId(comptime name: []const u8) usize {
    inline for (@typeInfo(glue.Kernels).@"struct".field_names, 1..) |name_, i| {
        if (comptime std.mem.eql(u8, name_, name)) return i;
    }
    unreachable;
}

const Launch = struct {
    var n: usize = 0;
    var seen: [8]Seen = undefined;

    fn run(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        const id = @intFromPtr(f);
        var s: Seen = .{ .id = id, .args = @splat(0) };
        const sizes: []const u8 = switch (id) {
            ids.norm => "88888844",
            ids.rows, ids.tile, ids.tile1 => "888888844",
            ids.decode, ids.d32s1, ids.d32s2, ids.d32s3, ids.d64s2 => "8888888844",
            else => "",
        };
        for (sizes, 0..) |size, i| {
            const at = args.?[i].?;
            s.args[i] = if (size == '8') @as(*const u64, @ptrCast(@alignCast(at))).* else @as(*const u32, @ptrCast(@alignCast(at))).*;
        }
        seen[n] = s;
        n += 1;
        return 0;
    }
};

fn check(prefill: bool, head: bool, rows: usize) !void {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Launch.run;
    d.api.cuLaunchKernelEx = @import("cuda_pdl.zig").forwardEx(Launch.run);
    var kernels: glue.Kernels = undefined;
    inline for (@typeInfo(glue.Kernels).@"struct".field_names, 1..) |name, i| @field(kernels, name) = .{ .d = &d, .handle = @ptrFromInt(i) };
    Launch.n = 0;

    const c: cfg.Config = .{};
    const sh = try shard.Share.of(&c, 0, 1);
    var w: weights.Weights = .{ .gpa = t.allocator, .config = c, .share = sh, .norm = .{ .bf16 = 0x10_0000 }, .head = .{ .bf16 = 0x20_0000 } };
    var b: fw.Buffers = undefined;
    b.prefill = prefill;
    b.hidden = 0x100_0000;
    b.fnormed = 0x200_0000;
    b.fxs = 0x300_0000;
    b.logits = 0x400_0000;
    b.kda.mm_part = 0x500_0000;
    var st: fw.State = undefined;
    var mods: fw.Modules = undefined;
    mods.glue = kernels;
    mods.index = null;
    mods.tiled = false;
    mods.packs = null;
    mods.copy_rows = undefined;
    mods.dsel = false;
    mods.dscore = null;
    mods.mla = null;
    var one: [1]fw.Rank = .{.{ .w = &w, .st = &st, .b = &b }};
    const f: fw.Forward = .{ .d = &d, .m = &mods, .stream = .{ .d = &d, .handle = null }, .ranks = &one, .exchange = .local };
    try f.finish(rows, head);

    const norm = Launch.seen[0];
    try t.expectEqual(@as(usize, ids.norm), norm.id);
    try t.expectEqualSlices(u64, &.{ b.hidden, c.hidden, w.norm.bf16, b.fnormed, c.hidden, b.fxs }, norm.args[0..6]);
    if (!head and prefill) {
        try t.expectEqual(@as(usize, 1), Launch.n);
        return;
    }
    try t.expectEqual(@as(usize, 2), Launch.n);
    const lm = Launch.seen[1];
    try t.expect(lm.id != ids.norm);
    if (prefill) {
        const want_rows: u64 = 1;
        const last = b.fnormed + (rows - 1) * c.hidden * 2;
        try t.expectEqualSlices(u64, &.{ last, w.head.bf16, b.logits, b.logits, want_rows, c.hidden, sh.vocab.len(), c.hidden }, lm.args[0..8]);
    } else {
        try t.expectEqualSlices(u64, &.{ b.fnormed, w.head.bf16, b.logits, rows, c.hidden, sh.vocab.len(), c.hidden }, lm.args[0..7]);
    }
}

test "prefill exact: E4 non-final lm_head is unread, final and decode heads keep their rows" {
    try check(true, false, 128);
    try check(true, true, 128);
    try check(false, true, 128);
}

test "prefill exact: MTP last-row routing keeps default and host-paged decode whole" {
    var st: fw.State = undefined;
    var b: fw.Buffers = undefined;
    var w: weights.Weights = undefined;
    w.config = .{};
    var one: [1]fw.Rank = .{.{ .w = &w, .st = &st, .b = &b }};
    var f: fw.Forward = .{ .d = undefined, .m = undefined, .stream = undefined, .ranks = &one, .exchange = .local };
    for ([_]bool{ false, true }) |prefill| {
        b.prefill = prefill;
        for ([_]bool{ false, true }) |enabled| {
            f.mtp_last_row = enabled;
            for ([_]u64{ 0, 4096 }) |pks| {
                st.mtp_caches.pks = pks;
                const route = fw.mtpRoute(&f, 4, true);
                const last = prefill or (enabled and pks == 0);
                try t.expectEqual(last, route.last_row);
                try t.expectEqual(if (last) @as(usize, 1) else 4, route.rows);
                try t.expectEqual(if (last) 3 * w.config.hidden * 2 else 0, route.tail);
            }
        }
    }
    b.prefill = false;
    f.mtp_last_row = true;
    st.mtp_caches.pks = 0;
    try t.expect(!fw.mtpRoute(&f, 1, true).last_row);
    try t.expect(!fw.mtpRoute(&f, 4, false).last_row);
}
