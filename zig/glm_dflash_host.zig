//! DFlash allocation boundaries and the production loader's partial owner, without a GPU.
const std = @import("std");
const glm = @import("glm");
const registry = @import("core").contract.registry;
const cuda = @import("cuda");
const t = std.testing;

test "glm dflash: gather and read fit for every supported rank count" {
    for (1..17) |world| for ([_]usize{ 1, 8, glm.dflash.MAX_M }) |rows| {
        const n = glm.dflash_candidates.Layout.init(world, rows, glm.dflash.CW);
        const gathered = world * rows * glm.dflash.CW;
        try t.expect(gathered <= n.device);
        try t.expect(n.read <= n.host);
        try t.expectEqual(gathered, n.read);
    };
}

test "glm dflash: the raw scratch holds every 4-bit linear's words, scales and biases at every rank count" {
    const df = glm.dflash;
    var short: usize = 0;
    for (1..17) |world| for ([_]bool{ false, true }) |wide| {
        const ff = df.mlpSplit(world, wide).ff;
        // dflash_load's 4-bit linears, rows by columns: qkv, kv, o, gate and up, the kernel projections, fc
        const linears = [_][2]usize{ .{ df.QKV, df.D }, .{ 2 * df.KVW, df.D }, .{ df.D, df.D }, .{ 2 * ff, df.D }, .{ df.KVW, df.D }, .{ df.D, 5 * df.D } };
        for (linears) |shape| {
            const nk = shape[0] * shape[1];
            const need = std.mem.alignForward(usize, nk / 2, 256) + std.mem.alignForward(usize, nk / 32, 256) + nk / 32;
            if (need <= df.rawBytes(4, world, wide)) continue;
            std.debug.print("world {d}: a {d}x{d} linear needs {d} raw bytes, the scratch has {d}\n", .{ world, shape[0], shape[1], need, df.rawBytes(4, world, wide) });
            short += 1;
        }
    };
    try t.expectEqual(@as(usize, 0), short);
}

test "glm dflash: the MLP splits at two ranks as on main; three or more only with TF_ROCE=1" {
    const df = glm.dflash;
    const whole: df.MlpSplit = .{ .split = false, .ff = df.FF };
    try t.expectEqual(whole, df.mlpSplit(1, true));
    try t.expectEqual(df.MlpSplit{ .split = true, .ff = df.FF / 2 }, df.mlpSplit(2, false)); // 6144: main's split, no opt-in
    try t.expectEqual(df.MlpSplit{ .split = true, .ff = df.FF / 2 }, df.mlpSplit(2, true));
    for ([_]usize{ 3, 4, 6, 8 }) |world| try t.expectEqual(whole, df.mlpSplit(world, false)); // main's whole MLP on every rank
    try t.expectEqual(df.MlpSplit{ .split = true, .ff = df.FF / 4 }, df.mlpSplit(4, true)); // 3072 on four Sparks
    try t.expectEqual(df.MlpSplit{ .split = true, .ff = 4096 }, df.mlpSplit(3, true));
    try t.expectEqual(whole, df.mlpSplit(5, true)); // 12288 / 5 is no share
    try t.expectEqual(whole, df.mlpSplit(7, true));
    for (1..17) |world| for ([_]bool{ false, true }) |wide| {
        const m = df.mlpSplit(world, wide);
        try t.expect(m.ff % 256 == 0); // the q4 pack's rows, glm_mm's K tiles and df_swiglu's columns all fit
        if (m.split) {
            try t.expectEqual(df.FF, m.ff * world); // the shares are the whole width
            for (0..world) |rank| try t.expect(rank * m.ff + m.ff <= df.FF); // every rank's slice lies inside
        } else try t.expectEqual(df.FF, m.ff);
    };
}

const Resources = struct {
    pred: []u16 = &.{},
    succ: []u16 = &.{},
    scratch: []u8 = &.{},
    h_cpos: []i64 = &.{},
    h_crings: []u64 = &.{},
    cand: []i32 = &.{},
    gh: []f32 = &.{},
};
const Ops = struct {
    g: std.mem.Allocator,
    fail_read: ?usize = null,
    pub fn readBook(o: *Ops, i: usize, dst: []u16) !void {
        if (o.fail_read == i) return error.BadBook;
        @memset(dst, 7);
    }
    pub fn scratch(o: *Ops, r: *Resources) !void {
        r.scratch = try o.g.alloc(u8, 64);
    }
    pub fn freeScratch(o: *Ops, r: *Resources) void {
        o.g.free(r.scratch);
    }
};
fn release(g: std.mem.Allocator, r: Resources) void {
    g.free(r.pred);
    g.free(r.succ);
    g.free(r.scratch);
    g.free(r.h_cpos);
    g.free(r.h_crings);
    g.free(r.cand);
    g.free(r.gh);
}
fn attempt(fail_index: usize, fail_read: ?usize) !void {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var f = std.testing.FailingAllocator.init(arena.allocator(), .{ .fail_index = fail_index });
    const g = f.allocator();
    var r: Resources = .{};
    var o: Ops = .{ .g = g, .fail_read = fail_read };
    if (glm.dflash_load_owner.acquire(g, &r, &o, Ops, .{ .book = 16, .ctx = 8, .cand = 32, .hidden = 16 })) |_| {
        try t.expect(fail_index >= 7 and fail_read == null);
        try t.expect(f.allocated_bytes > f.freed_bytes);
        release(g, r);
    } else |err| {
        try t.expect(err == error.OutOfMemory or err == error.BadBook);
    }
    try t.expectEqual(f.allocated_bytes, f.freed_bytes);
}

test "glm dflash: every allocation failure releases earlier tail resources" {
    for (0..8) |i| try attempt(i, null);
}

test "glm dflash: either book read failure releases its allocated storage" {
    try attempt(99, 0);
    try attempt(99, 1);
}

test "glm dflash: every loaded symbol has a measured launch record" {
    inline for (@typeInfo(glm.dflash.Kernels).@"struct".field_names) |name| {
        const symbol = "df_" ++ name;
        var found = false;
        for (registry.all_cuda) |r| if (std.mem.eql(u8, symbol, r.launch.symbol)) {
            found = true;
            try t.expectEqualStrings("glm_dflash", r.launch.image);
            for (r.usage) |u| try t.expect(u.regs > 0);
        };
        try t.expect(found);
    }
}

const Readback = struct {
    syncs: usize,
    copies: usize,
    cand: [8]i32,
    gh: [4]f32,
};
const Recording = struct {
    var bytes: [256]u8 = undefined;
    var syncs: usize = 0;
    var copies: usize = 0;
    fn wait(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        syncs += 1;
        return cuda.abi.success;
    }
    fn get(dst: ?*anyopaque, src: u64, n: usize) callconv(.c) cuda.abi.Result {
        const out: [*]u8 = @ptrCast(dst.?);
        const in: [*]const u8 = @ptrFromInt(src);
        @memcpy(out[0..n], in[0..n]);
        copies += 1;
        return cuda.abi.success;
    }
    fn device(out: *cuda.abi.Device) callconv(.c) cuda.abi.Result {
        out.* = 0;
        return cuda.abi.success;
    }
    fn attribute(out: *c_int, _: cuda.abi.DeviceAttribute, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
        out.* = 0; // discrete: the actual wrapper takes its direct device-to-host branch
        return cuda.abi.success;
    }
};
fn recordReadback(one_sync: bool) !Readback {
    Recording.syncs = 0;
    Recording.copies = 0;
    for (&Recording.bytes, 0..) |*b, i| b.* = @truncate(i);
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuMemcpyDtoH_v2 = Recording.get;
    d.api.cuStreamSynchronize = Recording.wait;
    d.api.cuCtxGetDevice = Recording.device;
    d.api.cuDeviceGetAttribute = Recording.attribute;
    d.func_name = null;
    cuda.download_stage.forget();
    defer cuda.download_stage.forget();
    var dr: glm.dflash.Drafter = undefined;
    const e = struct { d: *const cuda.Driver, stream: cuda.Stream, group: ?*cuda.tp_group.Group = null }{ .d = &d, .stream = .{ .d = &d, .handle = null } };
    var out: Readback = undefined;
    try glm.dflash_block.downloadResults(&dr, &e, std.mem.sliceAsBytes(&out.cand), @intFromPtr(&Recording.bytes), std.mem.sliceAsBytes(&out.gh), @intFromPtr(&Recording.bytes) + 64, one_sync);
    out.syncs = Recording.syncs;
    out.copies = Recording.copies;
    return out;
}

test "glm dflash: one rank-0 result readback sync preserves every host byte" {
    const before = try recordReadback(false);
    const after = try recordReadback(true);
    try t.expectEqual(@as(usize, 2), before.syncs);
    try t.expectEqual(@as(usize, 1), after.syncs);
    try t.expectEqual(@as(usize, 2), before.copies);
    try t.expectEqual(@as(usize, 2), after.copies);
    try t.expectEqualSlices(u8, std.mem.sliceAsBytes(&before.cand), std.mem.sliceAsBytes(&after.cand));
    try t.expectEqualSlices(u8, std.mem.sliceAsBytes(&before.gh), std.mem.sliceAsBytes(&after.gh));
    try t.expectEqualSlices(u8, Recording.bytes[0..32], std.mem.sliceAsBytes(&after.cand));
    try t.expectEqualSlices(u8, Recording.bytes[64..80], std.mem.sliceAsBytes(&after.gh));
}

test "glm dflash: one-sync is strict 0 or 1 and every rank agrees" {
    const Reader = struct {
        var value: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_DFLASH2_ONE_SYNC")) value else null;
        }
    };
    var problem: []const u8 = "";
    for ([_]?[]const u8{ null, "", "0" }) |value| {
        Reader.value = value;
        try t.expect(!(try glm.ranks.Settings.read(Reader.get, &problem)).dflash2_one_sync);
    }
    Reader.value = "1";
    const on = try glm.ranks.Settings.read(Reader.get, &problem);
    try t.expect(on.dflash2_one_sync);
    Reader.value = "2";
    try t.expectError(error.BadSetting, glm.ranks.Settings.read(Reader.get, &problem));
    try t.expectEqualStrings("TF_GLM_DFLASH2_ONE_SYNC: 0 or 1", problem);
    const different = [_]glm.ranks.Fingerprint{
        glm.ranks.Fingerprint.of(2, 0, on, 8192, true, @splat(0), @splat(0), @splat(0)),
        glm.ranks.Fingerprint.of(2, 1, .{}, 8192, true, @splat(0), @splat(0), @splat(0)),
    };
    try t.expectEqual(glm.ranks.Mismatch{ .field = .dflash2_one_sync, .rank = 1 }, glm.ranks.mismatch(&different).?);
    const same = [_]glm.ranks.Fingerprint{
        glm.ranks.Fingerprint.of(2, 0, on, 8192, true, @splat(0), @splat(0), @splat(0)),
        glm.ranks.Fingerprint.of(2, 1, on, 8192, true, @splat(0), @splat(0), @splat(0)),
    };
    try t.expect(glm.ranks.mismatch(&same) == null);
}
