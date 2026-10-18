//! Host tests of the served knobs' readers, and the fused dense prefill (TF_DSV41_PF_DENSE=fused) on the real emitter.
const std = @import("std");
const testing = std.testing;
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const bp = @import("block_prefill.zig");
const check = @import("calls_check.zig");
const pk = @import("serve_knobs.zig");
const knob_types = @import("knob_types.zig");
const Config = @import("config.zig").Config;

// CED replay's programs on the same emitter (TF_DSV41_PREFILL=replay)
test {
    _ = @import("ced_test.zig");
    _ = @import("multi_test.zig");
    _ = @import("pf4k_test.zig");
    _ = @import("pfovl_test.zig");
    _ = @import("pftbo_test.zig");
    _ = @import("streamrb_test.zig");
}

const Fake = struct {
    var pairs: []const [2][]const u8 = &.{};
    fn get(name: []const u8) ?[]const u8 {
        for (pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
        return null;
    }
};

test "router grouping/rotation fusion is explicitly opt-in" {
    Fake.pairs = &.{};
    try testing.expect(!pk.routerGroupRot(&Fake.get));
    for ([_][]const u8{ "0", "false", "yes", "2", "" }) |value| {
        Fake.pairs = &.{.{ "TF_DSV41_ROUTER_GROUP_ROT", value }};
        try testing.expect(!pk.routerGroupRot(&Fake.get));
    }
    Fake.pairs = &.{.{ "TF_DSV41_ROUTER_GROUP_ROT", " 1 " }};
    try testing.expect(pk.routerGroupRot(&Fake.get));
    Fake.pairs = &.{};
}

test "pfdense.pick: the override, the table (its small config at few rows), the heuristic; stages must divide K" {
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const entries = try knob_types.PfDense.parseTable(arena.allocator(),
        \\{"4096,1024": {"cfg": 1, "group": 4, "small": 3, "small_rows": 512}, "1024, 7168": {"cfg": 7}}
    );
    const p: knob_types.PfDense = .{ .entries = entries };
    try testing.expectEqual([2]u32{ 1, 4 }, p.pick(4096, 1024, 10, 2048));
    try testing.expectEqual([2]u32{ 3, 4 }, p.pick(4096, 1024, 10, 512));
    try testing.expectEqual([2]u32{ 7, 8 }, p.pick(1024, 7168, 8, 2048));
    // not in the table: pfdense.heuristic on 48 SMs (BM 128 at 512 CTAs; BM 64 when 128's last wave is a third empty)
    try testing.expectEqual([2]u32{ 0, 8 }, p.pick(4096, 4096, 10, 2048));
    try testing.expectEqual([2]u32{ 3, 8 }, (knob_types.PfDense{}).pick(4096, 1024, 10, 2048));
    const o: knob_types.PfDense = .{ .entries = entries, .override = .{ 5, 2 } };
    try testing.expectEqual([2]u32{ 5, 2 }, o.pick(4096, 1024, 10, 2048));
    try testing.expectError(error.BadTable, knob_types.PfDense.parseTable(arena.allocator(), "{\"4096\": {\"cfg\": 1}}"));
    try testing.expectError(error.BadTable, knob_types.PfDense.parseTable(arena.allocator(), "{\"4096,128\": {\"group\": 1}}"));
    try testing.expect(knob_types.PfDense.eligible(4096, 1024, 16, false));
    try testing.expect(!knob_types.PfDense.eligible(4096, 1024, 16, true)); // dense3 never repacks 16-bit words
    try testing.expect(!knob_types.PfDense.eligible(4096, 1024, 6, false));
    try testing.expect(!knob_types.PfDense.eligible(4160, 1024, 8, false));
}

test "serve knob readers: PF_DENSE modes, prefill rows, prefetch ahead, prefill mode" {
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const aa = arena.allocator();
    Fake.pairs = &.{};
    try testing.expect(try pk.pfDense(aa, testing.io, &Fake.get) == null);
    try testing.expectEqual(@as(?u32, null), try pk.prefillRows(&Fake.get));
    try testing.expect(!pk.prefetchAhead(&Fake.get));
    try testing.expectEqual(pk.PrefillMode.full, try pk.prefillMode(&Fake.get));
    Fake.pairs = &.{ .{ "TF_DSV41_PF_DENSE", "off" }, .{ "TF_DSV41_PREFILL_CHUNK", "2048" }, .{ "TF_DSV41_PREFILL_ROWS", "1024" }, .{ "TF_DSV41_PREFETCH_AHEAD", "1" }, .{ "TF_DSV41_PREFILL", "replay" } };
    try testing.expect(try pk.pfDense(aa, testing.io, &Fake.get) == null);
    try testing.expectEqual(@as(?u32, 1024), try pk.prefillRows(&Fake.get));
    try testing.expect(pk.prefetchAhead(&Fake.get));
    try testing.expectEqual(pk.PrefillMode.replay, try pk.prefillMode(&Fake.get));
    Fake.pairs = &.{ .{ "TF_DSV41_PF_DENSE", "fused" }, .{ "TF_DSV41_PF_DENSE_CFG", "3,16" } };
    const p = (try pk.pfDense(aa, testing.io, &Fake.get)).?;
    try testing.expectEqual(@as(?[2]u32, .{ 3, 16 }), p.override);
    Fake.pairs = &.{.{ "TF_DSV41_PF_DENSE", "tiles" }};
    try testing.expectError(error.BadKnob, pk.pfDense(aa, testing.io, &Fake.get));
    Fake.pairs = &.{.{ "TF_DSV41_PREFILL_ROWS", "1000" }};
    try testing.expectError(error.BadKnob, pk.prefillRows(&Fake.get));
    // the served table file through the reader
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "pfdense-table.json", .data = "{\"4096,1024\": {\"cfg\": 2, \"group\": 8}}" });
    const path = try tmp.dir.realPathFileAlloc(testing.io, "pfdense-table.json", aa);
    Fake.pairs = &.{ .{ "TF_DSV41_PF_DENSE", "fused" }, .{ "TF_DSV41_PF_DENSE_TABLE", path } };
    const t = (try pk.pfDense(aa, testing.io, &Fake.get)).?;
    try testing.expectEqual([2]u32{ 2, 8 }, t.pick(4096, 1024, 10, 2048));
    Fake.pairs = &.{};
}

test "TF_DSV41_PROMPT_TAIL: Python's server splits a prompt at its last token, Forward.prompt does not" {
    Fake.pairs = &.{};
    defer Fake.pairs = &.{};
    try testing.expectEqual(pk.PromptTail.verify, try pk.promptTail(&Fake.get));
    Fake.pairs = &.{.{ "TF_DSV41_PROMPT_TAIL", "prefill" }};
    try testing.expectEqual(pk.PromptTail.prefill, try pk.promptTail(&Fake.get));
    Fake.pairs = &.{.{ "TF_DSV41_PROMPT_TAIL", "decode" }};
    try testing.expectError(error.BadKnob, pk.promptTail(&Fake.get));
    // pieces prefill [0, n - 1), the finals make prompt[-1] the pending row (a 1-token prompt: no piece)
    for ([_]usize{ 1, 2, 45, 2048, 2049, 4097 }) |n| {
        const v = pk.promptSplit(n, .verify);
        try testing.expectEqual(n - 1, v.prefill);
        try testing.expectEqual(@as(usize, 1), v.window);
        const p = pk.promptSplit(n, .prefill);
        try testing.expectEqual(n, p.prefill);
        try testing.expectEqual(@as(usize, 0), p.window);
    }
}

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/expert-k2.txt"));
    // x3gm's K2s present (gate_forward.widths reads them from the pack): two widths a layer
    for (0..40) |L| try w.gm.put(a, @intCast(L), .{ (1 << 3) | (1 << 4), (1 << 3) | (1 << 4) });
    return w;
}

fn isDense(c: calls.Call) bool {
    const n = c.name;
    return std.mem.eql(u8, n, "_gemm") or std.mem.eql(u8, n, "tensorfold_exl3_linear_v4.rot_in") or
        std.mem.eql(u8, n, "tensorfold_exl3_linear_v4.unpack") or std.mem.eql(u8, n, "tf_dsv41_dense3_v1.unpack") or
        std.mem.eql(u8, n, "tf_dsv41_pfdense_v1.gemm");
}

fn tensor(x: calls.Arg) calls.Tensor {
    return x.t;
}

fn roleName(t: calls.Tensor) []const u8 {
    return switch (t.role) {
        .buf => |b| b,
        .weight => |w| w,
        .empty => "",
    };
}

test "TF_DSV41_PF_DENSE=fused on the prefill emitter: one fused GEMM a projection, the same reads and writes" {
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    var w = try widths(aa);
    const layers = [_]u32{ 1, 2, 3 };
    const pfd: knob_types.PfDense = .{};
    for ([_]i64{ 2048, 300, 12 }) |n| {
        const off = try bp.emitPrefill(aa, &cfg, &w, .{}, &layers, n, 0, true);
        const on = try bp.emitPrefill(aa, &cfg, &w, .{ .pfd = &pfd }, &layers, n, 0, true);
        // every call that is not a dense projection's: the same, in the same order
        var i: usize = 0;
        var j: usize = 0;
        var gemms: usize = 0;
        var fused: usize = 0;
        var left: usize = 0; // ineligible widths keep the unpack + `_gemm`
        while (true) {
            while (i < off.len and isDense(off[i])) : (i += 1) {
                if (std.mem.eql(u8, off[i].name, "_gemm")) gemms += 1;
            }
            while (j < on.len and isDense(on[j])) : (j += 1) {
                if (std.mem.eql(u8, on[j].name, "tf_dsv41_pfdense_v1.gemm")) fused += 1;
                if (std.mem.eql(u8, on[j].name, "_gemm")) left += 1;
            }
            if (i == off.len or j == on.len) break;
            try testing.expectEqualStrings(off[i].name, on[j].name);
            try testing.expectEqual(off[i].args.len, on[j].args.len);
            i += 1;
            j += 1;
        }
        try testing.expectEqual(off.len, i);
        try testing.expectEqual(on.len, j);
        try testing.expect(fused > 0 and left < gemms);
        // each fused GEMM: rot_in's workspace in, the unpack's words, the strides, svh, the whole output, the pick
        for (on, 0..) |c, k| {
            if (!std.mem.eql(u8, c.name, "tf_dsv41_pfdense_v1.gemm")) continue;
            const prev = on[k - 1];
            try testing.expectEqualStrings("tensorfold_exl3_linear_v4.rot_in", prev.name);
            const xh = tensor(c.args[0].arg);
            try testing.expectEqualStrings("s.pf.xh", roleName(xh));
            try testing.expectEqualStrings("s.pf.xh", roleName(tensor(prev.args[2].arg)));
            const words = tensor(c.args[1].arg);
            const out = tensor(c.args[6].arg);
            const k2: u32 = @intCast(c.args[7].arg.i);
            const K: usize = @intCast(xh.shape[1]);
            const N: usize = @intCast(out.shape[1]);
            try testing.expectEqual(xh.shape[0], n);
            try testing.expectEqual(out.shape[0], n);
            try testing.expectEqual(@as(i64, 1), out.stride[1]);
            try testing.expect(c.args[5].arg == .none);
            const lanes = c.args[8].arg.b;
            try testing.expect(knob_types.PfDense.eligible(@intCast(K), @intCast(N), k2, lanes));
            const name = roleName(words);
            try testing.expect(std.mem.endsWith(u8, name, if (lanes) ".lanes" else if (words.shape.len == 3) ".trellis" else if (N == 128) ".trellis" else ".T"));
            // the words cover the whole layer: K / 16 k steps of every 128-column block
            const tw: i64 = 4 * @as(i64, k2);
            var numel: i64 = 1;
            for (words.shape) |d| numel *= d;
            try testing.expectEqual(@as(i64, @intCast(K / 16 * N / 128)) * 8 * tw, numel);
            const pick = pfd.pick(K, N, k2, @intCast(n));
            try testing.expectEqual(@as(i64, pick[0]), c.args[9].arg.i);
            try testing.expectEqual(@as(i64, pick[1]), c.args[10].arg.i);
        }
        // the W_q workspace is gone: no role reads or writes "s.pf.w"
        for (on) |c| for (c.args) |x| if (x.arg == .t) try testing.expect(!std.mem.eql(u8, roleName(x.arg.t), "s.pf.w"));
    }
}

test "the stream top-k's resident scratch for the boot check: none under the threshold, stream_topk.BUDGET at 1M" {
    const cfg: Config = .{};
    // 4,096 positions: no layer sees 4,096 compressed keys
    try testing.expectEqual(@as(u64, 0), pk.streamScratch(&cfg, 4096, 2048));
    // 1M context: a 2,048-row segment streams 1,024 rows a launch (= BUDGET, 256 MiB); 512 rows fit whole (128 MiB)
    try testing.expectEqual(@as(u64, 256 << 20), pk.streamScratch(&cfg, 1 << 20, 2048));
    try testing.expectEqual(@as(u64, 128 << 20), pk.streamScratch(&cfg, 1 << 20, 512));
    // the emitter's own L.ix.stream at the limit's last segment is the same size
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    var w = try widths(aa);
    const layers = [_]u32{ 0, 1, 2, 3 };
    const cs = try bp.emitPrefill(aa, &cfg, &w, .{ .limit = 1 << 20, .rope_rows = (1 << 20) + 2048 }, &layers, 2048, (1 << 20) - 2048, false);
    var most: u64 = 0;
    for (cs) |c| for (c.args) |x| if (x.arg == .t and x.arg.t.role == .buf and std.mem.eql(u8, x.arg.t.role.buf, "L.ix.stream")) {
        var n: u64 = 8;
        for (x.arg.t.shape) |d| n *= @intCast(d);
        most = @max(most, n);
    };
    try testing.expectEqual(pk.streamScratch(&cfg, 1 << 20, 2048), most);
}

test "KV producer norm/store fusion is separately opt-in" {
    Fake.pairs = &.{};
    try testing.expect(!pk.kvNormStore(&Fake.get));
    for ([_][]const u8{ "0", "false", "yes", "2", "" }) |value| {
        Fake.pairs = &.{.{ "TF_DSV41_KV_NORM_STORE", value }};
        try testing.expect(!pk.kvNormStore(&Fake.get));
    }
    Fake.pairs = &.{.{ "TF_DSV41_ROUTER_GROUP_ROT", "1" }};
    try testing.expect(!pk.kvNormStore(&Fake.get));
    Fake.pairs = &.{.{ "TF_DSV41_KV_NORM_STORE", " 1 " }};
    try testing.expect(pk.kvNormStore(&Fake.get));
    try testing.expect(!pk.routerGroupRot(&Fake.get));
    Fake.pairs = &.{};
}
