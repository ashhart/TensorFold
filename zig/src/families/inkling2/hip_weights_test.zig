//! Inkling 2 Mini's text weights on one HIP rank without a device: the release's header over a recording runtime.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const st = core.safetensors;
const config = @import("config.zig");
const plan = @import("plan.zig");
const names = @import("names.zig");
const weights = @import("weights.zig");
const hw = @import("hip_weights.zig");
const t = std.testing;

const mini_header = @embedFile("fixtures/mini_model_header.txt");
/// The release's 1,244 tensors less the vision tower's 422.
const text_tensors = 822;

/// A runtime whose allocations are addresses nobody reads and whose uploads are recorded call by call.
const Device = struct {
    const Copy = struct { dst: u64, src: usize, n: usize };
    var copies: std.ArrayList(Copy) = .empty;
    var sizes: std.ArrayList(usize) = .empty;
    var frees: usize = 0;
    var next: u64 = 1 << 40;
    var fail_alloc = false;
    var fail_src: ?usize = null;

    fn alloc(out: *hip.abi.DevicePtr, n: usize) callconv(.c) hip.abi.Result {
        sizes.append(t.allocator, n) catch return 1;
        if (fail_alloc) return 2;
        out.* = @ptrFromInt(next);
        next += std.mem.alignForward(u64, n, 1 << 21);
        return 0;
    }

    fn free(_: hip.abi.DevicePtr) callconv(.c) hip.abi.Result {
        frees += 1;
        return 0;
    }

    fn upload(dst: hip.abi.DevicePtr, src: [*]const u8, n: usize) callconv(.c) hip.abi.Result {
        copies.append(t.allocator, .{ .dst = @intFromPtr(dst), .src = @intFromPtr(src), .n = n }) catch return 1;
        if (fail_src) |f| if (f == @intFromPtr(src)) return 1;
        return 0;
    }

    fn runtime() hip.Runtime {
        var r: hip.Runtime = .{ .lib = undefined, .api = undefined };
        r.api.hipMalloc = alloc;
        r.api.hipFree = free;
        r.api.hipMemcpyHtoD = upload;
        return r;
    }

    fn reset() void {
        copies.deinit(t.allocator);
        sizes.deinit(t.allocator);
        copies = .empty;
        sizes = .empty;
        frees = 0;
        next = 1 << 40;
        fail_alloc = false;
        fail_src = null;
    }
};

/// The release's model file as one mapped safetensors file: its real tensor spans over reserved, unreadable memory.
const Release = struct {
    arena: std.heap.ArenaAllocator,
    entries: st.Header,
    space: []align(std.heap.page_size_min) u8,
    ck: core.Checkpoint,
    c: config.Config,
    p: plan.Plan,

    fn open(r: *Release) !void {
        r.arena = .init(t.allocator);
        errdefer r.arena.deinit();
        const a = r.arena.allocator();
        r.entries = try @import("names_test.zig").unfold(a, mini_header, "inkling2-mini-model");
        var end: usize = 0;
        for (r.entries.values()) |*e| {
            var n = e.dtype.size();
            for (e.shape[0..e.rank]) |d| n *= d;
            e.begin = end;
            e.end = end + n;
            end += n;
        }
        r.space = try std.posix.mmap(null, end, .{}, .{ .TYPE = .PRIVATE, .ANONYMOUS = true, .NORESERVE = true }, -1, 0);
        errdefer std.posix.munmap(r.space);
        var file: st.File = undefined;
        file.map.memory = r.space;
        file.data = 0;
        file.names = r.entries;
        r.ck = .{ .gpa = t.allocator, .io = undefined };
        errdefer r.forget();
        try r.ck.files.append(t.allocator, file);
        try r.ck.skip("model.visual."); // as the CUDA opener does: text-only Mini never loads the tower
        var why: config.Why = .{};
        r.c = try config.parse(a, @import("config_test.zig").mini, &why);
        var prec: [24]plan.Precision = undefined;
        try names.precisions(r.entries, r.c.text.layers, &prec);
        r.p = try plan.build(&r.c.text, &prec);
    }

    /// The checkpoint's own lists; its one file is this test's, never opened, so it is never closed.
    fn forget(r: *Release) void {
        r.ck.used.deinit(t.allocator);
        r.ck.files.deinit(t.allocator);
    }

    fn close(r: *Release) void {
        r.forget();
        std.posix.munmap(r.space);
        r.arena.deinit();
    }

    fn bytes(r: *const Release, name: []const u8) []const u8 {
        const e = r.entries.get(name).?;
        return r.space[e.begin..e.end];
    }
};

/// Where the load put each text tensor, by name, and the names a field check has matched.
const Placed = struct {
    at: std.StringHashMapUnmanaged(u64) = .empty,
    matched: std.StringHashMapUnmanaged(void) = .empty,

    fn deinit(p: *Placed) void {
        p.at.deinit(t.allocator);
        p.matched.deinit(t.allocator);
    }

    fn expect(p: *Placed, got: u64, comptime fmt: []const u8, args: anytype) !void {
        var buf: [160]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, fmt, args);
        try t.expectEqual(p.at.get(name) orelse return error.TestNoSuchUpload, got);
        try p.matched.put(t.allocator, p.at.getKey(name).?, {});
    }
};

test "inkling2 hip weights: every text tensor of Mini's release lands once, 256-aligned, in one arena of the checkpoint's bytes" {
    defer Device.reset();
    var rel: Release = undefined;
    try rel.open();
    defer rel.close();
    var r = Device.runtime();
    var w = try hw.load(t.allocator, &r, &rel.ck, &rel.c, rel.p);
    var live = true;
    errdefer if (live) w.deinit();
    var padded: usize = 0;
    var raw: usize = 0;
    var count: usize = 0;
    for (rel.entries.keys(), rel.entries.values()) |name, e| if (std.mem.startsWith(u8, name, "model.llm.")) {
        padded += std.mem.alignForward(usize, e.end - e.begin, 256);
        raw += e.end - e.begin;
        count += 1;
    };
    try t.expectEqual(@as(usize, text_tensors), count);
    try t.expectEqualSlices(usize, &.{padded}, Device.sizes.items);
    try t.expectEqual(padded, try weights.checkpointBytes(&rel.ck));
    try t.expectEqual(raw, w.bytes);
    try t.expectEqual(@as(usize, 0), rel.ck.unused());
    const base = w.buffers.items[0].ptr;
    var copied: usize = 0;
    for (Device.copies.items) |x| copied += x.n;
    try t.expectEqual(raw, copied); // no tensor went up twice
    var by_src: std.AutoHashMapUnmanaged(usize, usize) = .empty;
    defer by_src.deinit(t.allocator);
    for (Device.copies.items, 0..) |x, i| try by_src.put(t.allocator, x.src, i);
    var placed: Placed = .{};
    defer placed.deinit();
    const Span = struct { at: u64, n: usize };
    var spans: std.ArrayList(Span) = .empty;
    defer spans.deinit(t.allocator);
    for (rel.entries.keys()) |name| if (std.mem.startsWith(u8, name, "model.llm.")) {
        const src = rel.bytes(name);
        var i = by_src.get(@intFromPtr(src.ptr)) orelse return error.TestTensorNeverUploaded;
        const dst = Device.copies.items[i].dst;
        var done: usize = 0;
        while (done < src.len) : (i += 1) { // the windows of one upload, in order
            const x = Device.copies.items[i];
            try t.expectEqual(@intFromPtr(src.ptr) + done, x.src);
            try t.expectEqual(dst + done, x.dst);
            done += x.n;
        }
        try t.expectEqual(src.len, done);
        try t.expectEqual(@as(u64, 0), (dst - base) % 256);
        try t.expect(dst >= base and dst + src.len <= base + padded);
        try placed.at.put(t.allocator, name, dst);
        try spans.append(t.allocator, .{ .at = dst, .n = src.len });
    };
    std.mem.sort(Span, spans.items, {}, struct {
        fn lt(_: void, x: Span, y: Span) bool {
            return x.at < y.at;
        }
    }.lt);
    for (spans.items[1..], spans.items[0 .. spans.items.len - 1]) |s, before| try t.expect(s.at >= before.at + before.n);

    // every address the launchers will read is the upload of the tensor it names
    try placed.expect(w.embed, "model.llm.embed.weight", .{});
    try placed.expect(w.embed_norm, "model.llm.embed_norm.weight", .{});
    try placed.expect(w.norm, "model.llm.norm.weight", .{});
    try placed.expect(w.unembed, "model.llm.unembed.weight", .{});
    try placed.expect(w.output_query, "model.llm.attn_res.output_query", .{});
    try placed.expect(w.cross_qm, "model.llm.middle_cycle_cross_pass_queries", .{});
    try placed.expect(w.cross_kv, "model.llm.middle_cycle_cross_pass_kv_queries", .{});
    try t.expectEqual(@as(usize, 24), w.layers.len);
    for (w.layers, 0..) |l, i| {
        const global = i % 6 == 5;
        try placed.expect(w.mlp_queries[i], "model.llm.attn_res.mlp_queries.{d}", .{i});
        if (i == 0) try t.expectEqual(@as(u64, 0), w.attn_queries[0]) else try placed.expect(w.attn_queries[i], "model.llm.attn_res.attn_queries.{d}", .{i - 1});
        const a = l.attn;
        const p = "model.llm.layers.{d}.";
        try placed.expect(a.wq, p ++ "attn.wq_du.weight", .{i});
        try placed.expect(a.wk, p ++ "attn.wk_dv.weight", .{i});
        try placed.expect(a.wv, p ++ "attn.wv_dv.weight", .{i});
        try placed.expect(a.wr, p ++ "attn.wr_du.weight", .{i});
        try placed.expect(a.wo, p ++ "attn.wo_ud.weight", .{i});
        try placed.expect(a.gate, p ++ "attn.gate_attn.weight", .{i});
        try placed.expect(a.sink, p ++ "attn.attn_sink_proj.weight", .{i});
        try placed.expect(a.q_scale, p ++ "attn.q_l2_clip.scale", .{i});
        try placed.expect(a.rel_scale, p ++ "attn.rel_l2_clip.scale", .{i});
        try placed.expect(a.rel_proj, p ++ "attn.rel_logits_proj.proj", .{i});
        if (global) {
            try placed.expect(a.k_conv, p ++ "attn.k_sconv.static_w", .{i});
            try placed.expect(a.v_conv, p ++ "attn.v_sconv.static_w", .{i});
            try placed.expect(a.k_dynamic, p ++ "attn.k_sconv.weight_proj", .{i});
            try placed.expect(a.v_dynamic, p ++ "attn.v_sconv.weight_proj", .{i});
            try placed.expect(a.norms[0], p ++ "q_input_norm.weight", .{i});
            try placed.expect(a.norms[1], p ++ "k_input_norm.weight", .{i});
            try placed.expect(a.norms[2], p ++ "v_input_norm.weight", .{i});
        } else {
            try placed.expect(a.k_conv, p ++ "attn.k_sconv.weight", .{i});
            try placed.expect(a.v_conv, p ++ "attn.v_sconv.weight", .{i});
            try t.expectEqual([2]u64{ 0, 0 }, [2]u64{ a.k_dynamic, a.v_dynamic });
            for (a.norms) |n| try placed.expect(n, p ++ "attn_norm.weight", .{i}); // q, k and v share one upload
        }
        try placed.expect(l.attn_conv, p ++ "attn_sconv.weight", .{i});
        try placed.expect(l.mlp_norm, p ++ "mlp_norm.weight", .{i});
        try placed.expect(l.mlp_conv, p ++ "mlp_sconv.weight", .{i});
        try placed.expect(l.router, p ++ "mlp.gate.weight", .{i});
        try placed.expect(l.router_bias, p ++ "mlp.gate.bias", .{i});
        // the stored expert format is the plan's, read from the header: NVFP4 on layers 0 to 19, MXFP8 on 20 to 23
        const e = l.experts;
        try t.expectEqualStrings(@tagName(rel.p.layers[i].precision), @tagName(e.format));
        try t.expectEqualStrings(if (i < 20) "nvfp4" else "mxfp8", @tagName(e.format));
        try t.expectEqual(@as(u32, 128), e.experts);
        try t.expectEqual([4]u32{ 2304, 2304, 2304, 1152 }, [4]u32{ e.w13.n, e.w13.k, e.w2.n, e.w2.k });
        inline for (.{ "w13", "w2" }) |which| {
            const proj = @field(e, which);
            try placed.expect(proj.codes, p ++ "mlp.experts." ++ which ++ "_weight", .{i});
            try placed.expect(proj.scales, p ++ "mlp.experts." ++ which ++ "_weight_scale", .{i});
            if (e.format == .nvfp4) {
                try placed.expect(proj.amax, p ++ "mlp.experts." ++ which ++ "_weight_amax", .{i});
                try placed.expect(proj.denominator, p ++ "mlp.experts." ++ which ++ "_weight_outer_scale_denominator", .{i});
            } else try t.expectEqual([2]u64{ 0, 0 }, [2]u64{ proj.amax, proj.denominator });
        }
        // the loop's gates, in loop_math's message order q, k, v and MLP, only on the core layers 6 to 17
        for (l.controllers, [4][]const u8{ "attn", "k", "v", "mlp" }) |g, role| {
            if (i < 6 or i >= 18) {
                try t.expectEqual([3]u64{ 0, 0, 0 }, [3]u64{ g.down, g.forget, g.gain });
                continue;
            }
            try placed.expect(g.down, p ++ "controllers.{s}.down_weight", .{ i, role });
            try placed.expect(g.forget, p ++ "controllers.{s}.forget_weight", .{ i, role });
            try placed.expect(g.gain, p ++ "controllers.{s}.gain_weight", .{ i, role });
        }
    }
    try t.expectEqual(@as(usize, text_tensors), placed.matched.count()); // no upload is left without its field
    live = false;
    w.deinit();
    try t.expectEqual(@as(usize, 1), Device.frees);
}

test "inkling2 hip weights: a core layer's cross-pass queries are rows 2j and 2j + 1 of both matrices, j its place in the core" {
    defer Device.reset();
    var rel: Release = undefined;
    try rel.open();
    defer rel.close();
    var r = Device.runtime();
    var w = try hw.load(t.allocator, &r, &rel.ck, &rel.c, rel.p);
    defer w.deinit();
    for (6..18) |layer| {
        const q = try hw.crossQueries(&w, @intCast(layer));
        const row = 2 * (layer - 6) * 2304 * 4;
        try t.expectEqual(w.cross_qm + row, q.qm);
        try t.expectEqual(w.cross_kv + row, q.kv);
    }
    for ([_]u32{ 0, 5, 18, 23 }) |layer| try t.expectError(error.Inkling2NotCoreLayer, hw.crossQueries(&w, layer));
}

test "inkling2 hip weights: a config without the loop is refused before any device allocation" {
    defer Device.reset();
    var rel: Release = undefined;
    try rel.open();
    defer rel.close();
    var r = Device.runtime();
    rel.c.text.loop = null;
    try t.expectError(error.Inkling2LoopRequired, hw.load(t.allocator, &r, &rel.ck, &rel.c, rel.p));
    try t.expectEqual(@as(usize, 0), Device.sizes.items.len);
    try t.expectEqual(@as(usize, 0), Device.copies.items.len);
}

test "inkling2 hip weights: the arena refuses a span past its end and keeps what it handed out" {
    defer Device.reset();
    var r = Device.runtime();
    var w: hw.Weights = .{ .gpa = t.allocator, .c = undefined, .p = undefined };
    var l: weights.Loader(hw.Buffer) = .{ .w = &w, .arena = try hw.Buffer.alloc(&r, 512) };
    defer l.arena.free();
    try t.expectEqual(l.arena.ptr, try l.alloc(200));
    try t.expectError(error.LoopWeightsOverflow, l.alloc(300)); // 256-aligned, it would end at 556
    try t.expectEqual(@as(usize, 200), l.used);
    try t.expectEqual(@as(usize, 200), w.bytes);
    try t.expectEqual(l.arena.ptr + 256, try l.alloc(256)); // the refused span left the arena as it was
    try t.expectEqual(@as(usize, 456), w.bytes);
}

test "inkling2 hip weights: every visit with messages reads the rows loop_engine reads; other layers and no loop are refused" {
    defer Device.reset();
    var rel: Release = undefined;
    try rel.open();
    defer rel.close();
    var r = Device.runtime();
    var w = try hw.load(t.allocator, &r, &rel.ck, &rel.c, rel.p);
    defer w.deinit();
    const lp = w.c.text.loop.?;
    const h: usize = w.c.text.hidden;
    var visits: usize = 0;
    for (rel.p.visits[0..rel.p.n_visits]) |v| if (v.hasMessages()) {
        visits += 1;
        const core_j = v.layer - lp.core_start; // loop_engine.zig's own arithmetic for the messages' query rows
        const q = try hw.crossQueries(&w, v.layer);
        try t.expectEqual(w.cross_qm + 2 * core_j * h * 4, q.qm);
        try t.expectEqual(w.cross_kv + 2 * core_j * h * 4, q.kv);
    };
    try t.expectEqual(@as(usize, 24), visits); // passes 2 and 3 of the 12 core layers
    try t.expectError(error.Inkling2NotCoreLayer, hw.crossQueries(&w, std.math.maxInt(u32)));
    const loop = w.c.text.loop;
    w.c.text.loop = null;
    defer w.c.text.loop = loop;
    try t.expectError(error.Inkling2NotCoreLayer, hw.crossQueries(&w, 6));
}

test "inkling2 hip weights: a failed allocation uploads nothing and a failed upload frees the arena once" {
    defer Device.reset();
    var rel: Release = undefined;
    try rel.open();
    defer rel.close();
    var r = Device.runtime();
    Device.fail_alloc = true;
    try t.expectError(error.HipFailed, hw.load(t.allocator, &r, &rel.ck, &rel.c, rel.p));
    try t.expectEqual(@as(usize, 0), Device.copies.items.len);
    try t.expectEqual(@as(usize, 0), Device.frees);
    Device.reset();
    var again: Release = undefined;
    try again.open();
    defer again.close();
    Device.fail_src = @intFromPtr(again.bytes("model.llm.layers.9.attn.wo_ud.weight").ptr);
    try t.expectError(error.HipFailed, hw.load(t.allocator, &r, &again.ck, &again.c, again.p));
    try t.expectEqual(@as(usize, 1), Device.sizes.items.len);
    try t.expectEqual(@as(usize, 1), Device.frees); // and std.testing.allocator sees no host leak
}
