//! MiniMax-H3-Image's prompt stage on CUDA: Qwen3-VL text to layer 50, refiner; ~85 GiB streams via two pinned windows.
const std = @import("std");
pub const seconds = @import("clock.zig").seconds;
const cuda = @import("cuda");
pub const rules = @import("picture_prompt.zig");
pub const reference = @import("prompt_reference.zig");
pub const case = @import("prompt_case.zig");
pub const picture = @import("picture.zig");
pub const checkpoint = @import("checkpoint.zig");
pub const read_ahead = @import("read_ahead.zig");
const report = @import("report.zig");
const bf16_round = @import("bf16_round");

const D = rules.head_dim;
const mode_qwen: c_int = 0;
const mode_h3: c_int = 1;

const image align(16) = @embedFile("fatbin_h3_image_prompt").*;
const kernel_names = [_][:0]const u8{ "h3p_norm", "h3p_heads", "h3p_attention", "h3p_glu", "h3p_add", "h3p_scatter_add", "h3p_embed", "h3p_splice", "h3p_linear", "h3p_round" };

/// A presentation's vision rows, as prompt.Vision: the tower's merged bf16 rows at the pads and DeepStack's blocks.
pub const Vision = struct {
    pads: []const u32,
    rows: []const u16,
    deepstack: [3][]const u16,

    fn check(v: Vision, T: usize, H: usize) !usize {
        const n = v.pads.len;
        for (v.pads, 0..) |at, i| if (at >= T or (i > 0 and at <= v.pads[i - 1])) return error.H3BadVisionRows;
        if (v.rows.len != n * H) return error.H3BadVisionRows;
        for (v.deepstack) |d| if (d.len != n * H) return error.H3BadVisionRows;
        return n;
    }
};

/// A checked weight tensor's bf16 bytes, read in place from a mapped shard.
pub const Tensor = []const u8;
/// A text layer's tensors in rules.Layer field order.
pub const LayerSet = rules.Layer(Tensor);
/// A refiner block's tensors in rules.Block field order.
pub const BlockSet = rules.Block(Tensor);

/// Device scratch parts, 256-aligned: products write shared fp32 `p` one at a time, h3p_round to bf16 out of place.
const EncodePart = enum { x, xn, p, qb, kb, vb, att, gb, ub, act, cos, sin, idx, vision, deep };
const RefinePart = enum { x, xn, p, qkv, att, wide, act };

fn Scratch(comptime Part: type) type {
    return struct {
        buf: cuda.DeviceBuffer,
        offs: [std.enums.values(Part).len]usize,

        fn make(d: *const cuda.Driver, sizes: std.enums.EnumArray(Part, usize)) !@This() {
            var s: @This() = .{ .buf = undefined, .offs = undefined };
            var total: usize = 0;
            for (std.enums.values(Part), 0..) |p, i| {
                s.offs[i] = total;
                total += std.mem.alignForward(usize, @max(sizes.get(p), 1), 256);
            }
            s.buf = try cuda.DeviceBuffer.alloc(d, total);
            return s;
        }

        fn at(s: @This(), p: Part) u64 {
            return s.buf.ptr + s.offs[@backingInt(p)];
        }
    };
}

/// One buffered device tensor: uploaded through the runner's stages, freed by `free`.
const Dev = struct {
    buf: cuda.DeviceBuffer,

    /// `bytes` into the slot kept for it, allocated on the first layer and reused.
    fn into(r: *Runner, slot: *?cuda.DeviceBuffer, bytes: []const u8) !void {
        if (slot.*) |*buf| {
            if (buf.len < bytes.len) return error.H3UnexpectedTensor;
            try r.copy(bytes, buf.ptr);
        } else {
            var buf = try cuda.DeviceBuffer.alloc(&r.driver, bytes.len);
            errdefer {
                // a staged copy can still run: drain before the destination leaves the device
                cuda.completion.drain(&r.driver, r.stream, "h3 prompt cuda: an upload did not drain");
                buf.free();
            }
            try r.copy(bytes, buf.ptr);
            slot.* = buf;
        }
    }
    /// The bf16 `bytes` as fp32 values, on the allocator's lifetime until the call returns.
    fn asF32(a: std.mem.Allocator, bytes: []const u8) ![]f32 {
        if (bytes.len % 2 != 0) return error.H3UnexpectedTensor;
        const bits = std.mem.bytesAsSlice(u16, bytes);
        const f = try a.alloc(f32, bits.len);
        for (f, bits) |*o, b| o.* = bf16_round.value(b);
        return f;
    }

    fn upload(r: *Runner, bytes: []const u8) !Dev {
        var d: Dev = .{ .buf = try cuda.DeviceBuffer.alloc(&r.driver, bytes.len) };
        errdefer {
            // a staged copy can still run: drain before the destination leaves the device
            cuda.completion.drain(&r.driver, r.stream, "h3 prompt cuda: an upload did not drain");
            d.buf.free();
        }
        try r.copy(bytes, d.buf.ptr);
        return d;
    }

    /// The bf16 tensor converted once to fp32 on the host, for the fp32 norm-weight inputs.
    fn uploadF32(r: *Runner, a: std.mem.Allocator, bytes: []const u8) !Dev {
        const f = try asF32(a, bytes);
        defer a.free(f);
        return upload(r, std.mem.sliceAsBytes(f));
    }
};

/// One product's plan (cuBLASLt bf16 in, fp32 sums and out); the separate bias plan gives fused epilogues a bias only.
const Products = struct {
    lt: cuda.cublaslt.Library,
    workspace: cuda.DeviceBuffer,
    gpa: std.mem.Allocator,
    /// (rows, n, k, bias) to its plan.
    plans: std.AutoHashMapUnmanaged([4]u64, cuda.cublaslt.Linear) = .empty,

    const workspace_bytes = 32 << 20;
    const canonical_rows = 16384;

    fn init(gpa: std.mem.Allocator, d: *const cuda.Driver) !Products {
        var lt = try cuda.cublaslt.Library.open();
        errdefer lt.close();
        return .{ .lt = lt, .gpa = gpa, .workspace = try cuda.DeviceBuffer.alloc(d, workspace_bytes) };
    }

    fn deinit(p: *Products) void {
        var it = p.plans.valueIterator();
        while (it.next()) |l| l.deinit();
        p.plans.deinit(p.gpa);
        p.workspace.free();
        p.lt.close();
    }

    /// y (rows, n) fp32 = x (rows, k) bf16 . w (n, k) bf16^T; bias [n] fp32 fused when nonzero.
    fn run(p: *Products, stream: cuda.Stream, x: u64, w: u64, bias: u64, y: u64, rows: usize, n: usize, k: usize) !void {
        const key = [4]u64{ rows, n, k, @intFromBool(bias != 0) };
        const got = try p.plans.getOrPut(p.gpa, key);
        if (!got.found_existing) {
            errdefer p.plans.removeByPtr(got.key_ptr);
            // Prefer the canonical algorithm, but short prompts must use an algorithm valid at their own shape.
            var probe = try cuda.cublaslt.Linear.initEpilogue(&p.lt, canonical_rows, n, k, .f32, workspace_bytes, if (bias != 0) .bias else .none);
            defer probe.deinit();
            var l = try cuda.cublaslt.Linear.initEpilogue(&p.lt, rows, n, k, .f32, workspace_bytes, if (bias != 0) .bias else .none);
            errdefer l.deinit();
            if (bias != 0) {
                const epilogue: u32 = @backingInt(cuda.cublaslt.Epilogue.bias);
                try p.lt.check(p.lt.api.cublasLtMatmulDescSetAttribute(l.desc, cuda.cublaslt.desc_epilogue, &epilogue, @sizeOf(u32)), "set H3 condition bias");
            }
            var checked: cuda.cublaslt.Heuristic = undefined;
            const s = p.lt.api.cublasLtMatmulAlgoCheck(l.handle, l.desc, l.a, l.b, l.c, l.c, &probe.algo, &checked);
            if (s == 0 and checked.state == 0 and checked.workspace_size <= workspace_bytes) {
                l.algo = probe.algo;
            } else {
                const actual = p.lt.api.cublasLtMatmulAlgoCheck(l.handle, l.desc, l.a, l.b, l.c, l.c, &l.algo, &checked);
                if (actual != 0 or checked.state != 0 or checked.workspace_size > workspace_bytes) return error.NoAlgorithm;
            }
            l.workspace_need = checked.workspace_size;
            if (bias != 0) l.epilogue = .bias;
            got.value_ptr.* = l;
        }
        if (bias != 0) return got.value_ptr.runBias(x, w, bias, y, p.workspace.ptr, workspace_bytes, stream.handle);
        try got.value_ptr.run(x, w, y, p.workspace.ptr, workspace_bytes, stream.handle);
    }
};

pub const Runner = struct {
    gpa: std.mem.Allocator,
    driver: cuda.Driver,
    ctx: cuda.Context,
    module: cuda.Module,
    functions: [kernel_names.len]cuda.Function,
    stream: cuda.Stream,
    products: Products,
    /// Two pinned windows: a chunk's copy runs while the next is being filled.
    stages: [2]cuda.copy_source.Stage = .{ .{}, .{} },
    /// GPU seconds of the last encode or refine.
    gpu: f64 = 0,

    fn function(r: *const Runner, comptime name: []const u8) cuda.Function {
        inline for (kernel_names, 0..) |n, i| if (comptime std.mem.eql(u8, n, name)) return r.functions[i];
        @compileError("no kernel " ++ name);
    }

    /// A launch whose failure leaves the device drained before buffers owned by the caller are freed.
    fn launch(r: *const Runner, comptime name: []const u8, grid: cuda.Dim3, threads: u32, args: *cuda.Args) !void {
        errdefer cuda.completion.drain(&r.driver, r.stream, "h3 prompt cuda: a launch did not drain");
        try cuda.launch.launch(r.function(name), .{ .grid = grid, .block = .{ .x = threads } }, r.stream, args);
    }

    /// The driver, context, kernels and cuBLASLt plans; heap-made: members hold `&r.driver`, which a copy would move.
    pub fn init(gpa: std.mem.Allocator) !*Runner {
        const r = try gpa.create(Runner);
        errdefer gpa.destroy(r);
        r.gpa = gpa;
        r.stages = .{ .{}, .{} };
        r.gpu = 0;
        r.driver = try cuda.Driver.open();
        errdefer r.driver.close();
        r.ctx = try cuda.Context.shared(&r.driver);
        errdefer r.ctx.deinit();
        r.module = try cuda.Module.load(&r.driver, &image);
        errdefer r.module.unload();
        inline for (kernel_names, 0..) |name, i| r.functions[i] = try r.module.function(name);
        r.stream = try cuda.Stream.init(&r.driver, true);
        errdefer r.stream.deinit();
        r.products = try Products.init(gpa, &r.driver);
        return r;
    }

    pub fn deinit(r: *Runner) void {
        r.stream.synchronize() catch |e| report.say("h3 prompt cuda: work still queued at close: {t}", .{e});
        for (&r.stages) |*stage| stage.deinit();
        r.products.deinit();
        r.stream.deinit();
        r.module.unload();
        _ = r.driver.api.cuDevicePrimaryCtxRelease_v2(r.ctx.device);
        r.driver.close();
        r.gpa.destroy(r);
    }

    /// `bytes` to `dst` in 64 MiB windows via two pinned stages, each waited when its turn returns; stream-ordered.
    fn copy(r: *Runner, bytes: []const u8, dst: u64) !void {
        var done: usize = 0;
        var turn: usize = 0;
        while (done < bytes.len) {
            const n = @min(bytes.len - done, 64 << 20);
            const window = try r.stages[turn].take(&r.driver, n);
            @memcpy(window, bytes[done..][0..n]);
            try r.stages[turn].copyRangeTo(.{ .d = &r.driver, .ptr = dst + done, .len = n }, 0, 0, n, r.stream);
            done += n;
            turn ^= 1;
        }
    }

    fn linear(r: *const Runner, x: u64, w: u64, y: u64, rows: usize, n: usize, k: usize) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ x, w, y }) |p| args.add(p);
        for ([_]usize{ rows, n, k }) |v| args.add(@as(c_int, @intCast(v)));
        try r.launch("h3p_linear", .{ .x = @intCast((rows * n + 255) / 256) }, 256, &args);
    }

    fn norm(r: *const Runner, x: u64, w: u64, y: u64, rows: usize, width: usize, eps: f32, mode: c_int) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ x, w, y }) |p| args.add(p);
        for ([_]usize{ width, @intCast(mode) }) |v| args.add(@as(c_int, @intCast(v)));
        args.add(eps);
        try r.launch("h3p_norm", .{ .x = @intCast(rows) }, 256, &args);
    }

    /// The text model's heads: q, k bf16 in place, normed, rotated by the [rows, 64] tables `rot` (cos, sin pointers).
    fn heads(r: *const Runner, q: u64, k: u64, wq: u64, wk: u64, rot: ?[2]u64, count: usize, kv_count: usize, q_row: usize, k_row: usize, head: usize, rows: usize, eps: f32, mode: c_int) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ q, k, wq, wk, if (rot) |t| t[0] else 0, if (rot) |t| t[1] else 0 }) |p| args.add(p);
        for ([_]usize{ count, kv_count, q_row, k_row, head, @intCast(mode), @intFromBool(rot != null) }) |v| args.add(@as(c_int, @intCast(v)));
        args.add(eps);
        try r.launch("h3p_heads", .{ .x = @intCast(rows), .y = @intCast((count + kv_count + 7) / 8) }, 256, &args);
    }

    fn attention(r: *const Runner, q: u64, k: u64, v: u64, o: u64, rows: usize, count: usize, kv_count: usize, q_row: usize, k_row: usize, v_row: usize, head: usize, causal: bool) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ q, k, v, o }) |p| args.add(p);
        for ([_]usize{ rows, count, kv_count, q_row, k_row, v_row, head, @intFromBool(causal) }) |n| args.add(@as(c_int, @intCast(n)));
        args.add(@as(f32, 1.0 / @sqrt(@as(f32, D))));
        try r.launch("h3p_attention", .{ .x = @intCast(rows), .y = @intCast((count + 7) / 8) }, 256, &args);
    }

    fn glu(r: *const Runner, g: u64, v: u64, o: u64, rows: usize, width: usize, g_row: usize, v_row: usize, mode: c_int) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ g, v, o }) |p| args.add(p);
        for ([_]usize{ width, g_row, v_row, @intCast(mode) }) |n| args.add(@as(c_int, @intCast(n)));
        try r.launch("h3p_glu", .{ .x = @intCast((width + 255) / 256), .y = @intCast(rows) }, 256, &args);
    }

    fn add(r: *const Runner, x: u64, y: u64, n: usize) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ x, y }) |p| args.add(p);
        args.add(@as(c_int, @intCast(n)));
        try r.launch("h3p_add", .{ .x = @intCast((n + 255) / 256) }, 256, &args);
    }

    fn round(r: *const Runner, x: u64, y: u64, n: usize) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ x, y }) |p| args.add(p);
        args.add(@as(c_int, @intCast(n)));
        try r.launch("h3p_round", .{ .x = @intCast((n + 255) / 256) }, 256, &args);
    }

    fn indexed(r: *const Runner, comptime name: []const u8, a_ptr: u64, b_ptr: u64, idx: u64, rows: usize, width: usize) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ a_ptr, b_ptr, idx }) |p| args.add(p);
        args.add(@as(c_int, @intCast(width)));
        try r.launch(name, .{ .x = @intCast((width + 255) / 256), .y = @intCast(rows) }, 256, &args);
    }

    /// Last-layer rows [T, hidden] bf16 for `ids` at M-RoPE `pos`, vision spliced at pads; `layer` live till return.
    pub fn encode(
        r: *Runner,
        a: std.mem.Allocator,
        g: rules.Text,
        embed: Tensor,
        ids: []const u32,
        pos: []const [3]i32,
        vision: ?Vision,
        layer: *const fn (ctx: *anyopaque, li: usize) anyerror!LayerSet,
        ctx: *anyopaque,
        out: []u16,
    ) !void {
        const T = ids.len;
        const H = g.hidden;
        const qh = g.heads * D;
        const kvh = g.kv_heads * D;
        if (T == 0 or pos.len != T or out.len != T * H or g.layers == 0 or embed.len != g.vocab * H * 2) return error.H3PromptRows;
        const n = if (vision) |v| try v.check(T, H) else 0;
        var sizes = std.enums.EnumArray(EncodePart, usize).initFill(T * H * 2);
        // `p` holds one product's fp32 rows at a time; the rest are bf16 buffers at their width
        sizes.set(.p, T * @max(H, @max(qh, g.ffn)) * 4);
        sizes.set(.qb, T * qh * 2);
        sizes.set(.kb, T * kvh * 2);
        sizes.set(.vb, T * kvh * 2);
        sizes.set(.att, T * qh * 2);
        sizes.set(.gb, T * g.ffn * 2);
        sizes.set(.ub, T * g.ffn * 2);
        sizes.set(.act, T * g.ffn * 2);
        sizes.set(.cos, T * 64 * 4);
        sizes.set(.sin, T * 64 * 4);
        sizes.set(.idx, (T + n) * 4);
        sizes.set(.vision, n * H * 2);
        sizes.set(.deep, 3 * n * H * 2);
        var s = try Scratch(EncodePart).make(&r.driver, sizes);
        defer s.buf.free();
        errdefer cuda.completion.drain(&r.driver, r.stream, "H3 prompt encode did not drain");

        // the rotary tables on the host, as picture_prompt.rotary writes them
        const cs = try a.alloc(f32, T * 64);
        defer a.free(cs);
        const sn = try a.alloc(f32, T * 64);
        defer a.free(sn);
        rules.rotary(cs, sn, pos, g.theta, g.sections);
        try r.copy(std.mem.sliceAsBytes(cs), s.at(.cos));
        try r.copy(std.mem.sliceAsBytes(sn), s.at(.sin));
        {
            const idx = try a.alloc(u32, T + n);
            defer a.free(idx);
            @memcpy(idx[0..T], ids);
            if (vision) |v| @memcpy(idx[T..], v.pads);
            try r.copy(std.mem.sliceAsBytes(idx), s.at(.idx));
            if (vision) |v| {
                try r.copy(std.mem.sliceAsBytes(v.rows), s.at(.vision));
                for (v.deepstack, 0..) |d, i| try r.copy(std.mem.sliceAsBytes(d), s.at(.deep) + i * n * H * 2);
            }
        }
        {
            var table = try Dev.upload(r, embed);
            defer table.buf.free();
            errdefer cuda.completion.drain(&r.driver, r.stream, "H3 prompt embedding did not drain");
            try r.indexed("h3p_embed", table.buf.ptr, s.at(.idx), s.at(.x), T, H);
            if (vision) |_| try r.indexed("h3p_splice", s.at(.x), s.at(.vision), s.at(.idx) + T * 4, n, H);
            // the table leaves the device once the gather and splice have run
            try r.stream.synchronize();
        }

        var begun = try cuda.Event.init(&r.driver, true);
        defer begun.deinit();
        var ended = try cuda.Event.init(&r.driver, true);
        defer ended.deinit();
        errdefer cuda.completion.drain(&r.driver, r.stream, "H3 prompt timing did not drain");
        try begun.record(r.stream);
        // one set of layer buffers for the run, each layer's tensors copied into it
        var w: LayerDev = .{ .bufs = @splat(null), .ptrs = undefined };
        defer w.free();
        errdefer cuda.completion.drain(&r.driver, r.stream, "H3 prompt layer did not drain");
        for (0..g.layers) |li| {
            const set = try layer(ctx, li);
            try uploadLayer(r, a, &w, set);
            try r.norm(s.at(.x), w.ptrs.norm1, s.at(.xn), T, H, g.eps, mode_qwen);
            try r.linear(s.at(.xn), w.ptrs.q, s.at(.p), T, qh, H);
            try r.round(s.at(.p), s.at(.qb), T * qh);
            try r.linear(s.at(.xn), w.ptrs.k, s.at(.p), T, kvh, H);
            try r.round(s.at(.p), s.at(.kb), T * kvh);
            try r.linear(s.at(.xn), w.ptrs.v, s.at(.p), T, kvh, H);
            try r.round(s.at(.p), s.at(.vb), T * kvh);
            try r.heads(s.at(.qb), s.at(.kb), w.ptrs.q_norm, w.ptrs.k_norm, .{ s.at(.cos), s.at(.sin) }, g.heads, g.kv_heads, qh, kvh, D, T, g.eps, mode_qwen);
            try r.attention(s.at(.qb), s.at(.kb), s.at(.vb), s.at(.att), T, g.heads, g.kv_heads, qh, kvh, kvh, D, true);
            try r.linear(s.at(.att), w.ptrs.o, s.at(.p), T, H, qh);
            try r.add(s.at(.x), s.at(.p), T * H);
            try r.norm(s.at(.x), w.ptrs.norm2, s.at(.xn), T, H, g.eps, mode_qwen);
            try r.linear(s.at(.xn), w.ptrs.gate, s.at(.p), T, g.ffn, H);
            try r.round(s.at(.p), s.at(.gb), T * g.ffn);
            try r.linear(s.at(.xn), w.ptrs.up, s.at(.p), T, g.ffn, H);
            try r.round(s.at(.p), s.at(.ub), T * g.ffn);
            try r.glu(s.at(.gb), s.at(.ub), s.at(.act), T, g.ffn, g.ffn, g.ffn, mode_qwen);
            try r.linear(s.at(.act), w.ptrs.down, s.at(.p), T, H, g.ffn);
            try r.add(s.at(.x), s.at(.p), T * H);
            if (n > 0 and li < 3) try r.indexed("h3p_scatter_add", s.at(.x), s.at(.deep) + li * n * H * 2, s.at(.idx) + T * 4, n, H);
        }
        try ended.record(r.stream);
        try ended.synchronize();
        r.gpu = @as(f64, try cuda.Event.elapsedMs(begun, ended)) / 1000;
        var dst: cuda.DeviceBuffer = .{ .d = &r.driver, .ptr = s.at(.x), .len = out.len * 2 };
        try dst.download(0, std.mem.sliceAsBytes(out));
    }

    /// A layer's 11 device buffers held for the run; a later tensor of another size is refused rather than reallocated.
    const LayerDev = struct {
        bufs: [11]?cuda.DeviceBuffer,
        ptrs: rules.Layer(u64),

        fn free(w: *LayerDev) void {
            for (&w.bufs) |*b| if (b.*) |*held| held.free();
        }
    };

    fn uploadLayer(r: *Runner, a: std.mem.Allocator, w: *LayerDev, set: LayerSet) !void {
        inline for (comptime std.meta.fieldNames(LayerSet), 0..) |field, i| {
            const t = @field(set, field);
            if (t.len % 2 != 0) return error.H3UnexpectedTensor;
            const is_norm = comptime std.mem.indexOf(u8, field, "norm") != null;
            if (is_norm) {
                const f = try Dev.asF32(a, t);
                defer a.free(f);
                try Dev.into(r, &w.bufs[i], std.mem.sliceAsBytes(f));
            } else try Dev.into(r, &w.bufs[i], t);
            @field(w.ptrs, field) = w.bufs[i].?.ptr;
        }
    }

    /// condition_proj, refiner blocks, final norm: rows [T, text] bf16 to `out` [T, hidden]; `block` live till return.
    pub fn refine(
        r: *Runner,
        a: std.mem.Allocator,
        g: rules.Refiner,
        cond_w: Tensor,
        cond_b: Tensor,
        final_norm: Tensor,
        block: *const fn (ctx: *anyopaque, bi: usize) anyerror!BlockSet,
        ctx: *anyopaque,
        rows: []const u16,
        out: []u16,
    ) !void {
        const H = g.hidden;
        const inner = g.heads * D;
        const T = rows.len / g.text;
        if (T == 0 or rows.len != T * g.text or out.len != T * H or g.blocks == 0) return error.H3PromptRows;
        var sizes = std.enums.EnumArray(RefinePart, usize).initFill(T * @max(g.text, H) * 2);
        sizes.set(.p, T * @max(H, @max(3 * inner, 2 * g.mlp)) * 4);
        sizes.set(.qkv, T * 3 * inner * 2);
        sizes.set(.att, T * inner * 2);
        sizes.set(.wide, T * 2 * g.mlp * 2);
        sizes.set(.act, T * g.mlp * 2);
        var s = try Scratch(RefinePart).make(&r.driver, sizes);
        defer s.buf.free();
        errdefer cuda.completion.drain(&r.driver, r.stream, "H3 refiner did not drain");
        try r.copy(std.mem.sliceAsBytes(rows), s.at(.x));

        var begun = try cuda.Event.init(&r.driver, true);
        defer begun.deinit();
        var ended = try cuda.Event.init(&r.driver, true);
        defer ended.deinit();
        errdefer cuda.completion.drain(&r.driver, r.stream, "H3 refiner timing did not drain");
        try begun.record(r.stream);
        // condition_proj: its bias fused in fp32 into the product, then the reference's one rounding
        {
            var cw = try Dev.upload(r, cond_w);
            defer cw.buf.free();
            var cb = try Dev.uploadF32(r, a, cond_b);
            defer cb.buf.free();
            errdefer cuda.completion.drain(&r.driver, r.stream, "H3 condition projection did not drain");
            try r.products.run(r.stream, s.at(.x), cw.buf.ptr, cb.buf.ptr, s.at(.p), T, H, g.text);
            try r.round(s.at(.p), s.at(.x), T * H);
            try r.stream.synchronize();
        }
        const fused_row = 3 * inner;
        // one set of block buffers for the run, each block's tensors copied into it
        var b: BlockDev = .{ .bufs = @splat(null), .ptrs = undefined };
        defer b.free();
        errdefer cuda.completion.drain(&r.driver, r.stream, "H3 refiner block did not drain");
        for (0..g.blocks) |bi| {
            const set = try block(ctx, bi);
            try uploadBlock(r, a, &b, set);
            try r.norm(s.at(.x), b.get("norm1"), s.at(.xn), T, H, g.eps, mode_h3);
            try r.products.run(r.stream, s.at(.xn), b.get("qkv"), 0, s.at(.p), T, 3 * inner, H);
            try r.round(s.at(.p), s.at(.qkv), T * 3 * inner);
            // the fused rows' q and k heads normed in place, no rotary; v sits `head` = 384 apart
            try r.heads(s.at(.qkv), s.at(.qkv) + D * 2, b.get("q_norm"), b.get("k_norm"), null, g.heads, g.heads, fused_row, fused_row, 3 * D, T, g.qk_eps, mode_h3);
            try r.attention(s.at(.qkv), s.at(.qkv) + D * 2, s.at(.qkv) + 2 * D * 2, s.at(.att), T, g.heads, g.heads, fused_row, fused_row, fused_row, 3 * D, false);
            try r.products.run(r.stream, s.at(.att), b.get("out"), 0, s.at(.p), T, H, inner);
            try r.add(s.at(.x), s.at(.p), T * H);
            try r.norm(s.at(.x), b.get("norm2"), s.at(.xn), T, H, g.eps, mode_h3);
            try r.products.run(r.stream, s.at(.xn), b.get("fc1"), 0, s.at(.p), T, 2 * g.mlp, H);
            try r.round(s.at(.p), s.at(.wide), T * 2 * g.mlp);
            try r.glu(s.at(.wide), s.at(.wide) + g.mlp * 2, s.at(.act), T, g.mlp, 2 * g.mlp, 2 * g.mlp, mode_h3);
            try r.products.run(r.stream, s.at(.act), b.get("fc2"), 0, s.at(.p), T, H, g.mlp);
            try r.add(s.at(.x), s.at(.p), T * H);
        }
        {
            var fw = try Dev.uploadF32(r, a, final_norm);
            defer fw.buf.free();
            errdefer cuda.completion.drain(&r.driver, r.stream, "H3 refiner norm did not drain");
            try r.norm(s.at(.x), fw.buf.ptr, s.at(.xn), T, H, g.final_eps, mode_h3);
            try r.stream.synchronize();
        }
        try ended.record(r.stream);
        try ended.synchronize();
        r.gpu = @as(f64, try cuda.Event.elapsedMs(begun, ended)) / 1000;
        // the final norm wrote bf16 rows straight to `out`'s shape
        var dst: cuda.DeviceBuffer = .{ .d = &r.driver, .ptr = s.at(.xn), .len = out.len * 2 };
        try dst.download(0, std.mem.sliceAsBytes(out));
    }

    /// A refiner block's 8 tensors' device buffers, held for the run as the encoder's layer buffers are.
    const BlockDev = struct {
        bufs: [8]?cuda.DeviceBuffer,
        ptrs: rules.Block(u64),

        fn free(b: *BlockDev) void {
            for (&b.bufs) |*d| if (d.*) |*held| held.free();
        }
        fn get(b: *const BlockDev, comptime field: []const u8) u64 {
            return @field(b.ptrs, field);
        }
    };

    fn uploadBlock(r: *Runner, a: std.mem.Allocator, b: *BlockDev, set: BlockSet) !void {
        inline for (comptime std.meta.fieldNames(BlockSet), 0..) |field, i| {
            const t = @field(set, field);
            if (t.len % 2 != 0) return error.H3UnexpectedTensor;
            const is_norm = comptime std.mem.indexOf(u8, field, "norm") != null;
            if (is_norm) {
                const f = try Dev.asF32(a, t);
                defer a.free(f);
                try Dev.into(r, &b.bufs[i], std.mem.sliceAsBytes(f));
            } else try Dev.into(r, &b.bufs[i], t);
            @field(b.ptrs, field) = b.bufs[i].?.ptr;
        }
    }
};

test "condition projection keeps bias when the actual-row heuristic cannot fuse it" {
    try @import("prompt_cuda_products_test.zig").check(Products, Products.run);
}
