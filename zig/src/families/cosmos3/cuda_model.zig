//! Cosmos3-Edge's reasoner on CUDA: text projections on MLX 8-bit tiles, cosmos3.cu's glue, SigLIP2 on cuBLASLt.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const c = @import("config.zig");
const vit = cuda.vit;
const AffineN = cuda.qlinear.AffineN;
const Allocator = std.mem.Allocator;

/// Rows a verify window holds, rows a call of windows holds, rows a prompt call holds (a longer prompt goes in chunks).
pub const window_rows = 16;
pub const batch_rows = 32;
pub const prompt_rows = 2048;
/// Positions a stream holds at most.
pub const max_context = 131072;
/// Rows a projection call reads through the one-row tiles at most; more take the prompt GEMM.
const decode_rows = 16;
/// Rows the attention kernel gives a block (a warp each).
const attn_rows = 8;

/// A text layer: q, k and v one projection (their rows stacked), o, the MLP's fc1 and fc2, the two norms.
pub const Layer = struct { input_norm: u64, post_norm: u64, qkv: AffineN, o: AffineN, fc1: AffineN, fc2: AffineN };
/// The fused projection's row: queries, then keys, then values.
const qkv_width = (c.query_heads + 2 * c.kv_heads) * c.head_dim;
/// A tower Linear widened to bf16 [n, k] with its bias [n].
pub const Dense = struct { w: u64, b: u64 };
pub const Norm = struct { w: u64, b: u64 };
pub const VisionLayer = struct { ln1: Norm, ln2: Norm, qkv: Dense, out: Dense, fc1: Dense, fc2: Dense };

/// A stream's keys and values: [layer][keys | values][kv head][capacity][head dim] bf16.
pub const Kv = struct {
    base: u64,
    capacity: usize,

    pub fn bytes(capacity: usize) usize {
        return c.layers * 2 * c.kv_heads * capacity * c.head_dim * 2;
    }
    fn at(kv: Kv, layer: usize, values: bool) u64 {
        const half = c.kv_heads * kv.capacity * c.head_dim * 2;
        return kv.base + (2 * layer + @intFromBool(values)) * half;
    }
};

/// One stream's rows in a call: its cache, where its rows sit, and its rows' (t, h, w) and cache slots.
pub const Segment = struct { kv: Kv, first: usize, rows: usize };

const Fns = struct { embed: cuda.Function, place: cuda.Function, norm: cuda.Function, qk: cuda.Function, relu2: cuda.Function, attn: cuda.Function, attn1: cuda.Function, widen8: cuda.Function };

pub const Model = struct {
    gpa: Allocator,
    driver: cuda.Driver,
    ctx: cuda.Context,
    stream: cuda.Stream,
    mods: [4]cuda.Module,
    wide: cuda.qlinear.AffineNKernels,
    fns: Fns,
    theta: f32,
    config: c.Config,
    owned: std.ArrayList(cuda.DeviceBuffer) = .empty,
    // weights
    embed: struct { codes: u64, scales: u64, biases: u64 },
    head: AffineN,
    norm: u64,
    layers: [c.layers]Layer,
    // scratch: one call of up to `rows` rows
    rows: usize,
    s: struct { ids: u64, pos: u64, slots: u64, map: u64, h: u64, x: u64, r: u64, y: u64, qkv: u64, queries: u64, act: u64, sums: u64, logits: u64 },
    host: cuda.HostBuffer, // pinned: the call's index rows in
    logits: cuda.HostBuffer, // pinned: the call's logits rows out (bf16, batch rows of the vocabulary)
    tower: ?Tower = null,

    pub fn alloc(m: *Model, bytes: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(&m.driver, @max(bytes, 256));
        errdefer b.free();
        try m.owned.append(m.gpa, b);
        return b.ptr;
    }

    fn put(m: *Model, dst: u64, bytes: []const u8) !void {
        // the checkpoint's mapped pages are pageable: copy_source stages them and returns once they have landed
        if (bytes.len > 0) cuda.copy_source.synchronousCopy(&m.driver, dst, bytes, std.heap.page_allocator) catch |e| {
            std.log.err("cosmos3: {d} bytes from {*} to 0x{x} failed", .{ bytes.len, bytes.ptr, dst });
            return e;
        };
    }

    /// The model in `dir` on device 0: its text model always, its image tower when `image_tokens` > 0.
    pub fn load(gpa: Allocator, io: std.Io, dir: []const u8, config: *const c.Config, rows: usize, image_tokens: usize) !*Model {
        if (config.bits != 8) return error.UnsupportedCosmosQuantization; // the tiles and the tower's widening read 8-bit codes
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.* = .{ .gpa = gpa, .driver = undefined, .ctx = undefined, .stream = undefined, .mods = undefined, .wide = undefined, .fns = undefined, .theta = config.theta, .config = config.*, .embed = undefined, .head = undefined, .norm = 0, .layers = undefined, .rows = rows, .s = undefined, .host = undefined, .logits = undefined };
        m.driver = try cuda.Driver.open();
        errdefer m.driver.close();
        m.ctx = try cuda.Context.init(&m.driver, 0);
        errdefer m.ctx.deinit();
        try m.ctx.makeCurrent();
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        const images = [_][]const u8{ cuda.kernels.affine8, cuda.kernels.affine_prefill, cuda.kernels.affine4_pack, cuda.kernels.cosmos3 };
        var loaded: usize = 0;
        errdefer for (m.mods[0..loaded]) |*mod| mod.unload();
        for (images, &m.mods) |img, *mod| {
            mod.* = try cuda.Module.load(&m.driver, img);
            loaded += 1;
        }
        const sms: usize = @intCast(try m.ctx.attribute(.multiprocessor_count));
        const gb10 = try m.ctx.attribute(.compute_capability_major) == 12 and try m.ctx.attribute(.compute_capability_minor) == 1;
        m.wide = try cuda.qlinear.AffineNKernels.resolve(&m.driver, m.mods[0], m.mods[1], m.mods[2], sms, gb10);
        errdefer m.wide.deinit();
        inline for (@typeInfo(Fns).@"struct".field_names) |name| @field(m.fns, name) = try m.mods[3].function("tf_cos3_" ++ name);
        m.stream = try cuda.Stream.init(&m.driver, true);
        errdefer m.stream.deinit();
        errdefer {
            for (m.owned.items) |*b| b.free();
            m.owned.deinit(gpa);
        }
        var ck = try core.Checkpoint.openModel(gpa, io, dir);
        defer ck.close();
        try m.loadText(&ck);
        try m.scratch(rows);
        if (image_tokens > 0) {
            m.tower = try Tower.init(m, &ck, image_tokens);
            // a camera frame's shapes (640 x 480 or 480 x 640: 1200 patches) planned now, not on the first question
            const gh = 30;
            const gw = 40;
            if (gh * gw <= m.tower.?.patches_most) {
                const pixels = try gpa.alloc(f32, gh * gw * c.patch_width);
                defer gpa.free(pixels);
                @memset(pixels, 0);
                try m.tower.?.encode(m, pixels, gh, gw, try m.alloc(2 * gh * gw / 4 * c.hidden));
            }
        }
        try m.stream.synchronize();
        return m;
    }

    pub fn deinit(m: *Model) void {
        m.ctx.makeCurrent() catch {};
        if (m.tower) |*t| t.deinit(m);
        m.host.free();
        m.logits.free();
        for (m.owned.items) |*b| b.free();
        m.owned.deinit(m.gpa);
        m.stream.deinit();
        m.wide.deinit();
        for (&m.mods) |*mod| mod.unload();
        m.ctx.deinit();
        m.driver.close();
        m.gpa.destroy(m);
    }

    fn tensor(ck: *core.Checkpoint, comptime fmt: []const u8, args: anytype, dtype: core.checkpoint.DType, shape: []const usize) ![]const u8 {
        var buf: [200]u8 = undefined;
        return (try ck.expect(try std.fmt.bufPrint(&buf, fmt, args), dtype, shape)).bytes;
    }

    /// 8-bit MLX projections of k inputs: `parts` (name, rows) stacked as one, staged at `stage`, packed as tiles.
    fn affine(m: *Model, ck: *core.Checkpoint, stage: u64, parts: []const struct { []const u8, usize }, k: usize) !AffineN {
        var n: usize = 0;
        for (parts) |p| n += p[1];
        // codes, then scales, then biases, each [n, ...] with the parts' rows in order
        const at = [3]u64{ stage, stage + std.mem.alignForward(u64, n * k, 256), stage + std.mem.alignForward(u64, n * k, 256) + std.mem.alignForward(u64, n * k / c.group * 2, 256) };
        var row: usize = 0;
        for (parts) |p| {
            const name, const rows = p;
            try m.put(at[0] + row * k, try tensor(ck, "{s}.weight", .{name}, .u32, &.{ rows, k / 4 }));
            try m.put(at[1] + row * k / c.group * 2, try tensor(ck, "{s}.scales", .{name}, .bf16, &.{ rows, k / c.group }));
            try m.put(at[2] + row * k / c.group * 2, try tensor(ck, "{s}.biases", .{name}, .bf16, &.{ rows, k / c.group }));
            row += rows;
        }
        const l = AffineN.layout(n, k, 8);
        const q: AffineN = .{ .w = try m.alloc(l.words), .s = try m.alloc(l.scales), .b = try m.alloc(l.scales), .n = n, .k = k, .npad = l.npad, .bits = 8 };
        try m.wide.packWords(m.stream, at[0], 8, n, k, q.w);
        try m.wide.packTable(m.stream, at[1], n, k, q.npad, q.s);
        try m.wide.packTable(m.stream, at[2], n, k, q.npad, q.b);
        try m.stream.synchronize(); // the next projection's staging overwrites this one's
        return q;
    }

    fn upload(m: *Model, bytes: []const u8) !u64 {
        const p = try m.alloc(bytes.len);
        try m.put(p, bytes);
        return p;
    }

    fn loadText(m: *Model, ck: *core.Checkpoint) !void {
        // staging for the largest projection: the head's codes and its tables
        var stage = try cuda.DeviceBuffer.alloc(&m.driver, c.vocab * c.hidden + 4 * c.vocab * (c.hidden / c.group) + 4096);
        defer stage.free();
        m.embed = .{
            .codes = try m.upload(try tensor(ck, "model.embed_tokens.weight", .{}, .u32, &.{ c.vocab, c.hidden / 4 })),
            .scales = try m.upload(try tensor(ck, "model.embed_tokens.scales", .{}, .bf16, &.{ c.vocab, c.hidden / c.group })),
            .biases = try m.upload(try tensor(ck, "model.embed_tokens.biases", .{}, .bf16, &.{ c.vocab, c.hidden / c.group })),
        };
        m.head = try m.affine(ck, stage.ptr, &.{.{ "lm_head", c.vocab }}, c.hidden);
        m.norm = try m.upload(try tensor(ck, "model.norm.weight", .{}, .bf16, &.{c.hidden}));
        const qd = c.query_heads * c.head_dim;
        const kvd = c.kv_heads * c.head_dim;
        for (&m.layers, 0..) |*l, i| {
            var b: [6][96]u8 = undefined;
            const p = struct {
                fn name(buf: *[96]u8, layer: usize, part: []const u8) ![]const u8 {
                    return std.fmt.bufPrint(buf, "model.layers.{d}.{s}", .{ layer, part });
                }
            }.name;
            l.input_norm = try m.upload(try tensor(ck, "model.layers.{d}.input_layernorm.weight", .{i}, .bf16, &.{c.hidden}));
            l.post_norm = try m.upload(try tensor(ck, "model.layers.{d}.post_attention_layernorm.weight", .{i}, .bf16, &.{c.hidden}));
            l.qkv = try m.affine(ck, stage.ptr, &.{ .{ try p(&b[0], i, "self_attn.q_proj"), qd }, .{ try p(&b[1], i, "self_attn.k_proj"), kvd }, .{ try p(&b[2], i, "self_attn.v_proj"), kvd } }, c.hidden);
            l.o = try m.affine(ck, stage.ptr, &.{.{ try p(&b[3], i, "self_attn.o_proj"), c.hidden }}, qd);
            l.fc1 = try m.affine(ck, stage.ptr, &.{.{ try p(&b[4], i, "mlp.fc1"), c.intermediate }}, c.hidden);
            l.fc2 = try m.affine(ck, stage.ptr, &.{.{ try p(&b[5], i, "mlp.fc2"), c.hidden }}, c.intermediate);
        }
    }

    fn scratch(m: *Model, rows: usize) !void {
        const R = rows + attn_rows;
        const sizes = [_]usize{ 4 * R, 12 * R, 4 * R, 4 * R, 2 * R * c.hidden, 2 * R * c.hidden, 2 * R * c.hidden, 2 * R * c.hidden, 2 * R * qkv_width, 2 * R * c.hidden, 2 * R * c.intermediate, 4 * R * (c.intermediate / 64), 2 * batch_rows * c.vocab };
        var at: [sizes.len]u64 = undefined;
        for (sizes, &at) |n, *p| p.* = try m.alloc(n);
        m.s = .{ .ids = at[0], .pos = at[1], .slots = at[2], .map = at[3], .h = at[4], .x = at[5], .r = at[6], .y = at[7], .qkv = at[8], .queries = at[9], .act = at[10], .sums = at[11], .logits = at[12] };
        m.host = try cuda.HostBuffer.alloc(&m.driver, 4 * R * 6);
        errdefer m.host.free();
        m.logits = try cuda.HostBuffer.alloc(&m.driver, 2 * batch_rows * c.vocab);
    }

    fn launch(m: *Model, f: cuda.Function, grid: [3]usize, block: usize, args: anytype) !void {
        var a: cuda.Args = .{};
        inline for (args) |v| a.add(v);
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = @intCast(block) } }, m.stream, &a);
    }

    /// y [rows, n] = x [rows, k] . W^T on the tiles: decode up to 16 rows on x's group sums `xs`, else the GEMM.
    fn project(m: *Model, w: AffineN, x: u64, xs: u64, y: u64, rows: usize) !void {
        if (rows > decode_rows) return m.wide.prompt(m.stream, x, w, y, rows);
        try m.wide.decode(m.stream, x, xs, w, y, rows);
    }

    fn normRows(m: *Model, weight: u64, add: bool, rows: usize) !void {
        try m.launch(m.fns.norm, .{ rows, 1, 1 }, 256, .{ m.s.h, m.s.r, weight, c.eps, @as(c_int, @intFromBool(add)), m.s.x, m.s.sums });
    }

    pub const Head = enum { last, all };
    /// The tower's rows for a call's images, and for each row of the call the image row it takes (-1: its token's).
    pub const Images = struct { rows: u64, map: []const i32 };

    /// `ids` through the layers per segment (cache, `pos`, `slots`), image rows over `map`; bf16 logits in `m.logits`.
    pub fn run(m: *Model, segments: []const Segment, ids: []const u32, pos: []const [3]i32, slots: []const u32, head: Head, images: ?Images) !void {
        const total = ids.len;
        if (total == 0 or total > m.rows or (head == .all and total > batch_rows)) return error.CosmosRowsExceeded;
        const host = m.host.slice(i32);
        for (ids, host[0..total]) |t, *o| o.* = @intCast(t);
        for (pos, 0..) |p, r| @memcpy(host[total + 3 * r ..][0..3], &p);
        for (slots, host[4 * total ..][0..total]) |t, *o| o.* = @intCast(t);
        if (images) |im| @memcpy(host[5 * total ..][0..total], im.map);
        const hb = std.mem.sliceAsBytes(host);
        const api = m.driver.api;
        const s = m.stream.handle;
        try m.driver.check(api.cuMemcpyHtoDAsync_v2(m.s.ids, hb.ptr, 4 * total, s), "ids");
        try m.driver.check(api.cuMemcpyHtoDAsync_v2(m.s.pos, hb.ptr + 4 * total, 12 * total, s), "pos");
        try m.driver.check(api.cuMemcpyHtoDAsync_v2(m.s.slots, hb.ptr + 16 * total, 4 * total, s), "slots");
        if (images != null) try m.driver.check(api.cuMemcpyHtoDAsync_v2(m.s.map, hb.ptr + 20 * total, 4 * total, s), "map");
        const out_rows: usize = if (head == .all) total else 1;
        try m.body(segments, total, head, images);
        try m.driver.check(api.cuMemcpyDtoHAsync_v2(m.logits.bytes.ptr, m.s.logits, 2 * out_rows * c.vocab, s), "logits");
        try m.stream.synchronize();
    }

    /// The layers and the head over the uploaded rows.
    fn body(m: *Model, segments: []const Segment, total: usize, head: Head, images: ?Images) !void {
        try m.launch(m.fns.embed, .{ total, c.hidden / 256, 1 }, 256, .{ m.s.ids, m.embed.codes, m.embed.scales, m.embed.biases, m.s.h });
        if (images) |im| try m.launch(m.fns.place, .{ total, c.hidden / 256, 1 }, 256, .{ m.s.h, im.rows, m.s.map });
        const log2_theta: f32 = std.math.log2(m.theta);
        const scale: f32 = 1.0 / @sqrt(@as(f32, c.head_dim));
        for (m.layers, 0..) |l, li| {
            try m.normRows(l.input_norm, li != 0, total);
            try m.project(l.qkv, m.s.x, m.s.sums, m.s.qkv, total);
            var base: usize = 0;
            for (segments) |seg| {
                const keys = seg.kv.at(li, false);
                const values = seg.kv.at(li, true);
                try m.launch(m.fns.qk, .{ c.query_heads + c.kv_heads, seg.rows, 1 }, 32, .{
                    m.s.qkv + 2 * base * qkv_width,    m.s.pos + 12 * base, m.s.slots + 4 * base, @as(c_int, @intCast(seg.kv.capacity)), log2_theta,
                    m.s.queries + 2 * base * c.hidden, keys,                values,
                });
                if (seg.rows == 1) try m.launch(m.fns.attn1, .{ c.query_heads, 1, 1 }, 256, .{
                    m.s.queries + 2 * base * c.hidden,     keys,  values,                                m.s.y + 2 * base * c.hidden, m.s.slots + 4 * base,
                    @as(c_int, @intCast(seg.kv.capacity)), scale, m.s.sums + 4 * base * (c.hidden / 64),
                }) else try m.launch(m.fns.attn, .{ c.query_heads, (seg.rows + attn_rows - 1) / attn_rows, 1 }, 32 * attn_rows, .{
                    m.s.queries + 2 * base * c.hidden,     keys,  values,                                m.s.y + 2 * base * c.hidden, m.s.slots + 4 * base, @as(c_int, @intCast(seg.rows)),
                    @as(c_int, @intCast(seg.kv.capacity)), scale, m.s.sums + 4 * base * (c.hidden / 64),
                });
                base += seg.rows;
            }
            try m.project(l.o, m.s.y, m.s.sums, m.s.r, total);
            try m.normRows(l.post_norm, true, total);
            try m.project(l.fc1, m.s.x, m.s.sums, m.s.act, total);
            try m.launch(m.fns.relu2, .{ total, c.intermediate / 256, 1 }, 256, .{ m.s.act, @as(c_int, c.intermediate), m.s.sums });
            try m.project(l.fc2, m.s.act, m.s.sums, m.s.r, total);
        }
        try m.normRows(m.norm, true, total);
        const out_rows: usize = if (head == .all) total else 1;
        const last = if (head == .all) 0 else total - 1;
        try m.project(m.head, m.s.x + 2 * last * c.hidden, m.s.sums + 4 * last * (c.hidden / 64), m.s.logits, out_rows);
    }

    /// Row `r`'s logits of the last call (bf16 words).
    pub fn logitsRow(m: *const Model, r: usize) []const u16 {
        return m.logits.slice(u16)[r * c.vocab ..][0..c.vocab];
    }
};

/// The image tower: SigLIP2 patches plus the resized 16 x 16 grid, 27 pre-norm blocks, then the 2 x 2 projector.
pub const Tower = struct {
    k: vit.Kernels,
    at: vit.Attention,
    g: vit.Gemms,
    patch: Dense,
    grid: []f32, // the learned position table, f32 [16 * 16, 1152], on the host
    layers: [c.v_layers]VisionLayer,
    post: Norm,
    merge_norm: Norm,
    fc1: Dense,
    fc2: Dense,
    patches_most: usize,
    b: struct { px: u64, pos: u64, h: u64, x: u64, qkv: u64, att: u64, mlp: u64, merged: u64, acc: u64 },
    staging: cuda.HostBuffer, // pinned: an image's bf16 patches, then its position rows

    fn init(m: *Model, ck: *core.Checkpoint, tokens: usize) !Tower {
        var t: Tower = undefined;
        t.k = try vit.Kernels.load(&m.driver);
        errdefer t.k.unload();
        t.at = try vit.Attention.load(&m.driver);
        errdefer t.at.unload();
        t.g = try vit.Gemms.init(m.gpa, &m.driver);
        errdefer t.g.deinit();
        const W = c.v_hidden;
        // staging for the widest Linear: the projector's first, its codes and its two tables
        const widest = c.projector_hidden * W * c.merge * c.merge;
        var stage = try cuda.DeviceBuffer.alloc(&m.driver, widest + 4 * widest / c.group + 4096);
        defer stage.free();
        t.patch = try widen(m, ck, stage.ptr, &.{.{ "vision.patch_embed", W }}, c.patch_width);
        const table = try Model.tensor(ck, "vision.position", .{}, .bf16, &.{ c.grid * c.grid, W });
        t.grid = try m.gpa.alloc(f32, c.grid * c.grid * W);
        errdefer m.gpa.free(t.grid);
        const words = std.mem.bytesAsSlice(u16, table);
        for (t.grid, 0..) |*g, i| g.* = @bitCast(@as(u32, words[i]) << 16);
        for (&t.layers, 0..) |*l, i| {
            var names: [6][96]u8 = undefined;
            const n = struct {
                fn f(b: *[96]u8, layer: usize, part: []const u8) ![]const u8 {
                    return std.fmt.bufPrint(b, "vision.layers.{d}.{s}", .{ layer, part });
                }
            }.f;
            l.ln1 = try norm(m, ck, try n(&names[0], i, "layer_norm1"), W);
            l.ln2 = try norm(m, ck, try n(&names[0], i, "layer_norm2"), W);
            const q = try n(&names[1], i, "attn.q_proj");
            const kk = try n(&names[2], i, "attn.k_proj");
            const v = try n(&names[3], i, "attn.v_proj");
            l.qkv = try widen(m, ck, stage.ptr, &.{ .{ q, W }, .{ kk, W }, .{ v, W } }, W);
            const o = try n(&names[4], i, "attn.out_proj");
            l.out = try widen(m, ck, stage.ptr, &.{.{ o, W }}, W);
            const f1 = try n(&names[4], i, "mlp.fc1");
            l.fc1 = try widen(m, ck, stage.ptr, &.{.{ f1, c.v_intermediate }}, W);
            const f2 = try n(&names[5], i, "mlp.fc2");
            l.fc2 = try widen(m, ck, stage.ptr, &.{.{ f2, W }}, c.v_intermediate);
        }
        t.post = try norm(m, ck, "vision.post_layernorm", W);
        t.merge_norm = try norm(m, ck, "projector.norm", W);
        t.fc1 = try widen(m, ck, stage.ptr, &.{.{ "projector.fc1", c.projector_hidden }}, W * c.merge * c.merge);
        t.fc2 = try widen(m, ck, stage.ptr, &.{.{ "projector.fc2", c.hidden }}, c.projector_hidden);
        const P = tokens * c.merge * c.merge;
        t.patches_most = P;
        const rows = P + 128; // the attention reads whole 128-row tiles
        t.b = .{
            .px = try m.alloc(2 * P * c.patch_width),
            .pos = try m.alloc(2 * rows * W),
            .h = try m.alloc(2 * rows * W),
            .x = try m.alloc(2 * rows * W),
            .qkv = try m.alloc(2 * rows * 3 * W),
            .att = try m.alloc(2 * rows * W),
            .mlp = try m.alloc(2 * rows * c.v_intermediate),
            .merged = try m.alloc(2 * tokens * c.projector_hidden),
            .acc = try m.alloc(4 * rows * @max(c.v_intermediate, 3 * W)),
        };
        t.staging = try cuda.HostBuffer.alloc(&m.driver, 2 * P * (c.patch_width + W));
        return t;
    }

    fn deinit(t: *Tower, m: *Model) void {
        t.staging.free();
        m.gpa.free(t.grid);
        t.g.deinit();
        t.at.unload();
        t.k.unload();
    }

    fn norm(m: *Model, ck: *core.Checkpoint, name: []const u8, width: usize) !Norm {
        return .{
            .w = try m.upload(try Model.tensor(ck, "{s}.weight", .{name}, .bf16, &.{width})),
            .b = try m.upload(try Model.tensor(ck, "{s}.bias", .{name}, .bf16, &.{width})),
        };
    }

    /// Linears `parts` (name, rows) of k inputs, stacked in order as one bf16 [sum rows, k] with their biases.
    fn widen(m: *Model, ck: *core.Checkpoint, stage: u64, parts: []const struct { []const u8, usize }, k: usize) !Dense {
        var n: usize = 0;
        for (parts) |p| n += p[1];
        const out: Dense = .{ .w = try m.alloc(2 * n * k), .b = try m.alloc(2 * n) };
        var row: usize = 0;
        for (parts) |p| {
            const name, const rows = p;
            const codes = try Model.tensor(ck, "{s}.weight", .{name}, .u32, &.{ rows, k / 4 });
            const scales = try Model.tensor(ck, "{s}.scales", .{name}, .bf16, &.{ rows, k / c.group });
            const biases = try Model.tensor(ck, "{s}.biases", .{name}, .bf16, &.{ rows, k / c.group });
            const bias = try Model.tensor(ck, "{s}.bias", .{name}, .bf16, &.{rows});
            const s_at = stage + std.mem.alignForward(u64, codes.len, 256);
            const b_at = s_at + std.mem.alignForward(u64, scales.len, 256);
            try m.put(stage, codes);
            try m.put(s_at, scales);
            try m.put(b_at, biases);
            try m.put(out.b + 2 * row, bias);
            const count: i64 = @intCast(rows * k);
            try m.launch(m.fns.widen8, .{ (rows * k + 255) / 256, 1, 1 }, 256, .{ stage, s_at, b_at, out.w + 2 * row * k, @as(c_int, @intCast(k)), count });
            try m.stream.synchronize();
            row += rows;
        }
        return out;
    }

    /// One image's patches (f32 [gh * gw, 768] in 2 x 2 block order) to its text rows ([gh * gw / 4, 2048] at `out`).
    pub fn encode(t: *Tower, m: *Model, pixels: []const f32, gh: u32, gw: u32, out: u64) !void {
        const P: usize = @as(usize, gh) * gw;
        if (gh % c.merge != 0 or gw % c.merge != 0 or P == 0) return error.CosmosImageGrid;
        if (P > t.patches_most) return error.ImageTooLarge;
        if (pixels.len != P * c.patch_width) return error.CosmosImageGrid;
        const W = c.v_hidden;
        const host = t.staging.slice(u16);
        const px = host[0 .. P * c.patch_width];
        for (px, pixels) |*d, x| d.* = vit.bf16(x);
        const pos = host[P * c.patch_width ..][0 .. P * W];
        try @import("cuda_image.zig").positions(m.gpa, t.grid, gh, gw, pos);
        const s = m.stream;
        const api = m.driver.api;
        try m.driver.check(api.cuMemcpyHtoDAsync_v2(t.b.px, px.ptr, 2 * px.len, s.handle), "patches");
        try m.driver.check(api.cuMemcpyHtoDAsync_v2(t.b.pos, pos.ptr, 2 * pos.len, s.handle), "positions");
        const k = &t.k;
        const g = &t.g;
        const b = t.b;
        try vit.linear(g, k, s, b.acc, b.px, t.patch.w, t.patch.b, b.h, P, W, c.patch_width);
        try k.pointwise(s, b.h, b.pos, b.h, P * W, .none);
        const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, c.v_head_dim)));
        for (t.layers) |l| {
            try k.layerNorm(s, b.h, l.ln1.w, l.ln1.b, b.x, P, W, c.v_eps);
            try vit.linear(g, k, s, b.acc, b.x, l.qkv.w, l.qkv.b, b.qkv, P, 3 * W, W);
            try t.at.full(s, b.qkv, b.att, P, c.v_heads, c.v_head_dim, scale);
            try vit.linear(g, k, s, b.acc, b.att, l.out.w, l.out.b, b.x, P, W, W);
            try k.pointwise(s, b.x, b.h, b.h, P * W, .none);
            try k.layerNorm(s, b.h, l.ln2.w, l.ln2.b, b.x, P, W, c.v_eps);
            try vit.linear(g, k, s, b.acc, b.x, l.fc1.w, l.fc1.b, b.mlp, P, c.v_intermediate, W);
            try k.pointwise(s, b.mlp, null, b.mlp, P * c.v_intermediate, .gelu_tanh);
            try vit.linear(g, k, s, b.acc, b.mlp, l.fc2.w, l.fc2.b, b.x, P, W, c.v_intermediate);
            try k.pointwise(s, b.x, b.h, b.h, P * W, .none);
        }
        try k.layerNorm(s, b.h, t.post.w, t.post.b, b.x, P, W, c.v_eps);
        // the projector: each patch normed on its own, then four neighbours (consecutive in block order) one row
        try k.layerNorm(s, b.x, t.merge_norm.w, t.merge_norm.b, b.att, P, W, 1e-6);
        const rows = P / (c.merge * c.merge);
        try vit.linear(g, k, s, b.acc, b.att, t.fc1.w, t.fc1.b, b.merged, rows, c.projector_hidden, W * c.merge * c.merge);
        try k.pointwise(s, b.merged, null, b.merged, rows * c.projector_hidden, .gelu);
        try vit.linear(g, k, s, b.acc, b.merged, t.fc2.w, t.fc2.b, out, rows, c.hidden, c.projector_hidden);
    }
};
