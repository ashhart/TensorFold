//! Ripple's attention on CUDA (group 10, dim 128): window_attention's prompt and decode, tree_attention's global.
const std = @import("std");
const cuda = @import("cuda");

const wa = cuda.window_attention;
const ta = cuda.tree_attention;
const Stream = cuda.Stream;
const Function = cuda.Function;

pub const head_dim = 128;
/// Query heads a block: Ripple's group 10 is no multiple of the shared prompt's 4 (WindowAttention.prompt refuses it).
pub const heads_a_block = 2;
/// Eight warps, each 16 rows of one head: 64 rows a block.
pub const rows_a_block = 16 * (8 / heads_a_block);

/// wattn_kernel<128, 8, 2, 8>: keys in a ring (sliding layers) or linear (global layers).
pub const prompt_symbol = "_ZN16tf_window_prompt12wattn_kernelILi128ELi8ELi2ELi8EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifii";
/// The same over a stream's pages (global layers only: no window).
pub const prompt_pg_symbol = "tf_window_prompt_h2_pg";
/// window_attention.zig's prompt_smem: eight staging slots of 32 keys at head dim 128.
const prompt_smem: u32 = 8 * 32 * 16 * 16;

pub const Kernels = struct {
    prompt: Function,
    prompt_pg: Function,

    /// The two instances from the window_attention image WindowAttention.load loaded (it owns the module).
    pub fn init(w: *const wa.WindowAttention) !Kernels {
        const p = try w.mod.function(prompt_symbol);
        try p.allowDynamicShared(prompt_smem);
        const pg = try w.mod.function(prompt_pg_symbol);
        try pg.allowDynamicShared(prompt_smem);
        return .{ .prompt = p, .prompt_pg = pg };
    }
};

/// A sliding layer: keys (pos - window, pos] of a ring holding key p at row p % ring (prompt and decode).
pub fn sliding(h: usize, hk: usize, window: usize, ring: usize, scale: f32) wa.Window {
    return .{ .h = h, .hk = hk, .d = head_dim, .win = window, .ring = ring, .scale = scale };
}

/// A global layer's prompt rows: every key up to the row's own, linear (key p at row p) or in pages.
pub fn global(h: usize, hk: usize, scale: f32, pg: ?wa.Pages) wa.Window {
    return .{ .h = h, .hk = hk, .d = head_dim, .win = 0, .ring = 0, .scale = scale, .pg = pg };
}

/// A ring holds every key a call's rows see: it never wraps in the call, or it takes the window and the rows.
fn ringHolds(w: wa.Window, p0: usize, n: usize) bool {
    if (w.ring == 0 or p0 + n <= w.ring) return true;
    return w.win > 0 and w.win - 1 + n <= w.ring;
}

/// The shapes the two-heads-a-block kernels take; the shared prompt's checks plus the ring's.
pub fn check(w: wa.Window, p0: usize, n: usize) error{WindowShape}!void {
    if (w.d != head_dim or w.hk == 0 or w.h % w.hk != 0 or (w.h / w.hk) % heads_a_block != 0) return error.WindowShape;
    if (p0 + n > std.math.maxInt(i32) or w.win > std.math.maxInt(i32) or w.ring > std.math.maxInt(i32)) return error.WindowShape;
    if (w.pg) |pg| if (w.win != 0 or w.ring != 0 or pg.tokens == 0 or pg.page_elems > std.math.maxInt(i63)) return error.WindowShape;
    if (!ringHolds(w, p0, n)) return error.WindowShape;
}

/// The launch: (rows / 64) x (key heads x head pairs) blocks of 256 threads, 64 KiB of staged keys.
pub fn config(w: wa.Window, n: usize) cuda.Config {
    const pairs = w.h / w.hk / heads_a_block;
    return .{ .grid = .{ .x = @intCast((n + rows_a_block - 1) / rows_a_block), .y = @intCast(w.hk * pairs), .z = 1 }, .block = .{ .x = 256 }, .shared = prompt_smem };
}

/// n prompt rows of one stream at p0, p0 + 1, ...: q [n, h, 128] -> out [n, h, 128]; kc/vc as WindowAttention.prompt.
pub fn prompt(k: *const Kernels, s: Stream, q: u64, kc: u64, vc: u64, out: u64, p0: usize, n: usize, w: wa.Window) !void {
    try check(w, p0, n);
    if (n == 0) return;
    var a: cuda.Args = .{};
    for ([_]u64{ q, kc, vc, out }) |v| a.add(v);
    for ([_]usize{ p0, n, w.h, w.hk, w.h / w.hk }) |v| a.add(@as(i32, @intCast(v)));
    a.add(w.scale);
    a.add(@as(i32, @intCast(w.win)));
    a.add(@as(i32, @intCast(w.ring)));
    if (w.pg) |pg| {
        a.add(pg.table);
        a.add(@as(i64, @intCast(pg.page_elems)));
        a.add(@as(i32, @intCast(pg.tokens)));
    }
    try cuda.launch.launch(if (w.pg != null) k.prompt_pg else k.prompt, config(w, n), s, &a);
}

/// Global layers' heads: 80 query heads over 8 key heads (group 10).
pub const global_heads = 80;
pub const global_kv_heads = 8;

/// A call's plan for tree_attention, int32 row_stream [m], streams [m, 4], depths [m], then int64 offs [m, 2].
fn planBytes(max_rows: usize) usize {
    return max_rows * (6 * 4 + 2 * 8);
}

/// Row r its own stream: committed keys 0..pos + r from the pages, depth 0 (no path keys); K, V from `origin`.
fn plan(buf: []u8, max_rows: usize, pos: usize, rows: usize, k_off: i64, v_off: i64) void {
    for (0..rows) |r| planRow(buf, max_rows, r, pos + r + 1, k_off, v_off);
}

/// Row r its own stream (and its call table row), `committed` keys from the pages, depth 0; K, V from `origin`.
fn planRow(buf: []u8, max_rows: usize, r: usize, committed: usize, k_off: i64, v_off: i64) void {
    const ints = std.mem.bytesAsSlice(i32, buf[0 .. 24 * max_rows]);
    const offs = std.mem.bytesAsSlice(i64, buf[24 * max_rows ..][0 .. 16 * max_rows]);
    ints[r] = @intCast(r);
    ints[max_rows + 4 * r ..][0..4].* = .{ @intCast(r), 1, @intCast(committed), 0 };
    ints[5 * max_rows + r] = 0;
    offs[2 * r] = k_off;
    offs[2 * r + 1] = v_off;
}

/// Every row's fixed fields, staged once: its own stream and table row, K at the pool, V `v_off` past it, committed 0.
fn stagePlan(buf: []u8, max_rows: usize, v_off: i64) void {
    for (0..max_rows) |r| planRow(buf, max_rows, r, 0, 0, v_off);
}

/// Row r's committed keys (pos[r] + 1) into the staged plan; returns the streams' bytes to upload, 16 a row.
fn commitRows(buf: []u8, max_rows: usize, pos: []const usize) []const u8 {
    const streams = std.mem.bytesAsSlice(i32, buf[4 * max_rows .. 20 * max_rows]);
    for (pos, 0..) |p, r| streams[4 * r + 2] = @intCast(p + 1);
    return buf[4 * max_rows ..][0 .. 16 * pos.len];
}

/// tree_attention's fp32 partials for `max_rows` rows over keys to `context`: o, max, sum by group, then slices.
pub fn scratchBytes(max_rows: usize, context: usize) usize {
    return ta.TreeAttention.scratchBytes(max_rows, global_heads, global_kv_heads, ta.TreeAttention.groups(context), head_dim);
}

/// The device's SMs: tree_attention takes one-warp blocks for small grids (the same bits as four-warp blocks).
fn smCount(d: *const cuda.Driver) !usize {
    var dev: cuda.abi.Device = 0;
    try d.check(d.api.cuCtxGetDevice(&dev), "cuCtxGetDevice");
    var sms: c_int = 0;
    try d.check(d.api.cuDeviceGetAttribute(&sms, .multiprocessor_count, dev), "cuDeviceGetAttribute");
    return @intCast(@max(sms, 1));
}

/// Global decode: tree_attention's paged kernels and merge, each row a stream of its own over the layer's pages.
pub const GlobalDecode = struct {
    tree: ta.TreeAttention,
    plan: cuda.DeviceBuffer,
    scratch: cuda.DeviceBuffer,
    host: []u8,
    gpa: std.mem.Allocator,
    max_rows: usize,
    /// The most keys a row sees (pos + rows): the partials are strided by its groups.
    context: usize,
    /// V's offset from K that `stage` uploaded (null: not staged); decodeGlobal's plan overwrites the staged fields.
    v_off: ?usize = null,

    /// The device bytes init allocates: the partials and the plan.
    pub fn bytes(max_rows: usize, context: usize) usize {
        return scratchBytes(max_rows, context) + std.mem.alignForward(usize, planBytes(max_rows), 256);
    }

    pub fn init(d: *const cuda.Driver, gpa: std.mem.Allocator, max_rows: usize, context: usize) !GlobalDecode {
        if (max_rows == 0 or context < max_rows or context > std.math.maxInt(i32)) return error.GlobalShape;
        var tree = try ta.TreeAttention.load(d);
        errdefer tree.unload();
        tree.sms = try smCount(d);
        const host = try gpa.alloc(u8, planBytes(max_rows));
        errdefer gpa.free(host);
        @memset(host, 0);
        var p = try cuda.DeviceBuffer.alloc(d, host.len);
        errdefer p.free();
        const sc = try cuda.DeviceBuffer.alloc(d, scratchBytes(max_rows, context));
        return .{ .tree = tree, .plan = p, .scratch = sc, .host = host, .gpa = gpa, .max_rows = max_rows, .context = context };
    }

    pub fn deinit(g: *GlobalDecode) void {
        g.scratch.free();
        g.plan.free();
        g.gpa.free(g.host);
        g.tree.unload();
    }

    /// The plan's fixed fields for every row, uploaded once (stagePlan): a round then uploads only its committed keys.
    pub fn stage(g: *GlobalDecode, s: Stream, v_off: usize) !void {
        if (v_off > std.math.maxInt(i63)) return error.GlobalShape;
        stagePlan(g.host, g.max_rows, @intCast(v_off));
        try g.plan.uploadAsync(0, g.host, s.handle);
        g.v_off = v_off;
    }

    /// The key groups a graph's rounds take: those of the context, so its grid and partials' strides serve every round.
    pub fn groups(g: *const GlobalDecode) usize {
        return ta.TreeAttention.groups(g.context);
    }
};

/// q [rows, 80, 128], row r at pos + r over keys 0..pos + r of `pages` (kc, vc: tf_rc_prep's pools) -> out.
pub const DecodeArgs = struct { q: u64, kc: u64, vc: u64, out: u64, pos: usize, rows: usize, pages: wa.Pages, scale: f32 };

/// The calls the plan takes: at most init's rows, keys to its context, pages of whole int32 positions.
pub fn admit(max_rows: usize, context: usize, a: DecodeArgs) error{GlobalShape}!void {
    if (a.rows > max_rows or a.pos > context or a.rows > context - a.pos) return error.GlobalShape;
    const pg = a.pages;
    if (pg.tokens == 0 or pg.tokens > std.math.maxInt(i32) or pg.page_elems == 0 or pg.page_elems > std.math.maxInt(i63)) return error.GlobalShape;
}

/// One global layer's decode rows. The plan is uploaded each call (not inside a CUDA graph capture).
pub fn decodeGlobal(g: *const GlobalDecode, s: Stream, a: DecodeArgs) !void {
    return decodeHeads(g, s, a, global_heads);
}

/// decodeGlobal with `h` query heads over the 8 key heads: Ripple's 80, or a control's other group.
pub fn decodeHeads(g: *const GlobalDecode, s: Stream, a: DecodeArgs, h: usize) !void {
    try admit(g.max_rows, g.context, a);
    if (h == 0 or h > global_heads or h % global_kv_heads != 0) return error.GlobalShape;
    if (a.rows == 0) return;
    const origin = @min(a.kc, a.vc);
    plan(g.host, g.max_rows, a.pos, a.rows, @intCast((a.kc - origin) / 2), @intCast((a.vc - origin) / 2));
    try g.plan.uploadAsync(0, g.host, s.handle);
    const pg: ta.Pages = .{ .table = a.pages.table, .page = a.pages.page_elems, .tok = a.pages.tokens, .stride = 0 };
    try run(g, s, a.q, a.kc, a.vc, a.out, a.rows, h, pg, ta.TreeAttention.groups(a.pos + a.rows), a.scale);
}

/// A round's global-layer plan: row r its own tree stream at pos[r], K at `kc`, V `v_off` past it, `ng` key groups.
pub const Planned = struct { rows: usize, ng: usize, v_off: usize };

/// The rounds the plan takes: 1 to init's rows, each position within its context, V offsets of int64 elements.
pub fn admitRows(max_rows: usize, context: usize, pos: []const usize, v_off: usize) error{GlobalShape}!void {
    if (pos.len == 0 or pos.len > max_rows or v_off > std.math.maxInt(i63)) return error.GlobalShape;
    for (pos) |p| if (p >= context) return error.GlobalShape;
}

/// The groups a call's partials are strided by: those of its row with the most keys.
pub fn groupsOf(pos: []const usize) usize {
    var most: usize = 0;
    for (pos) |p| most = @max(most, p + 1);
    return ta.TreeAttention.groups(most);
}

/// tree_attention.zig's choice for a call: one-warp blocks (true) or four-warp blocks, by rows, groups and key heads.
pub fn oneWarp(g: *const GlobalDecode, rows: usize, ng: usize) bool {
    return g.tree.sms > 0 and rows * ng * global_kv_heads <= ta.slice_items;
}

/// A round's rows into the staged plan, each a one-row call's stream; `fixed`: the context's key groups, same bits.
pub fn planRows(g: *const GlobalDecode, s: Stream, pos: []const usize, fixed: bool) !Planned {
    const v_off = g.v_off orelse return error.PlanNotStaged;
    try admitRows(g.max_rows, g.context, pos, v_off);
    try g.plan.uploadAsync(4 * g.max_rows, commitRows(g.host, g.max_rows, pos), s.handle);
    return .{ .rows = pos.len, .ng = if (fixed) g.groups() else groupsOf(pos), .v_off = v_off };
}

/// One global layer's planned rows, `h` query heads over the 8 key heads: q [rows, h, 128] -> out [rows, h, 128].
pub fn runRows(g: *const GlobalDecode, s: Stream, p: Planned, q: u64, kc: u64, out: u64, h: usize, pages: wa.Pages, scale: f32) !void {
    if (h == 0 or h > global_heads or h % global_kv_heads != 0) return error.GlobalShape;
    if (pages.tokens == 0 or pages.tokens > std.math.maxInt(i32) or pages.page_elems == 0 or pages.page_elems > std.math.maxInt(i63) or pages.stride > std.math.maxInt(i32)) return error.GlobalShape;
    const pg: ta.Pages = .{ .table = pages.table, .page = pages.page_elems, .tok = pages.tokens, .stride = pages.stride };
    try run(g, s, q, kc, kc + 2 * p.v_off, out, p.rows, h, pg, p.ng, scale);
}

/// tree_attention over the uploaded plan: `n` rows, `ng` groups of partials, the pages `pg`.
fn run(g: *const GlobalDecode, s: Stream, q: u64, kc: u64, vc: u64, out: u64, n: usize, h: usize, pg: ta.Pages, ng: usize, scale: f32) !void {
    const m = g.max_rows;
    const base = g.plan.ptr;
    const depths = base + 20 * m; // also `paths`: depth 0 reads none
    const rows: ta.Rows = .{ .row_stream = base, .streams = base + 4 * m, .offs = base + 24 * m, .origin = @min(kc, vc), .paths = depths, .depths = depths, .pages = pg };
    const sc = ta.TreeAttention.scratch(g.scratch.ptr, n, h, global_kv_heads, ng, head_dim);
    // kn, vn: path keys, of which there are none (each row's own key is committed in the pages)
    try g.tree.run(s, q, kc, vc, rows, sc, out, n, h, global_kv_heads, head_dim, ng, scale);
}

test "Ripple's prompt grid: 64 rows a block, five head pairs of each of 8 key heads" {
    const t = std.testing;
    const w = sliding(80, 8, 1024, 3072, 0.08838835);
    for ([_][2]u32{ .{ 2048, 32 }, .{ 1000, 16 }, .{ 333, 6 }, .{ 1, 1 } }) |c| {
        const cfg = config(w, c[0]);
        try t.expectEqual(c[1], @as(u32, cfg.grid.x));
        try t.expectEqual(@as(u32, 40), @as(u32, cfg.grid.y));
        try t.expectEqual(@as(u32, 256), @as(u32, cfg.block.x));
        try t.expectEqual(@as(u32, 65536), cfg.shared);
    }
}

test "the ring takes the window and a 2,048-row chunk past its wrap; a longer chunk or a short ring is refused" {
    const t = std.testing;
    const w = sliding(80, 8, 1024, 3072, 0.08838835);
    try check(w, 8216, 2048);
    try check(w, 0, 3072); // no wrap in the call
    try t.expectError(error.WindowShape, check(w, 8216, 2050));
    try t.expectError(error.WindowShape, check(sliding(80, 8, 1024, 3000, 0.08838835), 8216, 2048));
    try check(global(80, 8, 0.08838835, null), 6000, 2048); // linear keys: no ring
    var every = global(80, 8, 0.08838835, null);
    every.ring = 4096;
    try t.expectError(error.WindowShape, check(every, 4000, 200)); // every key, from a ring that wraps
}

test "shapes the kernel does not take are refused" {
    const t = std.testing;
    try t.expectError(error.WindowShape, check(sliding(72, 8, 1024, 3072, 1), 0, 16)); // group 9
    var wide = sliding(80, 8, 1024, 3072, 1);
    wide.d = 256;
    try t.expectError(error.WindowShape, check(wide, 0, 16));
    var pg = sliding(80, 8, 1024, 3072, 1);
    pg.pg = .{ .table = 0, .page_elems = 2 * 512 * 8 * 128, .tokens = 512 };
    try t.expectError(error.WindowShape, check(pg, 0, 16)); // pages hold global layers' keys only
    try check(global(80, 8, 1, .{ .table = 0, .page_elems = 2 * 512 * 8 * 128, .tokens = 512 }), 6000, 2048);
}

test "global decode plan: each row its own stream, committed to its own key, no path keys" {
    const t = std.testing;
    var buf: [planBytes(4)]u8 = @splat(0xee);
    plan(&buf, 4, 510, 3, 0, 2048);
    const ints = std.mem.bytesAsSlice(i32, buf[0..96]);
    const offs = std.mem.bytesAsSlice(i64, buf[96..]);
    for (0..3) |r| {
        const ri: i32 = @intCast(r);
        try t.expectEqual(ri, ints[r]);
        try t.expectEqualSlices(i32, &.{ ri, 1, 511 + ri, 0 }, &[4]i32{ ints[4 + 4 * r], ints[5 + 4 * r], ints[6 + 4 * r], ints[7 + 4 * r] });
        try t.expectEqual(@as(i32, 0), ints[20 + r]);
        try t.expectEqual(@as(i64, 0), offs[2 * r]);
        try t.expectEqual(@as(i64, 2048), offs[2 * r + 1]);
    }
    try t.expectEqual(@as(i32, @bitCast(@as(u32, 0xeeeeeeee))), ints[3]); // row 3: past the call's rows, untouched
}

test "global decode takes 1 to 16 rows to the context, and pages of whole positions" {
    const t = std.testing;
    const pg: wa.Pages = .{ .table = 0, .page_elems = 2 * 512 * 8 * 128, .tokens = 512 };
    const at = struct {
        fn f(pos: usize, rows: usize, p: wa.Pages) DecodeArgs {
            return .{ .q = 0, .kc = 0, .vc = 0, .out = 0, .pos = pos, .rows = rows, .pages = p, .scale = 0.08838835 };
        }
    }.f;
    try admit(16, 131072, at(131056, 16, pg));
    try admit(16, 131072, at(0, 1, pg));
    try t.expectError(error.GlobalShape, admit(16, 131072, at(131057, 16, pg))); // past the context
    try t.expectError(error.GlobalShape, admit(16, 131072, at(200000, 1, pg)));
    try t.expectError(error.GlobalShape, admit(16, 131072, at(0, 17, pg)));
    var none = pg;
    none.tokens = 0;
    try t.expectError(error.GlobalShape, admit(16, 131072, at(0, 1, none)));
}

test "global decode rows: each row its own stream and call table row, committed to its own position" {
    const t = std.testing;
    var buf: [planBytes(4)]u8 = @splat(0xee);
    const pos = [_]usize{ 4095, 511, 3071 };
    for (pos, 0..) |p, r| planRow(&buf, 4, r, p + 1, 0, 4608);
    const ints = std.mem.bytesAsSlice(i32, buf[0..96]);
    const offs = std.mem.bytesAsSlice(i64, buf[96..]);
    for (pos, 0..) |p, r| {
        const ri: i32 = @intCast(r);
        try t.expectEqual(ri, ints[r]);
        try t.expectEqualSlices(i32, &.{ ri, 1, @as(i32, @intCast(p + 1)), 0 }, &[4]i32{ ints[4 + 4 * r], ints[5 + 4 * r], ints[6 + 4 * r], ints[7 + 4 * r] });
        try t.expectEqual(@as(i64, 4608), offs[2 * r + 1]);
    }
    var one: [planBytes(4)]u8 = @splat(0xee);
    plan(&one, 4, 510, 3, 0, 4608);
    var rows: [planBytes(4)]u8 = @splat(0xee);
    for (0..3) |r| planRow(&rows, 4, r, 510 + r + 1, 0, 4608);
    try t.expectEqualSlices(u8, &one, &rows); // a stream's consecutive rows: M1's plan
    try t.expectEqual(@as(usize, 8), groupsOf(&pos)); // 4,096 keys: 8 groups of 512
    try t.expectEqual(@as(usize, 9), groupsOf(&.{4096}));
}

test "staged plan: fixed fields once, a round's committed keys its only upload, B1's per-row plan for its rows" {
    const t = std.testing;
    var staged: [planBytes(4)]u8 = @splat(0xee);
    stagePlan(&staged, 4, 4608);
    const first = [_]usize{ 4095, 511, 3071 };
    _ = commitRows(&staged, 4, &first);
    const pos = [_]usize{ 7, 3072 }; // a later round of fewer rows: rows past it keep the last round's keys
    const up = commitRows(&staged, 4, &pos);
    try t.expectEqual(@as(usize, 2 * 16), up.len);
    try t.expectEqual(@intFromPtr(&staged) + 16, @intFromPtr(up.ptr)); // streams' rows, after row_stream [4]
    var b1: [planBytes(4)]u8 = @splat(0xee);
    for (0..4) |r| planRow(&b1, 4, r, 0, 0, 4608);
    for (pos, 0..) |p, r| planRow(&b1, 4, r, p + 1, 0, 4608);
    planRow(&b1, 4, 2, 3072, 0, 4608);
    try t.expectEqualSlices(u8, &b1, &staged);
    const ints = std.mem.bytesAsSlice(i32, staged[0..96]);
    try t.expectEqualSlices(i32, &.{ 1, 1, 3073, 0 }, &[4]i32{ ints[8], ints[9], ints[10], ints[11] });
}

test "a graph's key groups: the context's, at least any round's; their partials fit init's scratch" {
    const t = std.testing;
    try t.expectEqual(@as(usize, 16), ta.TreeAttention.groups(8192)); // the forward test's context
    try t.expectEqual(@as(usize, 64), ta.TreeAttention.groups(32768)); // the default context
    try t.expect(groupsOf(&.{ 8191, 0 }) <= ta.TreeAttention.groups(8192));
    const w = ta.TreeAttention.scratch(0, 48, global_heads, global_kv_heads, ta.TreeAttention.groups(8192), head_dim);
    try t.expectEqual(@as(u64, scratchBytes(48, 8192)), w.sl + ta.slice_items * 4 * 10 * 4); // the last slice sum's end
}

test "global decode rows: 1 to init's rows, each position within the context" {
    const t = std.testing;
    try admitRows(16, 131072, &.{ 131071, 0, 70000 }, 512 * 1024 * 256);
    try t.expectError(error.GlobalShape, admitRows(16, 131072, &.{131072}, 0));
    try t.expectError(error.GlobalShape, admitRows(2, 131072, &.{ 1, 2, 3 }, 0));
    try t.expectError(error.GlobalShape, admitRows(2, 131072, &.{}, 0));
}

test "global decode scratch: 16 rows' partials over 131,072 keys (256 groups), then the slices" {
    try std.testing.expectEqual(@as(usize, (16 * 80 * 256 + 1024 * 4 * 10) * 130 * 4), scratchBytes(16, 131072));
    try std.testing.expectEqual(@as(usize, 191_692_800), scratchBytes(16, 131072));
}
