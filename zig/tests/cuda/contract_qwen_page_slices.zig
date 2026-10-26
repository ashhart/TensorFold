//! A paged prompt chunk's attention most_verify rows a launch (cuda_forward.zig) == one launch == contiguous kernels.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const pages = @import("contract_qwen_pages.zig");

const hd = pages.hd;
const q_heads = pages.q_heads;
const kv_heads = pages.kv_heads;
const max_chunks = 64;
const most_verify = 128; // cuda_round.zig: the rows the engine's partials hold

/// Two streams' rows (the second 0: a prompt chunk's one stream), cached positions, which go through pages.
const Case = struct { rows: [2]usize, lens: [2]usize, ptok: usize, paged: [2]bool };
const cases = [_]Case{
    // a prompt chunk on a pool lane: one paged stream
    .{ .rows = .{ 127, 0 }, .lens = .{ 0, 0 }, .ptok = 256, .paged = .{ true, false } },
    .{ .rows = .{ 128, 0 }, .lens = .{ 4096, 0 }, .ptok = 256, .paged = .{ true, false } },
    .{ .rows = .{ 129, 0 }, .lens = .{ 0, 0 }, .ptok = 256, .paged = .{ true, false } },
    .{ .rows = .{ 255, 0 }, .lens = .{ 300, 0 }, .ptok = 48, .paged = .{ true, false } },
    .{ .rows = .{ 256, 0 }, .lens = .{ 0, 0 }, .ptok = 256, .paged = .{ true, false } },
    .{ .rows = .{ 257, 0 }, .lens = .{ 4096, 0 }, .ptok = 256, .paged = .{ true, false } },
    .{ .rows = .{ 4096, 0 }, .lens = .{ 0, 0 }, .ptok = 256, .paged = .{ true, false } },
    .{ .rows = .{ 4096, 0 }, .lens = .{ 4096, 0 }, .ptok = 256, .paged = .{ true, false } },
    // two streams, one on pages and one contiguous, a slice crossing from one to the other
    .{ .rows = .{ 64, 63 }, .lens = .{ 100, 33 }, .ptok = 256, .paged = .{ true, false } },
    .{ .rows = .{ 64, 64 }, .lens = .{ 100, 33 }, .ptok = 48, .paged = .{ false, true } },
    .{ .rows = .{ 65, 64 }, .lens = .{ 700, 0 }, .ptok = 256, .paged = .{ true, false } },
    .{ .rows = .{ 128, 127 }, .lens = .{ 0, 5000 }, .ptok = 256, .paged = .{ false, true } },
    .{ .rows = .{ 128, 128 }, .lens = .{ 129, 1000 }, .ptok = 128, .paged = .{ true, false } },
    .{ .rows = .{ 129, 128 }, .lens = .{ 31, 4095 }, .ptok = 48, .paged = .{ false, true } },
    .{ .rows = .{ 200, 57 }, .lens = .{ 31, 4095 }, .ptok = 256, .paged = .{ true, true } },
    .{ .rows = .{ 2048, 2048 }, .lens = .{ 100, 3000 }, .ptok = 256, .paged = .{ true, false } },
};

const Fns = struct { contiguous: cuda.Function, paged: cuda.Function, merge: cuda.Function };
const Layout = enum { contiguous, paged };

/// Both streams' layer-1 attention rows: `slice` rows a partial and merge (0: all in one), partials of that many.
fn attend(gpu: check.Gpu, s: cuda.Stream, fns: Fns, f: pages.Fixture, c: Case, layout: Layout, slice: usize, a: std.mem.Allocator) ![]u8 {
    return attendRows(gpu, s, fns, f, c, layout, slice, true, a);
}

/// `attend`; `shift` false (the control) leaves the row tables at row 0 while a slice's queries and outputs move.
fn attendRows(gpu: check.Gpu, s: cuda.Stream, fns: Fns, f: pages.Fixture, c: Case, layout: Layout, slice: usize, shift: bool, a: std.mem.Allocator) ![]u8 {
    var rig = rig_.Rig{ .gpu = gpu, .s = s };
    defer rig.deinit();
    var prng = std.Random.DefaultPrng.init(3);
    const m = c.rows[0] + c.rows[1];
    const nseg: usize = if (c.rows[1] == 0) 1 else 2;
    const geo: pages.Geo = .{ .hid = 0, .nk = 0, .nv = 0, .qh = q_heads, .kvh = kv_heads, .nl = 0 };
    var segs: [2]pages.Seg = undefined;
    const arena = if (layout == .paged) try rig.input(try f.arena(a, prng.random(), .{ f.ids[0], f.ids[1] })) else 0;
    var row0: usize = 0;
    for (0..nseg) |st| {
        const on_pages = layout == .paged and c.paged[st];
        segs[st] = .{ .kv = if (on_pages) arena else try rig.input(f.kv[st]), .cap = @intCast(f.cap), .len = @intCast(c.lens[st]), .row0 = @intCast(row0), .rows = @intCast(c.rows[st]) };
        if (on_pages) {
            segs[st].pt = try rig.input(std.mem.sliceAsBytes(f.ids[st]));
            segs[st].ps = if (f.format == .fp8) f.pageBytes() else f.pageBytes() / 2;
            segs[st].ptok = @intCast(c.ptok);
        }
        row0 += c.rows[st];
    }
    const rowseg = try a.alloc(i32, m);
    const rowanc = try a.alloc(pages.Ancestor, m);
    for (0..m) |r| {
        const st: usize = @intFromBool(r >= c.rows[0]);
        rowseg[r] = @intCast(st); // absolute: the segment table is not shifted
        rowanc[r] = .{ .bits = 0, .depth = @intCast(r - st * c.rows[0]), .chain = 1 };
    }
    // the engine's params: word 2 the round's rows, which a slice's own row index stays below
    var params: [8]i32 = @splat(0);
    params[2] = @intCast(m);
    params[3] = @intCast(nseg);
    const segs_d = try rig.input(std.mem.sliceAsBytes(segs[0..nseg]));
    const rowseg_d = try rig.input(std.mem.sliceAsBytes(rowseg));
    const rowanc_d = try rig.input(std.mem.sliceAsBytes(rowanc));
    const params_d = try rig.input(std.mem.sliceAsBytes(&params));
    const qn = try rig.input(std.mem.sliceAsBytes(f.q[0 .. m * q_heads * hd]));
    const held = if (slice == 0) m else slice; // the partials' rows; the rig's guards catch a write past them
    const pm = try rig.scratch(held * q_heads * max_chunks * 4);
    const pl = try rig.scratch(held * q_heads * max_chunks * 4);
    const po = try rig.scratch(held * q_heads * max_chunks * hd * 4);
    const out = try rig.output(m * q_heads * hd * 2);
    const before = try a.dupe(u8, try rig.inputBytes());
    const partial = if (layout == .paged) fns.paged else fns.contiguous;
    const layer: c_int = 1;
    const row_bytes = q_heads * hd * 2;
    var r: usize = 0;
    while (r < m) : (r += held) {
        const n = @min(held, m - r);
        const tr = if (shift) r else 0;
        var args: cuda.Args = .{};
        inline for (.{ geo, qn + r * row_bytes, segs_d, rowseg_d + tr * 4, rowanc_d + tr * @sizeOf(pages.Ancestor), layer, params_d, @as(f32, 1.0 / 16.0), pm, pl, po }) |x| args.add(x);
        try cuda.launch.launch(partial, .{ .grid = .{ .x = max_chunks, .y = kv_heads, .z = @intCast(n) }, .block = .{ .x = 32 * (q_heads / kv_heads) } }, s, &args);
        args = .{};
        inline for (.{ geo, pm, pl, po, segs_d, rowseg_d + tr * 4, rowanc_d + tr * @sizeOf(pages.Ancestor), params_d, out + r * row_bytes }) |x| args.add(x);
        try cuda.launch.launch(fns.merge, .{ .grid = .{ .x = q_heads, .y = @intCast(n) }, .block = .{ .x = 32 } }, s, &args);
    }
    const got = try a.dupe(u8, try rig.bytes(out, m * row_bytes));
    try check.expect(try rig.intact() and !rig.overlap() and std.mem.eql(u8, before, try rig.inputBytes()), "qwen35 page slices: a launch changed an input or guard (rows {any}, slice {d})", .{ c.rows, slice });
    return got;
}

/// Every case, or (`small`) those of 129 to 257 rows: the compute-sanitizer run's.
pub fn run(gpu: check.Gpu, small: bool) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.qwen35);
    defer mod.unload();
    var kv8 = try cuda.Module.load(gpu.d, cuda.kernels.qwen35_kv8);
    defer kv8.unload();
    const merge = try mod.function("tf_q35_attn_merge");
    var red = false;
    for (std.enums.values(pages.Format)) |format| {
        const fns: Fns = switch (format) {
            .bf16 => .{ .contiguous = try mod.function("tf_q35_attn_partial_contiguous"), .paged = try mod.function("tf_q35_attn_partial_paged"), .merge = merge },
            .fp8 => .{ .contiguous = try kv8.function("tf_q35_attn_partial8_contiguous"), .paged = try kv8.function("tf_q35_attn_partial8_paged"), .merge = merge },
        };
        for (cases, 0..) |c, i| {
            const m = c.rows[0] + c.rows[1];
            if (small and (m < 129 or m > 257)) continue;
            var arena = std.heap.ArenaAllocator.init(gpu.gpa);
            defer arena.deinit();
            const a = arena.allocator();
            const f = try pages.Fixture.make(a, .{ .lens = c.lens, .rows = @max(c.rows[0], c.rows[1]), .ptok = c.ptok }, format, 0x51ce + i);
            const whole = try attend(gpu, s, fns, f, c, .contiguous, 0, a);
            const sliced = try attend(gpu, s, fns, f, c, .paged, most_verify, a);
            try check.sameBytes("qwen35 page slices: paged, most_verify rows a launch, against contiguous in one", sliced, whole);
            try check.sameBytes("qwen35 page slices: paged in one launch against contiguous", try attend(gpu, s, fns, f, c, .paged, 0, a), whole);
            try check.sameBytes("qwen35 page slices: contiguous, most_verify rows a launch, against one", try attend(gpu, s, fns, f, c, .contiguous, most_verify, a), whole);
            // red control: slices whose queries and outputs move but whose row tables stay at row 0
            if (!small and m > most_verify and !red) red = !std.mem.eql(u8, try attendRows(gpu, s, fns, f, c, .paged, most_verify, false, a), whole);
            std.debug.print("PASS qwen35 page slices {t}: rows {any}, lens {any}, {d}-position pages, paged {any}: sliced == one launch == contiguous, guards\n", .{ format, c.rows, c.lens, c.ptok, c.paged });
        }
    }
    if (!small) try check.expect(red, "qwen35 page slices: the unshifted-table control never fired", .{});
    if (!small) std.debug.print("PASS qwen35 page slices: control RED as planted (row tables not shifted with the slice)\n", .{});
}
