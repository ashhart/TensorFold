//! SenseNova-Loop-9B HF PyTorch oracle trace/generation parity verifier and LinearGpu microbench.
const std = @import("std");
const mtl = @import("metal");
const q = @import("qwen27");
const core = @import("core");

fn bf16ToF64(bits: u16) f64 {
    return @floatCast(@as(f32, @bitCast(@as(u32, bits) << 16)));
}

const Stats = struct { cos: f64, max_abs: f64 };

fn compareTensor(got: []const u16, st_bytes: []const u8, st_hdr: std.json.ObjectMap, st_base: usize, name: []const u8) !Stats {
    const ent = (st_hdr.get(name) orelse return error.MissingOracleTensor).object;
    const dtype = ent.get("dtype").?.string;
    const offs = ent.get("data_offsets").?.array.items;
    const raw = st_bytes[st_base + @as(usize, @intCast(offs[0].integer)) .. st_base + @as(usize, @intCast(offs[1].integer))];
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    var max_abs: f64 = 0;
    if (std.mem.eql(u8, dtype, "BF16")) {
        if (raw.len != got.len * 2) return error.OracleShapeMismatch;
        for (got, 0..) |g, i| {
            const a = bf16ToF64(g);
            const b = bf16ToF64(std.mem.readInt(u16, raw[i * 2 ..][0..2], .little));
            dot += a * b;
            na += a * a;
            nb += b * b;
            max_abs = @max(max_abs, @abs(a - b));
        }
    } else if (std.mem.eql(u8, dtype, "F32")) {
        if (raw.len != got.len * 4) return error.OracleShapeMismatch;
        for (got, 0..) |g, i| {
            const a = bf16ToF64(g);
            const b: f64 = @floatCast(@as(f32, @bitCast(std.mem.readInt(u32, raw[i * 4 ..][0..4], .little))));
            dot += a * b;
            na += a * a;
            nb += b * b;
            max_abs = @max(max_abs, @abs(a - b));
        }
    } else return error.BadOracleDType;
    const denom = @sqrt(na) * @sqrt(nb);
    return .{ .cos = if (denom > 0) dot / denom else 0, .max_abs = max_abs };
}

fn oracleLogitsRow(st_bytes: []const u8, st_hdr: std.json.ObjectMap, st_base: usize, name: []const u8, pos: usize, vocab: usize, out: []u16) !void {
    const ent = st_hdr.get(name).?.object;
    const offs = ent.get("data_offsets").?.array.items;
    const raw = st_bytes[st_base + @as(usize, @intCast(offs[0].integer)) .. st_base + @as(usize, @intCast(offs[1].integer))];
    const base = pos * vocab * 2;
    for (out, 0..) |*d, i| d.* = std.mem.readInt(u16, raw[base + i * 2 ..][0..2], .little);
}

fn top16Overlap(a_row: []const u16, b_row: []const u16, alloc: std.mem.Allocator) !usize {
    const n = a_row.len;
    const a_idx = try alloc.alloc(u32, n);
    defer alloc.free(a_idx);
    const b_idx = try alloc.alloc(u32, n);
    defer alloc.free(b_idx);
    for (0..n) |i| {
        a_idx[i] = @intCast(i);
        b_idx[i] = @intCast(i);
    }
    const Ctx = struct {
        vals: []const u16,
        pub fn lessThan(ctx: @This(), x: u32, y: u32) bool {
            return bf16ToF64(ctx.vals[x]) > bf16ToF64(ctx.vals[y]);
        }
    };
    std.mem.sort(u32, a_idx, Ctx{ .vals = a_row }, Ctx.lessThan);
    std.mem.sort(u32, b_idx, Ctx{ .vals = b_row }, Ctx.lessThan);
    var hits: usize = 0;
    for (a_idx[0..16]) |id| if (std.mem.indexOfScalar(u32, b_idx[0..16], id) != null) {
        hits += 1;
    };
    return hits;
}

fn ptrU16(r: q.projection.Ref, n: usize) []const u16 {
    const p: [*]const u16 = @ptrCast(@alignCast(r.buffer.contents() + r.offset));
    return p[0..n];
}

fn verifyTrace(a: std.mem.Allocator, gpa: std.mem.Allocator, m: *q.model.Model, runner: *q.decode_round.Runner, st_bytes: []const u8, st_hdr: std.json.ObjectMap, st_base: usize, trace_ids: []const u32, is_awq: bool) !void {
    const hidden = m.config.hidden;
    const vocab = m.config.vocab;
    const rows = trace_ids.len;
    const emb = m.weights.embed_bf16 orelse return error.MissingEmbedBf16;
    const emb_src = ptrU16(.{ .buffer = emb.buffer, .offset = emb.offset }, vocab * hidden);
    const emb_rows = try a.alloc(u16, rows * hidden);
    for (trace_ids, 0..) |id, ri| @memcpy(emb_rows[ri * hidden ..][0..hidden], emb_src[id * hidden ..][0..hidden]);

    try runner.taps.use(m.device, &.{ 3, 27, 51, 55 });
    runner.capture_taps = true;
    var round = try q.round_plan.Round.init(gpa, &.{.{ .slot = 0, .start = 0, .capacity = runner.capacity, .ids = trace_ids }}, @intCast(m.config.conv_kernel), runner.slots, @intCast(m.shard.global.vocab));
    defer round.deinit();
    m.kernels.prompt = true;
    try runner.advance(&round, .all);
    m.kernels.prompt = false;

    const checks = [_]struct { label: []const u8, got: []const u16, min_cos: f64 }{
        .{ .label = "embed_tokens", .got = emb_rows, .min_cos = 0.99999 },
        .{ .label = "exec3_out", .got = ptrU16(try runner.taps.plane(0), rows * hidden), .min_cos = 0.9995 },
        .{ .label = "exec27_h_after_collapse", .got = ptrU16(try runner.taps.plane(1), rows * hidden), .min_cos = 0.9995 },
        .{ .label = "exec51_h_after_collapse", .got = ptrU16(try runner.taps.plane(2), rows * hidden), .min_cos = 0.999 },
        .{ .label = "exec55_out", .got = ptrU16(try runner.taps.plane(3), rows * hidden), .min_cos = 0.999 },
        .{ .label = "final_norm", .got = ptrU16(m.frame.get(.input), rows * hidden), .min_cos = 0.999 },
        .{ .label = "logits_bf16", .got = ptrU16(m.frame.get(.logits), rows * vocab), .min_cos = if (is_awq) 0.9985 else 0.999 },
    };
    for (checks) |ck| {
        const st = try compareTensor(ck.got, st_bytes, st_hdr, st_base, ck.label);
        std.debug.print("TRACE {s:<24} cos={d:.7} max_abs={d:.5} min={d:.5}\n", .{ ck.label, st.cos, st.max_abs, ck.min_cos });
        if (st.cos < ck.min_cos or !std.math.isFinite(st.cos)) return error.OracleTraceCosineTooLow;
    }

    const metal_logits = ptrU16(m.frame.get(.logits), rows * vocab);
    const o_row = try a.alloc(u16, vocab);
    for (0..rows) |pos| {
        const m_row = metal_logits[pos * vocab ..][0..vocab];
        try oracleLogitsRow(st_bytes, st_hdr, st_base, "logits_bf16", pos, vocab, o_row);
        const m_top = try q.session.argmax(m_row);
        const o_top = try q.session.argmax(o_row);
        std.debug.print("TOP1 pos={d}: metal={d} oracle={d} match={}\n", .{ pos, m_top, o_top, m_top == o_top });
        if (m_top != o_top) return error.Top1Mismatch;
    }
    try oracleLogitsRow(st_bytes, st_hdr, st_base, "logits_bf16", rows - 1, vocab, o_row);
    const ov = try top16Overlap(metal_logits[(rows - 1) * vocab ..][0..vocab], o_row, a);
    std.debug.print("TOP16 overlap pos={d}: {d}/16\n", .{ rows - 1, ov });
    if (ov < 12) return error.Top16OverlapTooLow;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    if (args.len == 2 and std.mem.eql(u8, args[1], "--microbench")) {
        const dev = try mtl.Device.init();
        defer dev.deinit();
        const queue = try dev.queue();
        defer queue.deinit();
        return core.sensenova_loop.linear_gpu.LinearGpu.bench(gpa, dev, queue);
    }
    var model_dir: ?[]const u8 = null;
    var json_path: ?[]const u8 = null;
    var st_path: ?[]const u8 = null;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (i + 1 >= args.len) return error.MissingValue;
        if (std.mem.eql(u8, args[i], "--model")) model_dir = args[i + 1] else if (std.mem.eql(u8, args[i], "--oracle-json")) json_path = args[i + 1] else if (std.mem.eql(u8, args[i], "--oracle-st")) st_path = args[i + 1] else return error.UnknownOption;
        i += 1;
    }
    const dir = model_dir orelse return error.MissingModel;
    const m = try q.model.Model.load(gpa, io, dir, 128, null);
    defer m.deinit();
    const is_awq = m.weights.head.bits == 4;
    const oj_raw = try std.Io.Dir.cwd().readFileAlloc(io, json_path orelse return error.MissingOracleJson, a, .limited(32 << 20));
    const oj = try std.json.parseFromSlice(std.json.Value, a, oj_raw, .{});
    const st_bytes = try std.Io.Dir.cwd().readFileAlloc(io, st_path orelse return error.MissingOracleSt, a, .limited(64 << 20));
    const hlen: usize = @intCast(std.mem.readInt(u64, st_bytes[0..8], .little));
    const st_hdr = try std.json.parseFromSlice(std.json.Value, a, st_bytes[8 .. 8 + hlen], .{});
    const trace_ids: []const u32 = if (oj.value.object.get("trace_prompt_ids")) |v| blk: {
        const t_arr = v.array.items;
        const buf = try a.alloc(u32, t_arr.len);
        for (t_arr, buf) |x, *d| d.* = @intCast(x.integer);
        break :blk buf;
    } else &.{ 9707, 11, 1268, 527 };

    var runner = try q.decode_round.Runner.init(gpa, m, 1, 512);
    defer runner.deinit();
    try verifyTrace(a, gpa, m, &runner, st_bytes, st_hdr.value.object, 8 + hlen, trace_ids, is_awq);

    runner.capture_taps = false;
    const gen_val = oj.value.object.get("generations_greedy_64") orelse oj.value.object.get("generations_greedy") orelse return error.MissingGenerations;
    var it = gen_val.object.iterator();
    const s = q.session.Session{ .runner = &runner };
    while (it.next()) |kv| {
        if (kv.value_ptr.* != .object) continue;
        const p_items = kv.value_ptr.object.get("input_ids").?.array.items;
        const g_items = kv.value_ptr.object.get("generated_ids").?.array.items;
        const pids = try a.alloc(u32, p_items.len);
        const gids = try a.alloc(u32, g_items.len);
        for (p_items, pids) |v, *d| d.* = @intCast(v.integer);
        for (g_items, gids) |v, *d| d.* = @intCast(v.integer);
        try runner.reset(0);
        try s.prefill(pids, 128);
        var match: usize = 0;
        var exact = true;
        var first_got: u32 = 0;
        var first_exp: u32 = 0;
        for (gids, 0..) |exp, step| {
            const tok = try s.greedy();
            if (tok == exp and exact) match += 1 else if (exact) {
                exact = false;
                first_got = tok;
                first_exp = exp;
            }
            if (m.config.isEos(tok) or step + 1 == gids.len) break;
            try s.step(tok);
        }
        std.debug.print("GREEDY {s}: match={d}/{d} exact={} first_diff_pos={d} got={d} exp={d}\n", .{ kv.key_ptr.*, match, gids.len, exact, match, first_got, first_exp });
        if (match == 0) return error.GreedyDivergedImmediately;
    }
}
