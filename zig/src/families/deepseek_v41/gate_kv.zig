//! `tf-dsv41 kv MODE PACK REF OUT` (a process a rank): the KV pool's paged / split / compact / park checks.

const std = @import("std");
const cuda = @import("cuda");
const tp = @import("tp");
const model = @import("cuda_model.zig");
const block = @import("block.zig");
const Value = std.json.Value;

pub const Mode = enum { paged, split, compact, park };

fn sha(bytes: []const u8) [64]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
    return std.fmt.bytesToHex(d, .lower);
}

const env = @import("sys").env;

fn put(name: [*:0]const u8, value: []const u8) !void {
    var buf: [512]u8 = undefined;
    if (value.len >= buf.len) return error.Setenv;
    @memcpy(buf[0..value.len], value);
    buf[value.len] = 0;
    if (env.setenv(name, @ptrCast(&buf), 1) != 0) return error.Setenv;
}

pub fn main(gpa: std.mem.Allocator, io: std.Io, mode_name: []const u8, pack: []const u8, ref: []const u8, out: []const u8) !u8 {
    const mode = std.meta.stringToEnum(Mode, mode_name) orelse return error.BadMode;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cwd = std.Io.Dir.cwd();
    const tcfg = try tp.Config.fromEnv();
    const rank = tcfg.rank;
    try cwd.createDirPath(io, out);
    // the prompts: REF/p<N>/ (sorted), or REF itself
    var dirs: std.ArrayList([]const u8) = .empty;
    {
        var d = try cwd.openDir(io, ref, .{ .iterate = true });
        defer d.close(io);
        var it = d.iterate();
        while (try it.next(io)) |e| if (e.kind == .directory and e.name.len > 1 and e.name[0] == 'p' and std.ascii.isDigit(e.name[1]))
            try dirs.append(a, try std.fs.path.join(a, &.{ ref, e.name }));
        std.mem.sort([]const u8, dirs.items, {}, struct {
            fn lt(_: void, x: []const u8, y: []const u8) bool {
                return std.mem.lessThan(u8, x, y);
            }
        }.lt);
        if (dirs.items.len == 0) try dirs.append(a, ref);
    }
    const meta0 = try readJson(a, io, dirs.items[0], "ref.json");
    // the knobs this mode runs with (every rank alike), and the reference's
    try put("TF_DSV41_POOL_TOKENS", if (std.c.getenv("TF_DSV41_KV_CHECK_POOL_TOKENS")) |v| std.mem.span(v) else "16384");
    try put("TF_DSV41_BOOT_MEASURED", "0");
    const split = mode == .split or mode == .compact or (mode == .park and std.mem.eql(u8, envOr("TF_DSV41_KV_CHECK_PARK", "pool"), "split"));
    try put("TF_DSV41_KV_SPLIT", if (split) "1" else "0");
    try put("TF_DSV41_KV_SPLIT_COMPACT", if (mode == .compact) "force" else "0");
    if (mode == .park) try put("TF_DSV41_SESSION_DISK", try std.fmt.allocPrint(a, "{s}/disk-r{d}", .{ out, rank }));
    if (std.c.getenv("TF_DSV41_KV_CHECK_TOPP")) |v| try put("TF_DSV41_EXPERT_TOPP", std.mem.span(v));
    if (meta0.object.get("env")) |meta_env| if (meta_env.object.get("TF_DSV41_MHC_DEFER")) |v| if (v == .string and std.mem.eql(u8, v.string, "1")) try put("TF_DSV41_R1", "1");
    var lo: u32 = 0;
    var hi: u32 = 0;
    {
        var it = std.mem.tokenizeScalar(u8, meta0.object.get("layers").?.string, '-');
        lo = try std.fmt.parseInt(u32, it.next().?, 10);
        hi = try std.fmt.parseInt(u32, it.next().?, 10);
    }
    const layers = try a.alloc(u32, hi - lo + 1);
    for (layers, 0..) |*l, i| l.* = lo + @as(u32, @intCast(i));
    const t0 = std.Io.Clock.awake.now(io);
    const m = try model.Model.open(gpa, io, pack, dirs.items[0], .{ .drafts = false, .own_prefill = true, .layers = layers });
    defer m.close();
    const load_s = secs(io, t0);
    const ss = m.sessions orelse return error.NoPool;
    std.debug.print("rank {d}: kv {s}, {d} layers, load {d:.1} s, pool {d} tokens{s}\n", .{ rank, mode_name, layers.len, load_s, m.f.kv.?.opts.tokens, if (split) ", split KV" else "" });
    var log: std.Io.Writer.Allocating = .init(gpa);
    defer log.deinit();
    var all_ok = true;
    for (dirs.items) |dir| {
        const ok = try onePrompt(gpa, io, a, m, mode, dir, rank, &log.writer);
        all_ok = all_ok and ok;
        try ss.reset();
    }
    if (m.f.kv) |kx| try log.writer.print("{{\"rank\": {d}, \"exchanges\": {{\"dense\": {d}, \"unions\": {d}, \"union_rows_max\": {d}, \"bytes\": {d}}}}}\n", .{ rank, kx.stats.dense, kx.stats.unions, kx.stats.union_rows_max, kx.stats.bytes });
    try cwd.writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/kv-rank{d}.jsonl", .{ out, rank }), .data = log.written() });
    std.debug.print("{s} kv {s} rank {d}: {d} prompt(s)\n", .{ if (all_ok) "PASS" else "FAIL", mode_name, rank, dirs.items.len });
    return if (all_ok) 0 else 1;
}

fn envOr(name: [*:0]const u8, d: []const u8) []const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else d;
}

fn secs(io: std.Io, t0: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e9;
}

fn readJson(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) !Value {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, name }), a, .limited(1 << 28));
    return std.json.parseFromSliceLeaky(Value, a, text, .{});
}

/// One prompt: prefill, the state against the dump, the decode steps (park: the two round trips inside them).
fn onePrompt(gpa: std.mem.Allocator, io: std.Io, a: std.mem.Allocator, m: *model.Model, mode: Mode, dir: []const u8, rank: u32, log: *std.Io.Writer) !bool {
    const f = &m.f;
    const r = &m.runner;
    const ss = m.sessions.?;
    const meta = try readJson(a, io, dir, "ref.json");
    const mine = try readJson(a, io, dir, try std.fmt.allocPrint(a, "rank{d}.json", .{rank}));
    const sdir = try std.fmt.allocPrint(a, "{s}/rank{d}/state", .{ dir, rank });
    const st = try readJson(a, io, sdir, "state.json");
    // a fresh slot: bounded roles (SWA rings, carries) zeroed so rows a short prompt never writes compare equal
    try zeroBounded(a, io, m, st, sdir);
    const pids = meta.object.get("prompt_ids").?.array.items;
    const prompt = try a.alloc(u32, pids.len);
    for (prompt, pids) |*d, v| d.* = @intCast(v.integer);
    const want_tokens = meta.object.get("tokens").?.array.items;
    const want_logits = mine.object.get("logits_sha256").?.array.items;
    // the prompt through our prefill from the empty slot
    const tp0 = std.Io.Clock.awake.now(io);
    try f.prefill(prompt);
    const seg: usize = @intCast(f.opts.prefill_rows);
    const last: usize = if (prompt.len % seg == 0) seg else prompt.len % seg;
    const picks = try a.alloc(u32, last);
    try f.greedy(picks);
    const prefill_s = secs(io, tp0);
    const first = picks[last - 1];
    // the slot against the reference's dump
    const cmp = try compareState(gpa, a, io, m, st, sdir, rank);
    const pos_ok = f.slot.pos == @as(u64, @intCast(st.object.get("pos").?.integer));
    const tail = st.object.get("tail").?.array.items;
    var tail_ok = f.slot.tail_len == tail.len;
    for (tail, 0..) |t, i| if (i < f.slot.tail_len and f.slot.tail[i] != @as(u32, @intCast(t.integer))) {
        tail_ok = false;
    };
    // the decode steps (park: a RAM round trip a third in, an NVMe one two thirds in)
    const steps = want_tokens.len - 1;
    const V: usize = m.cfg.vocab / f.comm.world();
    const host_logits = try gpa.alloc(u8, 4 * V);
    defer gpa.free(host_logits);
    var tok: u32 = @intCast(want_tokens[0].integer);
    var tokens_equal: usize = 0;
    var logits_equal: usize = 0;
    var round_trips: [2]?f64 = .{ null, null };
    for (0..steps) |step| {
        if (mode == .park and (step == steps / 3 or step == 2 * steps / 3)) {
            const disk = step != steps / 3;
            const tr = std.Io.Clock.awake.now(io);
            // the slot's ids: an entry parked to NVMe restores with them (its tokens leave the index with RAM)
            const held = try gpa.alloc(u32, ss.hist[ss.cur()].items.len);
            defer gpa.free(held);
            for (ss.hist[ss.cur()].items, held) |t, *y| y.* = @intCast(t);
            const id = (try ss.save()) orelse return error.DuplicateEntry;
            if (disk) {
                if (f.slot.pos < 1024) {
                    try log.print("{{\"park\": \"skipped: {d} positions, under the tier's 1,024\"}}\n", .{f.slot.pos});
                } else _ = try ss.parkAll();
            }
            if (!try ss.restore(id, if (ss.store.entry(id).ram) null else held)) return error.RestoreFailed;
            round_trips[@intFromBool(disk)] = secs(io, tr);
        }
        const ids = [_]u32{tok};
        try f.window(&ids);
        try r.stream.synchronize();
        try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = r.addressOf("w.logits").?, .len = host_logits.len }, 0, host_logits);
        const lok = std.mem.eql(u8, &sha(host_logits), want_logits[step].string);
        logits_equal += @intFromBool(lok);
        var choice: [1]u32 = undefined;
        try f.greedy(&choice);
        tok = choice[0];
        const tok_ok = tok == @as(u32, @intCast(want_tokens[step + 1].integer));
        tokens_equal += @intFromBool(tok_ok);
        try f.keep(0);
        if (!tok_ok) break;
    }
    const want_first: u32 = @intCast(want_tokens[0].integer);
    const ok = cmp.equal == cmp.roles and pos_ok and tail_ok and first == want_first and tokens_equal == steps and logits_equal == steps;
    try log.print("{{\"prompt\": {d}, \"prefill_s\": {d:.2}, \"roles_equal\": {d}, \"roles\": {d}, \"first_differing_role\": \"{s}\", \"pos_equal\": {}, \"tail_equal\": {}, \"first_token\": {d}, \"want\": {d}, \"steps\": {d}, \"tokens_equal\": {d}, \"logits_equal\": {d}, \"ram_round_trip_s\": {?d:.3}, \"nvme_round_trip_s\": {?d:.3}, \"pass\": {}}}\n", .{ prompt.len, prefill_s, cmp.equal, cmp.roles, cmp.first_bad, pos_ok, tail_ok, first, want_first, steps, tokens_equal, logits_equal, round_trips[0], round_trips[1], ok });
    std.debug.print("{s} kv {t} rank {d}: {d} rows, state roles {d}/{d} (first differing: {s}), position {}, Engram tail {}, first token {d} vs {d}, tokens {d}/{d}, logits {d}/{d}{s}\n", .{ if (ok) "PASS" else "FAIL", mode, rank, prompt.len, cmp.equal, cmp.roles, cmp.first_bad, pos_ok, tail_ok, first, want_first, tokens_equal, steps, logits_equal, steps, if (mode == .park) ", through a RAM and an NVMe round trip" else "" });
    return ok;
}

const Cmp = struct { roles: usize = 0, equal: usize = 0, first_bad: []const u8 = "" };

/// Every bounded role the dump names (not the pool's `s.kv.*`) set to zero bytes, the dump file's length each.
fn zeroBounded(a: std.mem.Allocator, io: std.Io, m: *model.Model, st: Value, sdir: []const u8) !void {
    const r = &m.runner;
    var it = st.object.get("roles").?.object.iterator();
    while (it.next()) |e| {
        if (std.mem.startsWith(u8, e.key_ptr.*, "s.kv.")) continue;
        const p = r.addressOf(e.key_ptr.*) orelse return error.StateRole;
        const len = (try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ sdir, e.value_ptr.string }), a, .limited(1 << 32))).len;
        try (cuda.DeviceBuffer{ .d = r.d, .ptr = p, .len = len }).fill8(0, r.stream.handle);
    }
    try r.stream.synchronize();
}

/// The dump's roles against ours: the bounded roles byte for byte, the pool's logical rows through the slot's tables.
fn compareState(gpa: std.mem.Allocator, a: std.mem.Allocator, io: std.Io, m: *model.Model, st: Value, sdir: []const u8, rank: u32) !Cmp {
    const cwd = std.Io.Dir.cwd();
    const r = &m.runner;
    const kx = m.f.kv.?;
    try r.stream.synchronize();
    var c: Cmp = .{};
    var it = st.object.get("roles").?.object.iterator();
    while (it.next()) |e| {
        const role = e.key_ptr.*;
        const want = try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ sdir, e.value_ptr.string }), a, .limited(1 << 32));
        var eq: bool = undefined;
        if (std.mem.startsWith(u8, role, "s.kv.")) {
            if (!std.mem.endsWith(u8, role, ".rows")) continue; // physical placement differs by design
            const ik = std.mem.startsWith(u8, role, "s.kv.ik.L");
            const L = try std.fmt.parseInt(u32, role[if (ik) "s.kv.ik.L".len else "s.kv.comp.L".len .. role.len - ".rows".len], 10);
            const fam = if (ik) kx.layout.indexOf(L).? else kx.layout.compOf(L).?;
            const f = kx.layout.families()[fam];
            const rb: usize = f.row_bytes;
            const per = f.pageRows(kx.pool.page);
            const psh: u5 = @intCast(std.math.log2_int(u32, per));
            // the logical rows the dump holds: every row (index keys, a replicated comp), or the owned runs (split)
            var rows: std.ArrayList(u32) = .empty;
            if (!ik and f.split) {
                const j = try readJson(a, io, sdir, try std.fmt.allocPrint(a, "{s}.json", .{role}));
                for (j.object.get("comp_logical").?.array.items) |run| {
                    const x = run.array.items;
                    var t: u32 = @intCast(x[0].integer);
                    while (t < @as(u32, @intCast(x[1].integer))) : (t += 1) try rows.append(a, t);
                }
            } else for (0..want.len / rb) |t| try rows.append(a, @intCast(t));
            if (rows.items.len * rb != want.len) {
                eq = false;
            } else {
                const tensor = kx.dev.tensors[fam];
                const got = try gpa.alloc(u8, tensor.len);
                defer gpa.free(got);
                try cuda.DeviceBuffer.download(tensor, 0, got);
                eq = true;
                for (rows.items, 0..) |t, i| {
                    const page = t >> psh;
                    const tab = if (f.split) kx.slot.localTableAt(page) else kx.slot.tableAt(page);
                    const phys = (@as(usize, tab) << psh) | (t & (per - 1));
                    if (!std.mem.eql(u8, got[phys * rb ..][0..rb], want[i * rb ..][0..rb])) {
                        eq = false;
                        break;
                    }
                }
            }
        } else {
            const p = r.addressOf(role) orelse return error.StateRole;
            const got = try gpa.alloc(u8, want.len);
            defer gpa.free(got);
            try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = p, .len = got.len }, 0, got);
            eq = std.mem.eql(u8, got, want);
        }
        c.roles += 1;
        if (eq) c.equal += 1 else if (c.first_bad.len == 0) c.first_bad = role;
    }
    _ = rank;
    return c;
}
