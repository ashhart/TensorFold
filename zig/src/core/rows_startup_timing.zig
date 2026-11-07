//! tf-startup-states: seeds synthetic kept states in a store's on-disk format, then times that store's open.
const std = @import("std");
const builtin = @import("builtin");
const platform = @import("sys");
const rows = @import("rows_store.zig");
const imprint = @import("prompt_imprint.zig");
const corpus = @import("rows_startup_corpus.zig");
const Peak = @import("rows_peak_allocator.zig").Peak;
const dirs = @import("lanes").learned_dirs;
const Allocator = std.mem.Allocator;

const usage = "usage: tf-startup-states rows|imprint seed|open|phases|summary|clean COUNT\n";
const Store = enum { rows, imprint };
const identity: u64 = 0x27b;
const cap: u64 = 1 << 50;
const nowNs = rows.write.nowNs;

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 4) return fail();
    const store = std.meta.stringToEnum(Store, args[1]) orelse return fail();
    const count = std.fmt.parseInt(usize, args[3], 10) catch return fail();
    const tmp = if (platform.c.getenv("TMPDIR")) |p| std.mem.trimEnd(u8, std.mem.span(p), "/") else "/tmp";
    const root = try std.fmt.allocPrintSentinel(init.gpa, "{s}/tf-startup-states-{t}-{d}", .{ tmp, store, count }, 0);
    defer init.gpa.free(root);
    const runs = try std.fmt.allocPrintSentinel(init.gpa, "{s}.runs", .{root}, 0);
    defer init.gpa.free(runs);
    const a = init.gpa;
    const mode = args[2];
    if (std.mem.eql(u8, mode, "seed")) {
        clean(root, runs);
        switch (store) {
            .rows => try seedRows(a, root, count),
            .imprint => try seedImprint(a, root, count),
        }
    } else if (std.mem.eql(u8, mode, "open")) {
        try open(a, store, root, runs, count);
    } else if (std.mem.eql(u8, mode, "phases")) {
        switch (store) {
            .rows => try rowsPhases(a, root),
            .imprint => try imprintPhases(a, root),
        }
    } else if (std.mem.eql(u8, mode, "summary")) {
        try summary(a, store, runs, count);
    } else if (std.mem.eql(u8, mode, "clean")) {
        clean(root, runs);
    } else return fail();
    return 0;
}

fn fail() u8 {
    std.debug.print(usage, .{});
    return 2;
}

fn rowsOptions(root: []const u8) rows.Options {
    return .{ .root = root, .identity = identity, .cap = cap, .floor = 0, .daily = cap };
}

fn mib(n: u64) f64 {
    return @as(f64, @floatFromInt(n)) / (1 << 20);
}

fn ms(ns: i128) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

/// This process's physical footprint and its lifetime peak (Darwin rusage_info_v4, as server/process_memory.zig).
fn footprint() ?[2]u64 {
    if (builtin.os.tag != .macos) return null;
    const P = struct {
        extern "c" fn proc_pid_rusage(pid: c_int, flavor: c_int, buffer: *anyopaque) c_int;
    };
    var info: [37]u64 = undefined;
    if (P.proc_pid_rusage(platform.c.getpid(), 4, &info) != 0) return null;
    return .{ info[9], info[30] };
}

/// One file of the rows store, as its writer lays it out; a stub payload, since startup reads only the tail.
fn writeFile(dir: []const u8, name: rows.chain.Name, kind: rows.file.Kind, at: u32, pieces: []const []const u8) !void {
    const stem = rows.chain.hex(name);
    var w = try rows.file.Writer.open(dir, &stem, if (kind == .state) ".st" else ".blk");
    errdefer w.abort();
    if (kind == .state) try w.append(&@as([8]u8, @splat(0)));
    _ = try w.finish(.{ .kind = @backingInt(kind), .payload = 0, .payload_hash = 0, .name = name, .at = at, .block = rows.block_tokens }, pieces, false);
}

fn printPositions(ats: []u32) void {
    std.mem.sort(u32, ats, {}, std.sort.asc(u32));
    var sum: u64 = 0;
    for (ats) |x| sum += x;
    std.debug.print(" positions min={d} median={d} mean={d} max={d}", .{ ats[0], ats[ats.len / 2], sum / ats.len, ats[ats.len - 1] });
}

fn seedRows(a: Allocator, root: [:0]const u8, count: usize) !void {
    const t0 = nowNs();
    const r = try rows.Rows.open(a, corpus.snapshots(), rowsOptions(root));
    defer r.deinit();
    var turns = try corpus.Turns.init(a, 0x5eed);
    defer turns.deinit();
    var seen: std.AutoHashMapUnmanaged(rows.chain.Stem, void) = .empty;
    defer seen.deinit(a);
    var wants: std.ArrayList(rows.write.Want) = .empty;
    defer wants.deinit(a);
    const ats = try a.alloc(u32, count);
    defer a.free(ats);
    var n: usize = 0;
    var meta: u64 = 0;
    var spans: u64 = 0;
    var conversations: u64 = 0;
    var real: u64 = 0; // the blocks' bytes with the 27B's rows in them (each file here holds only its trailer)
    while (n < count) {
        try turns.next();
        conversations += 1;
        for (turns.marks.items) |m| {
            if (n == count) break;
            const toks = turns.tokens.items[0..m.at];
            const sp = turns.spans.items[0..m.spans];
            const ps = try r.parts(a, m.at);
            defer a.free(ps);
            wants.clearRetainingCapacity();
            try r.blockNames(a, ps, toks, sp, &.{}, m.at, &wants);
            for (wants.items) |w| {
                if ((try seen.getOrPut(a, rows.chain.stem(w.name))).found_existing) continue;
                try writeFile(r.dir, w.name, .block, 0, &.{});
                real += w.bytes;
            }
            const counts = [3]u32{ m.at, m.spans, 0 };
            const name = rows.chain.state(r.keying, toks, sp, &.{}, m.at);
            try writeFile(r.dir, name, .state, m.at, &.{ std.mem.asBytes(&counts), std.mem.sliceAsBytes(toks), std.mem.sliceAsBytes(sp) });
            ats[n] = m.at;
            meta += 12 + 4 * @as(u64, m.at) + 8 * @as(u64, m.spans);
            spans += m.spans;
            n += 1;
        }
    }
    std.debug.print("STARTUP_STATES_SEED store=rows states={d} blocks={d} real_blocks_gib={d:.0} conversations={d} meta_mib={d:.1} spans_per_state={d:.1} seed_s={d:.1}", .{ count, seen.count(), mib(real) / 1024, conversations, mib(meta), @as(f64, @floatFromInt(spans)) / @as(f64, @floatFromInt(count)), ms(nowNs() - t0) / 1e3 });
    printPositions(ats);
    std.debug.print("\n", .{});
}

/// Learned shared prefixes (--learn): 1K to 32K tokens log-uniform, a planned family's lookahead and 2,048-row starts.
fn seedImprint(a: Allocator, root: [:0]const u8, count: usize) !void {
    const t0 = nowNs();
    var m = try imprint.Imprint.open(a, root, identity, cap);
    defer m.deinit();
    var prng: std.Random.DefaultPrng = .init(0x1e47);
    const r = prng.random();
    var toks: std.ArrayList(u32) = .empty;
    defer toks.deinit(a);
    var starts: std.ArrayList(u32) = .empty;
    defer starts.deinit(a);
    const ats = try a.alloc(u32, count);
    defer a.free(ats);
    var meta: u64 = 0;
    for (ats) |*slot| {
        const at: u32 = @intFromFloat(1024 * std.math.pow(f64, 32, r.float(f64)));
        try toks.resize(a, at + 1);
        for (toks.items) |*t| t.* = r.uintLessThan(u32, 151_000);
        starts.clearRetainingCapacity();
        var s: u32 = 2048;
        while (s < at) : (s += 2048) try starts.append(a, s);
        const key = imprint.Imprint.keyOf(toks.items);
        try m.add(key, at, toks.items, starts.items, @as(u64, at) << 16);
        var buf: [1200]u8 = undefined;
        const p = try std.fmt.bufPrintSentinel(&buf, "{s}/{x:0>16}.bin", .{ m.dir, key }, 0);
        const fd = platform.c.open(p, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(platform.c.mode_t, 0o600));
        if (fd < 0) return error.SeedWrite;
        _ = platform.c.close(fd);
        slot.* = at;
        meta += 36 + 4 * @as(u64, at + 1) + 4 * starts.items.len;
    }
    std.debug.print("STARTUP_STATES_SEED store=imprint states={d} index_mib={d:.1} seed_s={d:.1}", .{ m.metas.items.len, mib(meta), ms(nowNs() - t0) / 1e3 });
    printPositions(ats);
    std.debug.print("\n", .{});
}

/// One timed open in a fresh process: its time, requested heap peak and live bytes, and the process's footprint peak.
fn open(a: Allocator, store: Store, root: [:0]const u8, runs: [:0]const u8, count: usize) !void {
    var peak: Peak = .{ .child = a };
    var found: usize = 0;
    var ns: i128 = 0;
    switch (store) {
        .rows => {
            const t0 = nowNs();
            const r = try rows.Rows.open(peak.allocator(), corpus.snapshots(), rowsOptions(root));
            ns = nowNs() - t0;
            found = r.states.items.len;
            r.deinit();
        },
        .imprint => {
            const t0 = nowNs();
            var m = try imprint.Imprint.open(peak.allocator(), root, identity, cap);
            ns = nowNs() - t0;
            found = m.metas.items.len;
            m.deinit();
        },
    }
    const fp = footprint() orelse .{ 0, 0 };
    std.debug.print("STARTUP_STATES store={t} states={d} ms={d:.1} heap_peak_mib={d:.1} footprint_peak_mib={d:.1}\n", .{ store, found, ms(ns), mib(peak.peak), mib(fp[1]) });
    if (found != count or peak.live != 0) return error.StatesLost;
    const fd = platform.c.open(runs, .{ .ACCMODE = .WRONLY, .CREAT = true, .APPEND = true }, @as(platform.c.mode_t, 0o600));
    if (fd < 0) return error.RunsWrite;
    defer _ = platform.c.close(fd);
    var buf: [128]u8 = undefined;
    const line = try std.fmt.bufPrint(&buf, "{d} {d} {d}\n", .{ ns, peak.peak, fp[1] });
    if (platform.c.write(fd, line.ptr, line.len) != @as(isize, @intCast(line.len))) return error.RunsWrite;
}

fn summary(a: Allocator, store: Store, runs: [:0]const u8, count: usize) !void {
    var cols: [3]std.ArrayList(u64) = .{ .empty, .empty, .empty };
    defer for (&cols) |*c| c.deinit(a);
    const fd = platform.c.open(runs, .{ .ACCMODE = .RDONLY }, @as(platform.c.mode_t, 0));
    if (fd < 0) return error.NoRuns;
    defer _ = platform.c.close(fd);
    var buf: [4096]u8 = undefined;
    const n = platform.c.read(fd, buf[0..].ptr, buf.len);
    if (n <= 0) return error.NoRuns;
    var lines = std.mem.tokenizeScalar(u8, buf[0..@intCast(n)], '\n');
    while (lines.next()) |line| {
        var it = std.mem.tokenizeScalar(u8, line, ' ');
        for (&cols) |*c| try c.append(a, try std.fmt.parseInt(u64, it.next() orelse return error.BadRuns, 10));
    }
    for (&cols) |*c| std.mem.sort(u64, c.items, {}, std.sort.asc(u64));
    const t = cols[0].items;
    const k = t.len / 2;
    std.debug.print("STARTUP_STATES_MEDIAN store={t} states={d} runs={d} ms={d:.1} min_ms={d:.1} max_ms={d:.1} heap_peak_mib={d:.1} footprint_peak_mib={d:.1} (max {d:.1})\n", .{ store, count, t.len, ms(t[k]), ms(t[0]), ms(t[t.len - 1]), mib(cols[1].items[k]), mib(cols[2].items[k]), mib(cols[2].items[t.len - 1]) });
}

/// Rows.open's work again by its parts, through the same public calls, after one open (caches as the timed runs left).
fn rowsPhases(a: Allocator, root: [:0]const u8) !void {
    const r = try rows.Rows.open(a, corpus.snapshots(), rowsOptions(root));
    defer r.deinit();
    const listed = try walk(a, r.dir, .none);
    const blocks = try walk(a, r.dir, .blocks);
    const states = try walk(a, r.dir, .states);
    var t0 = nowNs();
    var tokens: u64 = 0;
    for (r.states.items) |s| {
        tokens += s.tokens.len;
        const name = rows.chain.state(r.keying, s.tokens, s.spans, s.starts, s.at);
        if (!std.mem.eql(u8, &name, &s.name)) return error.NameDiffers;
    }
    const named = nowNs() - t0;
    t0 = nowNs();
    var wants: std.ArrayList(rows.write.Want) = .empty;
    defer wants.deinit(a);
    var names: u64 = 0;
    for (r.states.items) |*s| {
        wants.clearRetainingCapacity();
        try r.wantsOf(a, s, &wants);
        for (wants.items) |w| if (!r.blocks.contains(rows.chain.stem(w.name))) return error.BlockMissing;
        names += wants.items.len;
    }
    const planned = nowNs() - t0;
    std.debug.print("STARTUP_STATES_PHASES store=rows states={d} blocks={d} readdir_ms={d:.1} block_sizes_ms={d:.1} state_reads_ms={d:.1} name_checks_ms={d:.1} block_plans_ms={d:.1} block_refs={d} tokens={d}\n", .{ r.states.items.len, r.blocks.count(), ms(listed), ms(blocks - listed), ms(states - listed), ms(named), ms(planned), names, tokens });
}

const Walk = enum { none, blocks, states };

/// One pass over the directory, with startup's per-file work for blocks (their sizes) or states (trailer and metadata).
fn walk(a: Allocator, dir: [:0]const u8, what: Walk) !i128 {
    const t0 = nowNs();
    const d = platform.c.opendir(dir) orelse return error.RowsDir;
    defer _ = platform.c.closedir(d);
    while (platform.c.readdir(d)) |ent| {
        if (ent.type != platform.c.DT.REG) continue;
        const name = std.mem.sliceTo(&ent.name, 0);
        var buf: [1200]u8 = undefined;
        const p = try std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ dir, name }, 0);
        if (what == .blocks and std.mem.endsWith(u8, name, ".blk")) _ = try dirs.fileBytes(p);
        if (what == .states and std.mem.endsWith(u8, name, ".st")) {
            var o = try rows.file.open(p, .state, null);
            defer o.close();
            a.free(try rows.file.meta(&o, a));
        }
    }
    return nowNs() - t0;
}

/// Imprint.open's per-record parts measured alone: the unindexed-file sweep, and readIndex's duplicate checks.
fn imprintPhases(a: Allocator, root: [:0]const u8) !void {
    var t0 = nowNs();
    var m = try imprint.Imprint.open(a, root, identity, cap);
    defer m.deinit();
    const opened = nowNs() - t0;
    t0 = nowNs();
    try dirs.removeUnindexed(a, m.dir, m.metas.items);
    const swept = nowNs() - t0;
    t0 = nowNs();
    var keys: std.AutoHashMapUnmanaged(u64, void) = .empty; // as readIndex grows it, one record at a time
    defer keys.deinit(a);
    for (m.metas.items) |x| _ = try keys.getOrPut(a, x.key);
    const dupes = nowNs() - t0;
    std.debug.print("STARTUP_STATES_PHASES store=imprint states={d} open_ms={d:.1} sweep_ms={d:.1} duplicate_checks_ms={d:.1} rest_ms~{d:.1}\n", .{ m.metas.items.len, ms(opened), ms(swept), ms(dupes), ms(opened - swept - dupes) });
}

/// The seeded tree (two levels: identities and their files) and the runs file removed.
fn clean(root: [:0]const u8, runs: [:0]const u8) void {
    _ = platform.c.unlink(runs);
    remove(root, 1);
}

fn remove(dir: [:0]const u8, depth: u32) void {
    for (0..16) |_| {
        const d = platform.c.opendir(dir) orelse return;
        while (platform.c.readdir(d)) |ent| {
            const name = std.mem.sliceTo(&ent.name, 0);
            if (std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) continue;
            var buf: [1200]u8 = undefined;
            const p = std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ dir, name }, 0) catch continue;
            if (ent.type == platform.c.DT.DIR and depth > 0) remove(p, depth - 1) else _ = platform.c.unlink(p);
        }
        _ = platform.c.closedir(d);
        if (platform.c.rmdir(dir) == 0) return;
    }
}

test {
    _ = corpus;
}
