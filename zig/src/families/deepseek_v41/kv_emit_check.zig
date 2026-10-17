//! `tf-dsv41 kv-emit CAPTURE`: the emitters' calls in pool mode (paged and split) against what kv_state binds.

const std = @import("std");
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const block_prefill = @import("block_prefill.zig");
const buffers = @import("buffers.zig");
const check = @import("calls_check.zig");
const Config = @import("config.zig").Config;

fn roleOf(x: calls.Arg) []const u8 {
    return switch (x) {
        .t, .opaque_table => |t| switch (t.role) {
            .buf => |b| b,
            .weight => |w| w,
            .empty => "",
        },
        else => "",
    };
}

fn named(c: *const calls.Call, name: []const u8) ?calls.Arg {
    for (c.args) |a| if (std.mem.eql(u8, a.name, name)) return a.arg;
    return null;
}

fn int(x: ?calls.Arg) i64 {
    return if (x) |v| switch (v) {
        .i => |i| i,
        else => -1,
    } else -1;
}

const Ctx = struct {
    a: std.mem.Allocator,
    log: *std.Io.Writer,
    where: []const u8,
    bad: usize = 0,

    fn fail(c: *Ctx, comptime f: []const u8, args: anytype) !void {
        c.bad += 1;
        if (c.bad <= 20) try c.log.print("  {s}: " ++ f ++ "\n", .{c.where} ++ args);
    }
};

/// One program's calls in a pool mode: the invariants of the module doc.
fn checkCalls(x: *Ctx, cfg: *const Config, cs: []const calls.Call, layers: []const u32, split: bool) !void {
    var exchanges: usize = 0;
    var pending = false; // an exchange ran and its rows are live
    var want: usize = 0;
    for (layers) |L| {
        const md = cfg.mode(L);
        if ((md == .full or md == .reindex) and cfg.compressRatio(L) > 0) want += 1;
    }
    for (cs) |*c| {
        for (c.args) |a| {
            const r = roleOf(a.arg);
            if (std.mem.startsWith(u8, r, "s.L") and (std.mem.indexOf(u8, r, ".comp.") != null or std.mem.endsWith(u8, r, ".ik")))
                try x.fail("{s} still names the contiguous role {s}", .{ c.name, r });
        }
        if (c.glue) {
            if (std.mem.startsWith(u8, c.name, "glue.kx_")) {
                if (!split) try x.fail("an exchange in paged mode", .{});
                exchanges += 1;
                pending = true;
            }
            continue;
        }
        if (std.mem.eql(u8, c.name, "_kv_store")) {
            const ratio = int(named(c, "RATIO"));
            if (ratio <= 0) continue;
            const pt = roleOf(named(c, "PT").?);
            if (!std.mem.eql(u8, pt, if (split) "s.kv.ct" else "s.kv.pt")) try x.fail("_kv_store ratio {d} through {s}", .{ ratio, pt });
            if (int(named(c, "PSH")) != block.Pool.shift(ratio)) try x.fail("_kv_store PSH {d}", .{int(named(c, "PSH"))});
            if (int(named(c, "v_stride")) != 584 or int(named(c, "s_stride")) != 584) try x.fail("_kv_store strides", .{});
            if (!std.mem.startsWith(u8, roleOf(named(c, "V").?), "s.kv.comp.L")) try x.fail("_kv_store into {s}", .{roleOf(named(c, "V").?)});
        } else if (std.mem.eql(u8, c.name, "_index_k") or std.mem.eql(u8, c.name, "_scores") or std.mem.eql(u8, c.name, "_scores_b")) {
            if (!std.mem.startsWith(u8, roleOf(named(c, "IK").?), "s.kv.ik.L")) try x.fail("{s} keys {s}", .{ c.name, roleOf(named(c, "IK").?) });
            if (!std.mem.eql(u8, roleOf(named(c, "PT").?), "s.kv.pt")) try x.fail("{s} through {s}", .{ c.name, roleOf(named(c, "PT").?) });
        } else if (std.mem.eql(u8, c.name, "tf_dsv41_attn_cuda_v1.attn")) {
            const cv = roleOf(c.args[1].arg);
            if (cv.len == 0) continue; // an SWA-only layer
            const pt = roleOf(c.args[11].arg);
            const psh = c.args[20].arg.i;
            if (split) {
                if (!std.mem.eql(u8, cv, "w.kx.recv") or !std.mem.eql(u8, roleOf(c.args[3].arg), "w.kx.tok") or pt.len != 0 or psh != 0)
                    try x.fail("attn reads {s} at {s} (pt {s}, psh {d})", .{ cv, roleOf(c.args[3].arg), pt, psh });
                if (!pending) try x.fail("attn before any exchange", .{});
            } else if (!std.mem.startsWith(u8, cv, "s.kv.comp.L") or !std.mem.eql(u8, pt, "s.kv.pt") or psh <= 0)
                try x.fail("attn reads {s} through {s} psh {d}", .{ cv, pt, psh });
        } else if (std.mem.eql(u8, c.name, "_fused")) {
            if (!named(c, "COMP").?.b) continue;
            const cv = roleOf(named(c, "CV").?);
            if (split) {
                if (!std.mem.eql(u8, cv, "w.kx.recv") or !std.mem.eql(u8, roleOf(named(c, "TOK").?), "w.kx.tok") or named(c, "PT").? != .none)
                    try x.fail("_fused reads {s}", .{cv});
                if (!pending) try x.fail("_fused before any exchange", .{});
            } else if (!std.mem.startsWith(u8, cv, "s.kv.comp.L") or !std.mem.eql(u8, roleOf(named(c, "PT").?), "s.kv.pt"))
                try x.fail("_fused reads {s}", .{cv});
        }
    }
    if (split and exchanges != want) try x.fail("{d} exchanges, {d} compressed index layers", .{ exchanges, want });
}

/// The pool every check binds: `limit` positions in one slot, as kv_state.zig sizes it for world 2.
fn poolOf(limit: i64, split: bool) block.Pool {
    const pages = @divExact(limit, block.Pool.page);
    return .{ .comp_pages = if (split) @divExact(pages, 2) + 2 else pages + 1, .ik_pages = pages + 1, .pts = pages, .split = split };
}

pub fn checkAll(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, experts: []const u8, log: *std.Io.Writer) !usize {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cwd = std.Io.Dir.cwd();
    var w = try check.widthsFromCapture(a, try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "weights.json" }), a, .limited(1 << 28)), experts);
    const cfg: Config = .{};
    const ws = try check.windows(a, io, dir);
    try check.gmWidths(a, &w, ws);
    var bad: usize = 0;
    var programs: usize = 0;
    for ([_]bool{ false, true }) |split| {
        var opts = try check.optionsOf(a, io, dir);
        opts.gm_v2 = check.gmMode(ws);
        opts.r1_sig = !opts.r1 and try check.r1SigOf(a, ws);
        opts.pool = poolOf(opts.limit, split);
        const P = opts.pool.?;
        for (ws) |win| {
            const cs = if (win.prefill)
                try block_prefill.emitPrefill(a, &cfg, &w, opts, win.layers, win.n, win.start, true)
            else
                try block.emit(a, &cfg, &w, opts, win.layers, win.n, win.start, true);
            var x: Ctx = .{ .a = a, .log = log, .where = try std.fmt.allocPrint(a, "{s} {s} {s}", .{ if (split) "split" else "paged", win.set, win.phase }) };
            try checkCalls(&x, &cfg, cs, win.layers, split);
            // the plan's pool roles fit the tensors kv_state binds
            var plan: buffers.Plan = .{ .a = a };
            try plan.add(cs);
            for (plan.sizes.keys(), plan.sizes.values()) |k, v| {
                const cap: u64 = if (std.mem.startsWith(u8, k, "s.kv.comp.L"))
                    @intCast(P.comp_pages * block.Pool.page * block.Pool.row_bytes) // ratio 1's rows a page: the most
                else if (std.mem.startsWith(u8, k, "s.kv.ik.L"))
                    @intCast(P.ik_pages * block.Pool.page * block.Pool.ik_bytes)
                else if (std.mem.eql(u8, k, "s.kv.pt") or std.mem.eql(u8, k, "s.kv.ct"))
                    @intCast(4 * P.pts)
                else
                    continue;
                if (v > cap) try x.fail("plan: {s} reaches {d} B, the pool holds {d}", .{ k, v, cap });
            }
            try log.print("{s}: {d} calls, {s}\n", .{ x.where, cs.len, if (x.bad == 0) "ok" else "FAIL" });
            bad += x.bad;
            programs += 1;
        }
    }
    try log.print("kv-emit: {d} programs, {d} problems\n", .{ programs, bad });
    return bad;
}
