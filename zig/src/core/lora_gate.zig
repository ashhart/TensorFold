//! The gates of `tensorfold merge --quantize`: identity of every untouched byte, FP64 precision, the quantizer itself.
const std = @import("std");
const st = @import("safetensors.zig");
const lm = @import("lora_merge.zig");
const lp = @import("lora_plan.zig");
const Allocator = std.mem.Allocator;
const Io = std.Io;

/// Whether `t` is numerically all zero, so a negative zero counts; NaN does not.
fn tensorZero(t: st.Tensor) bool {
    const step = t.dtype.size();
    var i: usize = 0;
    while (i + step <= t.bytes.len) : (i += step) {
        const mag: u32 = switch (t.dtype) {
            .f32 => std.mem.readInt(u32, t.bytes[i..][0..4], .little) & 0x7fffffff,
            else => std.mem.readInt(u16, t.bytes[i..][0..2], .little) & 0x7fff,
        };
        if (mag != 0) return false;
    }
    return true;
}

/// Whether the adapter has no effect: alpha 0, or an all-zero factor in every projection (negative zero included).
pub fn zeroEffect(p: *const lp.Plan) bool {
    if (p.scale == 0) return true;
    for (p.jobs) |j| {
        if (!tensorZero(p.adapter.get(j.a).?) and !tensorZero(p.adapter.get(j.b).?)) return false;
    }
    return true;
}

/// Every gate on a merged folder; true when all pass.
pub fn check(gpa: Allocator, arena: Allocator, io: Io, w: *Io.Writer, dense_dir: []const u8, adapter: []const u8, dest: []const u8, shipped_dir: []const u8) !bool {
    var dense = try lp.Shards.init(gpa, arena, io, w, dense_dir);
    defer dense.deinit();
    var old = try lp.Shards.init(gpa, arena, io, w, shipped_dir);
    defer old.deinit();
    var new = try lp.Shards.init(gpa, arena, io, w, dest);
    defer new.deinit();
    var p = try lp.plan(gpa, arena, io, w, &dense, &old, adapter);
    defer p.deinit(io);
    const zero = zeroEffect(&p);
    try w.print("adapter: {d} projections, r = {d}, alpha / r = {d}, zero effect: {}\n", .{ p.jobs.len, p.r, p.scale, zero });
    var ok = try identity(gpa, arena, io, w, shipped_dir, dest, &p, zero);
    for (p.jobs) |j| {
        const x = try lp.denseOf(gpa, w, &dense, j);
        defer gpa.free(x);
        const bq = try lp.quantOf(gpa, &old, j);
        defer lp.free(gpa, bq);
        const mq = try lp.quantOf(gpa, &new, j);
        defer lp.free(gpa, mq);
        const f = try lp.factors(gpa, w, &p, j);
        defer gpa.free(f.a);
        defer gpa.free(f.b);
        const r = try lm.measure(gpa, j.group, j.n, j.k, x, .{ .a = f.a, .b = f.b, .r = p.r, .scale = p.scale }, lm.viewOf(bq), lm.viewOf(mq));
        const m = r.merged;
        const s = r.shipped;
        try w.print("{s} [{d} x {d}, group {d}]:\n  merged : worst column {d}: max {e:.3} (rms {e:.3}); aggregate rms {e:.3}, max {e:.3}\n  shipped: worst column {d}: max {e:.3} (rms {e:.3}); aggregate rms {e:.3}, max {e:.3}\n  worst column merged / shipped: {d:.4} (printed, not gated); aggregate rms merged / shipped: {d:.4}; within tolerance: {s}\n", .{ j.module, j.n, j.k, j.group, m.worst_col, m.worst_col_err, m.worst_col_rms, m.rms_err, m.max_err, s.worst_col, s.worst_col_err, s.worst_col_rms, s.rms_err, s.max_err, r.worstRatio(), r.rmsRatio(), if (r.withinTolerance()) "yes" else "NO" });
        var bad: usize = 0;
        const lora: lm.Lora = .{ .a = f.a, .b = f.b, .r = p.r, .scale = p.scale };
        const differ = lm.recompute(gpa, j.group, j.n, j.k, x, lora, lm.viewOf(mq), &bad) catch |e| switch (e) {
            error.NonFinite => return lp.refuse(w, j.module, try std.fmt.allocPrint(arena, "group {d}: a non-finite value, range or scale", .{bad})),
            else => return e,
        };
        try w.print("  recomputed from the dense weights and the adapter: {d} of {d} groups differ from OUT\n", .{ differ, mq.scales.len });
        ok = ok and differ == 0;
        const exact = r.rmsRatio() == 1 and r.worstRatio() == 1;
        if (zero and !exact) try w.writeAll("  zero effect but the error differs from the shipped quantization's\n");
        ok = ok and r.withinTolerance() and (!zero or exact);
    }
    try w.print("{s}\n", .{if (ok) "merge check: all gates pass" else "merge check: FAILED"});
    return ok;
}

/// The quantizer's output on each module's dense weight against the shipped words, scales and biases.
pub fn quantizerCheck(gpa: Allocator, arena: Allocator, io: Io, w: *Io.Writer, dense_dir: []const u8, shipped_dir: []const u8, modules: []const []const u8) !bool {
    var dense = try lp.Shards.init(gpa, arena, io, w, dense_dir);
    defer dense.deinit();
    var shipped = try lp.Shards.init(gpa, arena, io, w, shipped_dir);
    defer shipped.deinit();
    const quant = try lp.quantConfig(arena, io, w, shipped_dir);
    var ok = true;
    for (modules) |module| {
        const name = try lp.resolveShipped(arena, w, &shipped, module);
        const lay = try lp.layoutOf(arena, w, &shipped, quant, name);
        const j: lp.Job = .{ .module = module, .shipped = name, .n = lay.n, .k = lay.k, .group = lay.group, .dtype = lay.dtype };
        const x = try lp.denseOf(gpa, w, &dense, j);
        defer gpa.free(x);
        const old = try lp.quantOf(gpa, &shipped, j);
        defer lp.free(gpa, old);
        const new = try lm.allocQuant(gpa, j.n, j.k, j.group);
        defer lm.freeQuant(gpa, new);
        if (try lm.quantize(j.group, x, new)) |bad| return lp.refuse(w, module, try std.fmt.allocPrint(arena, "group {d}: a non-finite value, range or scale", .{bad}));
        const a = lm.agreement(j.group, lm.viewOf(old), lm.viewOf(new));
        try w.print("{s} [{d} x {d}, group {d}]: scales {d:.4}%, biases {d:.4}%, codes {d:.4}%, whole groups {d:.4}%: {s}\n", .{ module, j.n, j.k, j.group, a.scales * 100, a.biases * 100, a.codes * 100, a.groups * 100, if (a.identical()) "identical" else "DIFFERS" });
        ok = ok and a.identical();
    }
    try w.print("{s}\n", .{if (ok) "quantizer check: byte-identical on every tensor" else "quantizer check: FAILED"});
    return ok;
}

/// The names of the regular files in `dir`; anything else in it is reported on `w` and fails the gate.
fn listFiles(arena: Allocator, io: Io, w: *Io.Writer, dir: []const u8, ok: *bool) !std.StringArrayHashMapUnmanaged(void) {
    var names: std.StringArrayHashMapUnmanaged(void) = .empty;
    var d = try Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
    defer d.close(io);
    var it = d.iterate();
    while (try it.next(io)) |e| {
        if (!lp.isRegular(d, io, e)) {
            try w.print("{s}: {s} holds something that is not a regular file (or a dangling symlink)\n", .{ e.name, dir });
            ok.* = false;
            continue;
        }
        try names.put(arena, try arena.dupe(u8, e.name), {});
    }
    return names;
}

const Range = struct { begin: usize, end: usize };

fn lessRange(_: void, a: Range, b: Range) bool {
    return a.begin < b.begin;
}

/// Every file of SHIPPED against OUT's: the same set, the same lengths, the same bytes outside the named payloads.
pub fn identity(gpa: Allocator, arena: Allocator, io: Io, w: *Io.Writer, shipped: []const u8, dest: []const u8, p: *const lp.Plan, zero: bool) !bool {
    var named: std.StringHashMapUnmanaged(void) = .empty;
    for (p.jobs) |j| for ([_][]const u8{ "weight", "scales", "biases" }) |leaf| try named.put(arena, try std.fmt.allocPrint(arena, "{s}.{s}", .{ j.shippedName(), leaf }), {});
    var ok = true;
    const src = try listFiles(arena, io, w, shipped, &ok);
    const dst = try listFiles(arena, io, w, dest, &ok);
    for (dst.keys()) |name| if (!src.contains(name)) {
        try w.print("{s}: in OUT but not in the shipped checkpoint\n", .{name});
        ok = false;
    };
    var identical: usize = 0;
    var touched: usize = 0;
    for (src.keys()) |name| {
        if (!dst.contains(name)) {
            try w.print("{s}: missing in OUT\n", .{name});
            ok = false;
            continue;
        }
        const a_path = try std.fs.path.join(arena, &.{ shipped, name });
        const b_path = try std.fs.path.join(arena, &.{ dest, name });
        if (!std.mem.endsWith(u8, name, ".safetensors")) {
            const x = try Io.Dir.cwd().readFileAlloc(io, a_path, arena, .limited(1 << 28));
            const y = try Io.Dir.cwd().readFileAlloc(io, b_path, arena, .limited(1 << 28));
            if (std.mem.eql(u8, x, y)) identical += 1 else {
                try w.print("{s}: differs\n", .{name});
                ok = false;
            }
            continue;
        }
        var a = try st.File.open(gpa, io, a_path);
        defer a.close(io);
        var b = try st.File.open(gpa, io, b_path);
        defer b.close(io);
        if (std.mem.eql(u8, a.map.memory, b.map.memory)) {
            identical += 1;
            continue;
        }
        touched += 1;
        const bad = !try touchedShard(arena, w, name, &a, &b, &named, zero);
        ok = ok and !bad;
    }
    try w.print("identity: {d} files byte-identical, {d} touched: {s}\n", .{ identical, touched, if (ok) "ok" else "FAIL" });
    return ok;
}

/// A shard that differs from the shipped one: true when only the named payloads moved (and none for a zero adapter).
fn touchedShard(arena: Allocator, w: *Io.Writer, name: []const u8, a: *const st.File, b: *const st.File, named: *const std.StringHashMapUnmanaged(void), zero: bool) !bool {
    const x = a.map.memory;
    const y = b.map.memory;
    if (x.len != y.len) {
        try w.print("{s}: length {d} differs from the shipped {d}\n", .{ name, y.len, x.len });
        return false;
    }
    if (zero) {
        try w.print("{s}: zero effect but the shard differs\n", .{name});
        return false;
    }
    var ranges: std.ArrayList(Range) = .empty;
    var moved: usize = 0;
    for (a.names.keys(), a.names.values()) |key, e| {
        if (!named.contains(key)) continue;
        try ranges.append(arena, .{ .begin = a.data + e.begin, .end = a.data + e.end });
        const t = a.get(key).?;
        const u = b.get(key) orelse {
            try w.print("{s}: {s} is missing in OUT\n", .{ name, key });
            return false;
        };
        if (!std.mem.eql(u8, t.bytes, u.bytes)) moved += 1;
    }
    std.mem.sort(Range, ranges.items, {}, lessRange);
    var at: usize = 0;
    for (ranges.items) |r| {
        if (!std.mem.eql(u8, x[at..r.begin], y[at..r.begin])) return outside(w, name, at);
        at = r.end;
    }
    if (!std.mem.eql(u8, x[at..], y[at..])) return outside(w, name, at);
    try w.print("{s}: touched shard, same length, every byte outside {d} named payloads identical\n", .{ name, moved });
    return true;
}

fn outside(w: *Io.Writer, name: []const u8, from: usize) !bool {
    try w.print("{s}: bytes outside the named tensors differ (after offset {d})\n", .{ name, from });
    return false;
}
