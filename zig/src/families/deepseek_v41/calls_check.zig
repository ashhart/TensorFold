//! `tf-dsv41 check`: the emitters' calls against a capture's launches, on the host (kernels, shapes, roles).

const std = @import("std");
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const block_prefill = @import("block_prefill.zig");
const Config = @import("config.zig").Config;
const buffers = @import("buffers.zig");
const cuda = @import("cuda");
const tc = @import("triton_call.zig");
const Value = std.json.Value;

/// Dense K2s from the capture's weights.json (each "<prefix>.trellis"), the experts' ranges from a listing.
pub fn widthsFromCapture(a: std.mem.Allocator, weights_json: []const u8, experts: []const u8) !block.Widths {
    var w: block.Widths = .{};
    const root = try std.json.parseFromSliceLeaky(Value, a, weights_json, .{});
    var it = root.object.iterator();
    while (it.next()) |e| {
        const name = e.key_ptr.*;
        if (!std.mem.endsWith(u8, name, ".trellis")) continue;
        const shape = e.value_ptr.object.get("shape").?.array.items;
        if (shape.len != 3) continue;
        try w.dense.put(a, name[0 .. name.len - ".trellis".len], @intCast(@divExact(shape[2].integer, 8)));
    }
    try parseExperts(a, &w, experts);
    return w;
}

/// A "prefix k2" listing (fixtures/dense-k2.txt) into the dense widths.
pub const parseDense = @import("widths_listing.zig").parseDense;

/// The whole model emitted on the host from the K2 listings: every call's count and the buffer plan's bytes.
pub fn planAll(gpa: std.mem.Allocator, dense: []const u8, experts: []const u8, log: *std.Io.Writer) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var w: block.Widths = .{};
    try parseDense(a, &w, dense);
    try parseExperts(a, &w, experts);
    const cfg: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    var plan: buffers.Plan = .{ .a = a };
    for ([_]i64{ 1, 2, 3, 4, 8, 16 }) |n| {
        const cs = try block.emit(a, &cfg, &w, .{}, &backbone, n, 4096 - 16, true);
        try plan.add(cs);
        var tri: usize = 0;
        var pf: usize = 0;
        for (cs) |c| {
            if (c.triton) tri += 1;
            if (std.mem.startsWith(u8, c.name, "tf_dsv41_l2p")) pf += 1;
        }
        try log.print("backbone window, {d} rows: {d} launches ({d} Triton, {d} extension incl. {d} L2 prefetch)\n", .{ n, cs.len, tri, cs.len - tri, pf });
    }
    for ([_]u32{ 40, 41, 42 }) |L| {
        const cs = try block.emit(a, &cfg, &w, .{}, &.{L}, 16, 4096 - 16, false);
        try plan.add(cs);
        try log.print("DSpark block {d}, 16 rows: {d} launches\n", .{ L, cs.len });
    }
    const t = plan.totals();
    const names = [_][]const u8{ "persistent (s.)", "window (w.)", "layer (L.)" };
    for (names, 0..) |nm, i| try log.print("{s}: {d} roles, {d:.1} MiB\n", .{ nm, t.roles[i], @as(f64, @floatFromInt(t.bytes[i])) / (1 << 20) });
}

pub const parseExperts = @import("widths_listing.zig").parseExperts;

/// (role, captured buffer id) of every buffer-backed tensor of a call (its arguments and the capture's, same shape).
pub fn bindings(a: std.mem.Allocator, want: calls.Arg, cap: Value, out: *std.ArrayList(Binding)) !void {
    switch (want) {
        .list => |items| for (items, cap.object.get("items").?.array.items) |x, y| try bindings(a, x, y, out),
        .t, .opaque_table => |t| switch (t.role) {
            .buf => |r| if (cap.object.get("buf")) |id| {
                var numel: i64 = 1;
                for (t.shape) |d| numel *= d;
                if (numel > 0) try out.append(a, .{ .role = r, .id = id.integer });
            },
            else => {},
        },
        else => {},
    }
}

pub const Binding = struct { role: []const u8, id: i64 };

/// The window's dataflow by content: a role read from another role's leftover storage is wired wrong (reported).
pub const Flow = struct {
    a: std.mem.Allocator,
    /// role -> (captured storage, its bytes' digest) after the role's last call
    last: std.StringHashMapUnmanaged([2][]const u8) = .empty,
    glue: std.StringArrayHashMapUnmanaged(u32) = .empty,
    /// window / layer roles first touched by a call that leaves them unchanged: inputs the forward's glue fills
    inputs: std.StringArrayHashMapUnmanaged(u32) = .empty,
    /// roles a glue step touched since coming into scope: a first read equal to freed storage is not a wrong wire
    glued: std.StringHashMapUnmanaged(void) = .empty,
    miswired: u32 = 0,

    pub fn begin(f: *Flow, b: calls.Begin) void {
        if (b == .none) return;
        inline for (.{ &f.last, &f.glued }) |m| {
            var drop: std.ArrayList([]const u8) = .empty;
            var it = m.keyIterator();
            while (it.next()) |k| {
                if (std.mem.startsWith(u8, k.*, "s.")) continue;
                if (b == .layer and std.mem.startsWith(u8, k.*, "w.")) continue;
                drop.append(f.a, k.*) catch {};
            }
            for (drop.items) |k| _ = m.remove(k);
        }
    }

    /// A glue step of ours (no captured launch): the roles it reads or writes.
    pub fn glueCall(f: *Flow, c: *const calls.Call) !void {
        for (c.args) |x| if (x.arg == .t and x.arg.t.role == .buf) try f.glued.put(f.a, x.arg.t.role.buf, {});
    }

    pub fn call(f: *Flow, c: *const calls.Call, op: std.json.ObjectMap, log: *std.Io.Writer, where: []const u8) !void {
        var bs: std.ArrayList(Binding) = .empty;
        const got = op.get("args").?.array.items;
        const m = @min(c.args.len, got.len); // a count mismatch is calls.check's to report
        for (c.args[0..m], got[0..m]) |x, y| try bindings(f.a, x.arg, y, &bs);
        const bufs = op.get("buffers").?.array.items;
        for (bs.items) |b| {
            const o = for (bufs) |v| {
                if (v.object.get("id").?.integer == b.id) break v.object;
            } else continue;
            const before = o.get("before").?;
            if (before != .string) continue;
            const key = o.get("key").?.string;
            if (f.last.get(b.role)) |h| {
                if (!std.mem.eql(u8, h[1], before.string)) {
                    const g = try f.glue.getOrPut(f.a, b.role);
                    if (!g.found_existing) g.value_ptr.* = 0;
                    g.value_ptr.* += 1;
                }
                continue;
            }
            // a first touch the call leaves as it was is a read: something outside the calls filled it
            if (!std.mem.startsWith(u8, b.role, "s.") and std.mem.eql(u8, before.string, o.get("after").?.string)) {
                const g = try f.inputs.getOrPut(f.a, b.role);
                if (!g.found_existing) g.value_ptr.* = 0;
                g.value_ptr.* += 1;
            }
            // a first touch that writes overwrites whatever a freed storage held; only a read can be miswired
            const read = std.mem.eql(u8, before.string, o.get("after").?.string) and !f.glued.contains(b.role);
            var it = f.last.iterator();
            while (it.next()) |e| if (read and std.mem.eql(u8, e.value_ptr[0], key) and std.mem.eql(u8, e.value_ptr[1], before.string)) {
                f.miswired += 1;
                try log.print("  {s} {s}: {s} first read holds what {s} last held\n", .{ where, c.name, b.role, e.key_ptr.* });
                break;
            };
        }
        for (bs.items) |b| {
            const o = for (bufs) |v| {
                if (v.object.get("id").?.integer == b.id) break v.object;
            } else continue;
            try f.last.put(f.a, b.role, .{ try f.a.dupe(u8, o.get("key").?.string), try f.a.dupe(u8, o.get("after").?.string) });
        }
    }
};

/// The emitter options a capture ran with, from meta.json's env (R1 when its knobs were set).
pub fn optionsOf(a: std.mem.Allocator, io: std.Io, dir: []const u8) !block.Options {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "meta.json" }), a, .limited(1 << 24));
    const meta = try std.json.parseFromSliceLeaky(Value, a, text, .{});
    var o: block.Options = .{};
    const env = meta.object.get("env") orelse return o;
    if (env.object.get("TF_TP_WORLD")) |v| if (v == .string) {
        o.world = try std.fmt.parseInt(u32, v.string, 10);
    };
    if (env.object.get("TF_TP_RANK")) |v| if (v == .string) {
        o.rank = try std.fmt.parseInt(u32, v.string, 10);
    };
    const on = struct {
        fn f(e: Value, k: []const u8) bool {
            const v = e.object.get(k) orelse return false;
            return v == .string and std.mem.eql(u8, v.string, "1");
        }
    }.f;
    o.r1 = on(env, "TF_DSV41_MHC_DEFER") and on(env, "TF_DSV41_RG_PRUNE") and on(env, "TF_DSV41_ATTN_ROPE") and on(env, "TF_DSV41_RMS_FOLD");
    return o;
}

pub const Window = struct { set: []const u8, phase: []const u8, layers: []const u32, n: i64, start: i64, ops: []std.json.ObjectMap, prefill: bool = false };

/// The capture's phases in order: each set's decode windows from the prompt's end, then its prefill segment.
pub fn windows(a: std.mem.Allocator, io: std.Io, dir: []const u8) ![]Window {
    const cwd = std.Io.Dir.cwd();
    const meta = try std.json.parseFromSliceLeaky(Value, a, try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "meta.json" }), a, .limited(1 << 24)), .{});
    const prompt = meta.object.get("prompt").?.integer;
    const sizes = meta.object.get("windows").?.array.items;
    const text = try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "ops.jsonl" }), a, .limited(1 << 34));
    var out: std.ArrayList(Window) = .empty;
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    var cur: ?*Window = null;
    var ops: std.ArrayList(std.json.ObjectMap) = .empty;
    while (lines.next()) |line| {
        const op = (try std.json.parseFromSliceLeaky(Value, a, line, .{})).object;
        const set = op.get("set").?.string;
        const phase = op.get("phase").?.string;
        if (phase.len < 2 or (phase[0] != 'w' and phase[0] != 'p')) continue;
        if (cur == null or !std.mem.eql(u8, cur.?.set, set) or !std.mem.eql(u8, cur.?.phase, phase)) {
            if (cur) |c| c.ops = try ops.toOwnedSlice(a);
            const n = try std.fmt.parseInt(i64, phase[1..], 10);
            var start = prompt;
            for (sizes) |s| {
                if (phase[0] == 'w' and s.integer == n) break;
                start += s.integer;
            }
            const ls = meta.object.get("sets").?.object.get(set).?.object.get("layers").?.array.items;
            const layers = try a.alloc(u32, ls.len);
            for (ls, layers) |x, *y| y.* = @intCast(x.integer);
            try out.append(a, .{ .set = set, .phase = phase, .layers = layers, .n = n, .start = start, .ops = &.{}, .prefill = phase[0] == 'p' });
            cur = &out.items[out.items.len - 1];
        }
        try ops.append(a, op);
    }
    if (cur) |c| c.ops = try ops.toOwnedSlice(a);
    return out.items;
}

/// The routed experts' K2s per layer (Widths.gm) from the prefill phases' x3gm launches (the widths launched).
pub fn gmWidths(a: std.mem.Allocator, w: *block.Widths, ws: []const Window) !void {
    for (ws) |win| for (win.ops) |op| {
        const name = op.get("name").?.string;
        const down = std.mem.eql(u8, name, "tf_dsv41_x3gm_v1.down");
        if (!down and !std.mem.eql(u8, name, "tf_dsv41_x3gm_v1.gateup")) continue;
        const args = op.get("args").?.array.items;
        const svh = args[if (down) 8 else 10].object.get("weight").?.string;
        const L = try std.fmt.parseInt(u32, svh[1..std.mem.indexOfScalar(u8, svh, '.').?], 10);
        const k2: u5 = @intCast(args[if (down) 12 else 16].object.get("v").?.integer);
        const gop = try w.gm.getOrPut(a, L);
        if (!gop.found_existing) gop.value_ptr.* = .{ 0, 0 };
        gop.value_ptr[@intFromBool(down)] |= @as(u32, 1) << k2;
    };
}

/// True for a capture whose mHC boundary launched with 26 arguments and spin -1 (`block.Options.r1_sig`).
pub fn r1Sig(ops_lists: []const []std.json.ObjectMap) bool {
    for (ops_lists) |ops| for (ops) |op| {
        const name = (op.get("name") orelse continue);
        if (name != .string or !std.mem.eql(u8, name.string, "tf_dsv41_mhc_cuda_v1.run")) continue;
        const args = (op.get("args") orelse continue).array.items;
        if (args.len != 26) return false;
        const spin = args[24].object.get("v") orelse return false;
        return spin == .integer and spin.integer < 0;
    };
    return false;
}

/// `r1Sig` over a capture's windows.
pub fn r1SigOf(a: std.mem.Allocator, ws: []const Window) !bool {
    const lists = try a.alloc([]std.json.ObjectMap, ws.len);
    for (lists, ws) |*l, w| l.* = w.ops;
    return r1Sig(lists);
}

pub fn gmMode(ws: []const Window) @import("knob_types.zig").GmMode {
    const Names = struct {
        ws: []const Window,
        w: usize = 0,
        i: usize = 0,
        pub fn next(it: *@This()) ?[]const u8 {
            while (it.w < it.ws.len) {
                const ops = it.ws[it.w].ops;
                if (it.i < ops.len) {
                    it.i += 1;
                    return ops[it.i - 1].get("name").?.string;
                }
                it.w += 1;
                it.i = 0;
            }
            return null;
        }
    };
    var it: Names = .{ .ws = ws };
    return @import("gm2pf.zig").ofLaunches(&it);
}

/// Triton variant choice on the host: exactly one compiled variant may match, the one the capture launched.
pub const Variants = struct {
    specs: []const cuda.aot.Spec,
    checked: usize = 0,
    wrong: usize = 0,

    fn base(_: *anyopaque, t: calls.Tensor) anyerror!u64 {
        var numel: i64 = 1;
        for (t.shape) |n| numel *= n;
        return if (numel == 0) 0 else 0x7f00_0000_0000 + @as(u64, @intCast(t.offset)); // a 256-aligned buffer + the view
    }

    pub fn check(v: *Variants, a: std.mem.Allocator, c: *const calls.Call, op: std.json.ObjectMap) !?[]const u8 {
        if (!c.triton) return null;
        var mine: std.ArrayList(cuda.aot.Spec) = .empty;
        for (v.specs) |s| if (std.mem.eql(u8, s.@"fn", c.name)) try mine.append(a, s);
        var dummy: u8 = 0;
        const sp = try tc.split(a, c, mine.items, .{ .ctx = &dummy, .of = base });
        v.checked += 1;
        var hit: ?[]const u8 = null;
        var n: usize = 0;
        for (mine.items) |s| if (cuda.aot.specMatches(s, sp.args, sp.consts)) {
            n += 1;
            hit = s.hash;
        };
        const want = op.get("hash").?.string;
        if (n == 1 and std.mem.eql(u8, hit.?, want)) return null;
        v.wrong += 1;
        return try std.fmt.allocPrint(a, "{d} of {d} variants match (want {s})", .{ n, mine.items.len, want[0..12] });
    }
};

/// Checks every window; one line a window plus each mismatch (up to `show` a window) on `log`. Returns mismatches.
pub fn checkAll(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, experts: []const u8, log: *std.Io.Writer, show: usize) !usize {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cwd = std.Io.Dir.cwd();
    var w = try widthsFromCapture(a, try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "weights.json" }), a, .limited(1 << 28)), experts);
    const cfg: Config = .{};
    const ws = try windows(a, io, dir);
    try gmWidths(a, &w, ws);
    var opts = try optionsOf(a, io, dir);
    opts.gm_v2 = gmMode(ws);
    opts.r1_sig = !opts.r1 and try r1SigOf(a, ws);
    if (opts.r1) try log.print("options: R1 (the capture's knobs)\n", .{});
    if (opts.r1_sig) try log.print("options: R1's trailing arguments at their off values (the twin's bindings)\n", .{});
    if (opts.gm_v2 != .off) try log.print("options: x3gm v2 {t} (the capture's launches)\n", .{opts.gm_v2});
    var bad: usize = 0;
    // the variant gate when the capture's AOT set carries the specialization
    var variants: ?Variants = null;
    if (cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "aot", "aot.json" }), a, .limited(1 << 26))) |text| {
        if (std.mem.indexOf(u8, text, "\"consts\"") != null) variants = .{ .specs = (try cuda.aot.parseSpecs(a, text)).value.kernels };
    } else |_| {}
    var roles: calls.Roles = .{ .a = a };
    var flow: Flow = .{ .a = a };
    var last_set: []const u8 = "";
    for (ws) |win| {
        if (!std.mem.eql(u8, last_set, win.set)) { // each set ran in its own engine
            roles = .{ .a = a };
            flow.last = .empty;
        }
        last_set = win.set;
        const all = if (win.prefill)
            try block_prefill.emitPrefill(a, &cfg, &w, opts, win.layers, win.n, win.start, true)
        else
            try block.emit(a, &cfg, &w, opts, win.layers, win.n, win.start, true);
        const cs = try calls.launches(a, all);
        var equal: usize = 0;
        var shown: usize = 0;
        const m = @min(cs.len, win.ops.len);
        var g: usize = 0; // the next of `all`: our glue steps and the scopes in emission order
        for (cs[0..m], win.ops[0..m], 0..) |*c, op, i| {
            roles.begin(c.begin);
            while (all[g].glue) : (g += 1) {
                flow.begin(all[g].begin);
                try flow.glueCall(&all[g]);
            }
            flow.begin(all[g].begin);
            g += 1;
            try flow.call(c, op, log, try std.fmt.allocPrint(a, "{s} {s} #{d}", .{ win.set, win.phase, i }));
            const vwhy = if (variants) |*v| try v.check(a, c, op) else null;
            if (vwhy) |why| {
                bad += 1;
                if (shown < show) try log.print("  {s} {s} #{d} {s}: variant: {s}\n", .{ win.set, win.phase, i, c.name, why });
                shown += 1;
            }
            if (try calls.check(a, c, op, &roles)) |why| {
                bad += 1;
                if (shown < show) try log.print("  {s} {s} #{d} {s}: {s}\n", .{ win.set, win.phase, i, c.name, why });
                shown += 1;
            } else equal += 1;
        }
        if (cs.len != win.ops.len) bad += 1;
        try log.print("{s} {s} (layers {any}, n {d}, from {d}): {d}/{d} calls equal, {d} emitted\n", .{ win.set, win.phase, win.layers, win.n, win.start, equal, win.ops.len, cs.len });
        try log.flush();
    }
    try log.print("dataflow: {d} first reads hold another role's bytes; roles written between calls by uncaptured ops (reads):", .{flow.miswired});
    for (flow.glue.keys(), flow.glue.values()) |k, v| try log.print(" {s}={d}", .{ k, v });
    if (variants) |v| try log.print("\nTriton variants: {d} calls, {d} picked wrong", .{ v.checked, v.wrong }) else try log.print("\nTriton variants: not checked (no aot.json with constexprs)", .{});
    try log.print("\nread-only first touches (inputs glue fills; in-place updates not listed):", .{});
    for (flow.inputs.keys(), flow.inputs.values()) |k, v| try log.print(" {s}={d}", .{ k, v });
    try log.print("\n", .{});
    return bad + flow.miswired;
}
