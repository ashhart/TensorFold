//! EXL3 whole layers: a KDA and a DSA MoE layer of real RED-SNOW through core/exl3_load vs the engine's fixture bytes.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const quant_ref_tests = @import("quant_ref_tests.zig");
const exl3_load = @import("core").exl3_load;
const exl3 = cuda.exl3;
const exl3e = cuda.exl3_experts;
const exl3_upload = cuda.exl3_upload;
const exl3_tests = @import("exl3_tests.zig");
const Run = exl3_tests.Run;
const Rig = quant_ref_tests.RigT;

pub fn run(r: *Rig, state: *Run, dir: []const u8) !void {
    if (std.mem.eql(u8, dir, "none")) {
        std.debug.print("SKIP exl3.layer.python\n", .{});
        return;
    }
    const o = manifest(r, dir) catch |e| {
        std.debug.print("FAIL exl3.layer.python: {t}\n", .{e});
        state.failed = true;
        return;
    };
    const layers = o.get("layers") orelse {
        std.debug.print("SKIP exl3.layer.python: no layers entry\n", .{});
        return;
    };
    const model = o.get("model").?.string;
    var scan = exl3_load.scan(r.gpu.gpa, r.gpu.io, model) catch |e| {
        std.debug.print("FAIL exl3.layer.python: {t}\n", .{e});
        state.failed = true;
        return;
    };
    defer scan.deinit();
    var idx = std.StringHashMapUnmanaged(usize).empty;
    for (scan.groups, 0..) |g, i| try idx.put(r.a, g.prefix, i);
    var dense = try exl3.Dense.load(r.gpu.d);
    defer dense.unload();
    var exp = try exl3e.Experts.load(r.gpu.d);
    defer exp.unload();
    var ok = true;
    layerChecks(r, &scan, &idx, &dense, &exp, dir, layers.array.items) catch |e| {
        state.fail(e, "exl3.layer.python");
        ok = false;
    };
    if (ok) check.pass("exl3.layer.python", .{});
}

fn layerChecks(r: *Rig, scan: *exl3_load.Scan, idx: *std.StringHashMapUnmanaged(usize), dense: *exl3.Dense, exp: *const exl3e.Experts, dir: []const u8, layers: []std.json.Value) !void {
    const t0 = check.now(r.gpu.io);
    var total: usize = 0;
    for (layers) |item| {
        const o = item.object;
        const id = o.get("id").?.string;
        var up: exl3_upload.Uploader = .{ .d = r.gpu.d, .gpa = r.gpu.gpa, .max_buffers = scan.groups.len * 4 + 12 };
        defer up.deinit();
        for (o.get("groups").?.array.items) |gitem| try denseGroup(r, scan, idx, dense, dir, &up, gitem.object);
        try expertLayer(r, scan, idx, exp, dir, &up, o.get("experts").?.object);
        std.debug.print("RESULT exl3.layer.python.{s}: {d:.1} MiB of weights on the device\n", .{ id, @as(f64, @floatFromInt(up.bytes)) / (1 << 20) });
        total += up.bytes;
    }
    const dt = @as(f64, @floatFromInt(check.now(r.gpu.io) - t0)) / 1e9;
    std.debug.print("RESULT exl3.layer.python: {d} layers, {d:.1} MiB uploaded, {d:.2} s\n", .{ layers.len, @as(f64, @floatFromInt(total)) / (1 << 20), dt });
}

/// One non-routed group: checkpoint tensors through the uploader, fixture inputs and bytes through the prompt path.
fn denseGroup(r: *Rig, scan: *exl3_load.Scan, idx: *std.StringHashMapUnmanaged(usize), dense: *exl3.Dense, dir: []const u8, up: *exl3_upload.Uploader, o: std.json.ObjectMap) !void {
    const name = o.get("name").?.string;
    const gi = idx.get(name) orelse return error.TestFailed;
    const g = scan.groups[gi];
    try check.expect(g.halves == 2 * bitsOf(o.get("bits").?) and g.k == @as(usize, @intCast(o.get("k").?.integer)) and
        g.n == @as(usize, @intCast(o.get("n").?.integer)), "exl3.layer.python {s}: the checkpoint group is not the fixture's", .{name});
    const parts = try scan.parts(r.a, gi);
    const w = try up.dense(parts, g);
    const xmap = o.get("x").?.object;
    const ymap = o.get("y").?.object;
    for (o.get("rows").?.array.items) |rv| {
        const m: usize = @intCast(rv.integer);
        const key = try std.fmt.allocPrint(r.a, "{d}", .{m});
        const dx = try r.dev(try readFile(r, dir, xmap.get(key).?.string));
        const want = try readFile(r, dir, ymap.get(key).?.string);
        const dy = try r.zeros(m * g.n * 2);
        try dense.prompt(r.s, w, dx, .bf16, m, dy, .bf16);
        const got = try r.back(dy, want.len);
        try check.sameBytes(try std.fmt.allocPrint(r.a, "exl3.layer.python {s} {d} rows", .{ name, m }), got, want);
    }
}

/// One layer's routed experts: all E experts' projections uploaded, the fixture's routing and inputs per row count.
fn expertLayer(r: *Rig, scan: *exl3_load.Scan, idx: *std.StringHashMapUnmanaged(usize), exp: *const exl3e.Experts, dir: []const u8, up: *exl3_upload.Uploader, o: std.json.ObjectMap) !void {
    const base = o.get("name").?.string;
    const e_count: usize = @intCast(o.get("count").?.integer);
    const dims: usize = @intCast(o.get("dims").?.integer);
    const width: usize = @intCast(o.get("width").?.integer);
    const slots: usize = @intCast(o.get("slots").?.integer);
    var gate = try r.a.alloc(exl3_upload.Proj, e_count);
    var upj = try r.a.alloc(exl3_upload.Proj, e_count);
    var down = try r.a.alloc(exl3_upload.Proj, e_count);
    const shorts = [_][]const u8{ "gate", "up", "down" };
    const parts_names = [_][]const u8{ "gate_proj", "up_proj", "down_proj" };
    for (shorts, parts_names) |short, part| {
        const k2map = o.get(try std.fmt.allocPrint(r.a, "{s}_k2", .{short})).?.array;
        const list: *[]exl3_upload.Proj = switch (short[0]) {
            'g' => &gate,
            'u' => &upj,
            else => &down,
        };
        for (0..e_count) |e| {
            const prefix = try std.fmt.allocPrint(r.a, "{s}.{d}.{s}", .{ base, e, part });
            const gi = idx.get(prefix) orelse return error.TestFailed;
            const g = scan.groups[gi];
            try check.expect(g.halves == bitsOf(k2map.items[e]), "exl3.layer.python {s}: expert {d} is {d} halves, the fixture says {d}", .{ prefix, e, g.halves, bitsOf(k2map.items[e]) });
            const parts = try scan.parts(r.a, gi);
            list.*[e] = .{ .trellis = parts.trellis.bytes, .suh = parts.suh.bytes, .svh = parts.svh.bytes, .k2 = g.halves, .signs = parts.suh_packed };
        }
    }
    const l = try exl3_upload.layer(up, r.a, dims, width, .mul1, @floatCast(o.get("limit").?.float), @intCast(o.get("act_mode").?.integer), gate, upj, down);
    const xmap = o.get("x").?.object;
    const pickmap = o.get("pick").?.object;
    const wtsmap = o.get("wts").?.object;
    const ymap = o.get("y").?.object;
    const outmap = o.get("out").?.object;
    for (o.get("rows").?.array.items) |rv| {
        const m: usize = @intCast(rv.integer);
        const key = try std.fmt.allocPrint(r.a, "{d}", .{m});
        const dx = try r.dev(try readFile(r, dir, xmap.get(key).?.string));
        const dpick = try r.dev(try readFile(r, dir, pickmap.get(key).?.string));
        const dwts = try r.dev(try readFile(r, dir, wtsmap.get(key).?.string));
        const sb = try r.zeros(try exl3e.scratchBytes(l, m, slots));
        const sc = try exl3e.scratch(sb, l, m, slots);
        const out = try r.zeros(m * dims * 4);
        try exp.run(r.s, l, dx, dims, .bf16, dpick, dwts, m, out, sc);
        const got_out = try r.back(out, m * dims * 4);
        try check.sameBytes(try std.fmt.allocPrint(r.a, "exl3.layer.python {s} experts {d} rows out", .{ base, m }), got_out, try readFile(r, dir, outmap.get(key).?.string));
        const got_y = try r.back(sc.y, m * slots * dims * 4);
        try check.sameBytes(try std.fmt.allocPrint(r.a, "exl3.layer.python {s} experts {d} rows y", .{ base, m }), got_y, try readFile(r, dir, ymap.get(key).?.string));
    }
}

fn manifest(r: *Rig, dir: []const u8) !std.json.ObjectMap {
    const text = try std.Io.Dir.cwd().readFileAlloc(r.gpu.io, try std.fs.path.join(r.a, &.{ dir, "manifest.json" }), r.a, .limited(1 << 24));
    const parsed = try std.json.parseFromSlice(std.json.Value, r.a, text, .{});
    return parsed.value.object;
}

fn bitsOf(v: std.json.Value) u32 {
    return switch (v) {
        .integer => |i| @intCast(i),
        .float => |f| @intFromFloat(f),
        else => unreachable,
    };
}

fn readFile(r: *Rig, dir: []const u8, name: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(r.gpu.io, try std.fs.path.join(r.a, &.{ dir, name }), r.a, .limited(1 << 30));
}
