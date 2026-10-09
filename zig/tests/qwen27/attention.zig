//! Compare native target attention, active raw partials and cache movement against pinned first-party M5 byte fixtures.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const q = @import("attention");
const Ref = @import("attention").Cache;
const FilePin = struct { name: []const u8, sha256: []const u8 };
const Case = struct { name: []const u8, parents: []const []const i32, starts: []const u32, capacities: []const u32, keep: []const []const u32, files: [2]FilePin };
const Manifest = struct { schema: []const u8, cases: []const Case };
const Counts = struct { cases: usize = 0, rows: usize = 0, checked_bytes: usize = 0, raw_partial_bytes: usize = 0, kept_rows: usize = 0, next_rows: usize = 0, two_layer_same_cb_cases: usize = 0, negative_controls: usize = 0 };
const Field = enum { qg, raw_k, v, q_gain, k_gain, norm_q, norm_k, query, key, out };
const Frame = struct {
    buffers: [10]mtl.Buffer,
    rows: u32,
    pub fn init(device: mtl.Device, file: *const core.safetensors.File, rows: u32, next: bool) !Frame {
        const r: usize = rows;
        const sizes = [_]usize{ r * 12288 * 2, r * 1024 * 2, r * 1024 * 2, 512, 512, r * 6144 * 2, r * 1024 * 2, r * 6144 * 2, r * 1024 * 2, r * 6144 * 2 };
        var buffers: [10]mtl.Buffer = undefined;
        var made: usize = 0;
        errdefer for (buffers[0..made]) |buffer| buffer.deinit();
        for (sizes, &buffers) |n, *buffer| {
            buffer.* = try device.buffer(n, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            @memset(buffer.contents()[0..n], 0xa5);
            made += 1;
        }
        const prefix: []const u8 = if (next) "next." else "";
        var name: [48]u8 = undefined;
        for ([_][]const u8{ "qg", "raw_k", "v", "q_gain", "k_gain" }, 0..) |field, i| {
            const actual_prefix = if (i < 3) prefix else "";
            const t = try tensor(file, try std.fmt.bufPrint(&name, "{s}{s}", .{ actual_prefix, field }), .bf16, sizes[i]);
            @memcpy(buffers[i].contents()[0..sizes[i]], t.bytes);
        }
        return .{ .buffers = buffers, .rows = rows };
    }
    fn deinit(f: *Frame) void {
        for (f.buffers) |buffer| buffer.deinit();
    }
    fn get(f: *const Frame, field: Field) @FieldType(q.Preprocess, "qg") {
        return .{ .buffer = f.buffers[@backingInt(field)] };
    }
    fn checkInputs(f: *const Frame, file: *const core.safetensors.File, next: bool, count: *Counts) !void {
        var name: [48]u8 = undefined;
        for ([_][]const u8{ "qg", "raw_k", "v", "q_gain", "k_gain" }, 0..) |field, i| {
            const prefix: []const u8 = if (next and i < 3) "next." else "";
            try checkBuffer(file, try std.fmt.bufPrint(&name, "{s}{s}", .{ prefix, field }), f.buffers[i], count);
        }
    }
    fn preprocess(f: *const Frame, op: *const q.Attention, e: mtl.ComputeEncoder, scratch: *const q.Scratch) !void {
        try op.preprocess(e, .{ .qg = f.get(.qg), .key = f.get(.raw_k), .q_gain = f.get(.q_gain), .k_gain = f.get(.k_gain), .norm_q = f.get(.norm_q), .norm_k = f.get(.norm_k), .query = f.get(.query), .rotated_key = f.get(.key), .positions = scratch.get(.positions), .rows = f.rows, .eps = 1e-6, .theta = 10000000 });
    }
    fn encode(f: *const Frame, op: *const q.Attention, e: mtl.ComputeEncoder, scratch: *const q.Scratch, binding: *const q.Binding, plan: q.Plan, caches: []const Ref, gate: bool) !void {
        try op.encode(e, scratch, binding, plan, f.get(.query), f.get(.key), f.get(.v), f.get(.out), caches, if (gate) f.get(.qg) else null, false);
    }
};
const CacheOwner = struct {
    buffers: [16]mtl.Buffer = undefined,
    caches: [8]Ref = undefined,
    streams: usize = 0,
    fn init(device: mtl.Device, file: *const core.safetensors.File, capacities: []const u32) !CacheOwner {
        var result: CacheOwner = .{};
        var made: usize = 0;
        errdefer for (result.buffers[0..made]) |buffer| buffer.deinit();
        var name: [40]u8 = undefined;
        for (capacities, 0..) |cap, st| {
            const bytes = @as(usize, cap) * 4 * 256 * 2;
            for ([_][]const u8{ "k", "v" }, 0..) |part, kind| {
                const t = try tensor(file, try std.fmt.bufPrint(&name, "cache.{s}{d}", .{ part, st }), .bf16, bytes);
                const at = 2 * st + kind;
                result.buffers[at] = try device.buffer(bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
                @memcpy(result.buffers[at].contents()[0..bytes], t.bytes);
                made += 1;
            }
            result.caches[st] = .{ .keys = .{ .buffer = result.buffers[2 * st] }, .values = .{ .buffer = result.buffers[2 * st + 1] }, .capacity = cap };
            result.streams += 1;
        }
        return result;
    }
    fn deinit(c: *CacheOwner) void {
        for (c.buffers[0 .. c.streams * 2]) |buffer| buffer.deinit();
    }
    fn reset(c: *CacheOwner, file: *const core.safetensors.File) !void {
        var name: [40]u8 = undefined;
        for (0..c.streams) |st| for ([_][]const u8{ "k", "v" }, 0..) |part, kind| {
            const buffer = c.buffers[2 * st + kind];
            const t = try tensor(file, try std.fmt.bufPrint(&name, "cache.{s}{d}", .{ part, st }), .bf16, buffer.length());
            @memcpy(buffer.contents()[0..buffer.length()], t.bytes);
        };
    }
};
fn tensor(file: *const core.safetensors.File, name: []const u8, dtype: core.safetensors.DType, bytes: usize) !core.safetensors.Tensor {
    const t = file.get(name) orelse return error.MissingFixture;
    if (t.dtype != dtype or t.bytes.len != bytes) return error.FixtureShape;
    return t;
}
fn compare(actual: []const u8, expected: []const u8) !void {
    if (actual.len != expected.len) return error.FixtureShape;
    if (std.mem.eql(u8, actual, expected)) return;
    for (actual, expected, 0..) |got, wanted, byte| if (got != wanted) {
        std.debug.print("first differing byte {d}: got {d}, expected {d}\n", .{ byte, got, wanted });
        break;
    };
    return error.AttentionBits;
}
fn checkBuffer(file: *const core.safetensors.File, name: []const u8, buffer: mtl.Buffer, count: *Counts) !void {
    const t = try tensor(file, name, .bf16, buffer.length());
    compare(buffer.contents()[0..buffer.length()], t.bytes) catch |err| {
        std.debug.print("FAIL {s}\n", .{name});
        return err;
    };
    count.checked_bytes += buffer.length();
}
fn checkFrame(file: *const core.safetensors.File, frame: *const Frame, next: bool, gated: bool, count: *Counts) !void {
    var name: [48]u8 = undefined;
    const prefix: []const u8 = if (next) "expected.next." else "expected.";
    for ([_]struct { field: Field, name: []const u8 }{ .{ .field = .norm_q, .name = "norm_q" }, .{ .field = .norm_k, .name = "norm_k" }, .{ .field = .query, .name = "q" }, .{ .field = .key, .name = "k" }, .{ .field = .out, .name = if (gated) "out" else "ungated" } }) |check| {
        try checkBuffer(file, try std.fmt.bufPrint(&name, "{s}{s}", .{ prefix, check.name }), frame.buffers[@backingInt(check.field)], count);
    }
}
fn checkCaches(file: *const core.safetensors.File, caches: *const CacheOwner, prefix: []const u8, count: *Counts) !void {
    var name: [48]u8 = undefined;
    for (0..caches.streams) |st| for ([_][]const u8{ "k", "v" }, 0..) |part, kind| {
        try checkBuffer(file, try std.fmt.bufPrint(&name, "expected.{s}.{s}{d}", .{ prefix, part, st }), caches.buffers[2 * st + kind], count);
    };
}
fn partials(file: *const core.safetensors.File, scratch: *const q.Scratch, plan: q.Plan, count: *Counts) !void {
    for ([_]q.Field{ .poa, .pma, .pla, .pob, .pmb, .plb }) |field| {
        const prefix = field == .poa or field == .pma or field == .pla;
        const width: usize = if (prefix) plan.tiles * 16 else plan.rows * 16;
        const chunks: usize = if (prefix) plan.prefix_chunks else plan.tail_chunks;
        if (chunks == 0) continue;
        const dim: usize = if (field == .poa or field == .pob) 256 else 1;
        const bytes = 4 * chunks * width * dim * 4;
        var name: [32]u8 = undefined;
        const expected = try tensor(file, try std.fmt.bufPrint(&name, "expected.{s}", .{@tagName(field)}), .f32, bytes);
        const got = scratch.get(field).buffer.contents()[0..bytes];
        for (0..plan.streams) |st| {
            const at = 8 + st * 12;
            const active_chunks: usize = @intCast(plan.meta[at + (if (prefix) @as(usize, 1) else 5)]);
            const rows: usize = @intCast(plan.meta[at + 2]);
            const first: usize = @intCast(plan.meta[at + (if (prefix) @as(usize, 3) else 6)]);
            for (0..4) |head| for (0..active_chunks) |chunk| {
                for (0..rows) |row| for (0..6) |group| {
                    const index = (head * chunks + chunk) * width + (if (prefix) first * 16 + row * 6 + group else (first + row) * 16 + group);
                    const begin = index * dim * 4;
                    compare(got[begin..][0 .. dim * 4], expected.bytes[begin..][0 .. dim * 4]) catch |err| {
                        std.debug.print("FAIL {s} stream={d} head={d} chunk={d} row={d} group={d}\n", .{ @tagName(field), st, head, chunk, row, group });
                        return err;
                    };
                    count.raw_partial_bytes += dim * 4;
                };
            };
        }
    }
}
fn terminal(cb: mtl.CommandBuffer, e: mtl.ComputeEncoder) !void {
    e.end();
    cb.commit();
    cb.wait();
    if (cb.status() != .completed or cb.failure() != null) return error.GpuFailed;
}
fn safePath(a: std.mem.Allocator, dir: []const u8, name: []const u8) ![]const u8 {
    if (name.len == 0 or std.fs.path.isAbsolute(name) or std.mem.indexOf(u8, name, "..") != null or std.mem.indexOfScalar(u8, name, '/') != null) return error.FixturePath;
    return std.fs.path.join(a, &.{ dir, name });
}
fn runCase(init: std.process.Init, device: mtl.Device, queue: mtl.Queue, op: *const q.Attention, dir: []const u8, case: Case, count: *Counts) !void {
    const a = init.arena.allocator();
    const streams = case.parents.len;
    if (streams == 0 or streams > 8 or case.starts.len != streams or case.capacities.len != streams or case.keep.len != streams) return error.FixtureTopology;
    var windows: [8]q.planning.Window = undefined;
    var keys: u32 = 0;
    for (case.parents, 0..) |parents, st| {
        if (parents.len == 0 or parents.len > 32 or case.capacities[st] > 2048) return error.FixtureTopology;
        windows[st] = .{ .parents = parents, .start = case.starts[st], .capacity = case.capacities[st] };
        keys = @max(keys, case.capacities[st]);
    }
    var plan = try q.Plan.init(a, windows[0..streams], .{});
    defer plan.deinit();
    const limits: q.Limits = .{ .rows = plan.rows, .streams = plan.streams, .keys = keys };
    var scratch = try q.Scratch.init(device, limits);
    defer scratch.deinit();
    try scratch.upload(plan);
    var files: [2]core.safetensors.File = undefined;
    var frames: [2]Frame = undefined;
    var owners: [2]CacheOwner = undefined;
    var bindings: [2]q.Binding = undefined;
    var nf: usize = 0;
    var nr: usize = 0;
    var nc: usize = 0;
    var nb: usize = 0;
    defer {
        for (bindings[0..nb]) |*binding| binding.deinit();
        for (owners[0..nc]) |*owner| owner.deinit();
        for (frames[0..nr]) |*frame| frame.deinit();
        for (files[0..nf]) |*file| file.close(init.io);
    }
    for (case.files, 0..) |pin, layer| {
        files[layer] = try core.safetensors.File.open(a, init.io, try safePath(a, dir, pin.name));
        nf += 1;
        if (files[layer].map.memory.len > 256 << 20 or pin.sha256.len != 64) return error.FixtureBound;
        var sha: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(files[layer].map.memory, &sha, .{});
        if (!std.mem.eql(u8, &std.fmt.bytesToHex(sha, .lower), pin.sha256)) return error.FixtureDigest;
        frames[layer] = try Frame.init(device, &files[layer], plan.rows, false);
        nr += 1;
        owners[layer] = try CacheOwner.init(device, &files[layer], case.capacities);
        nc += 1;
        bindings[layer] = try q.Binding.init(device, limits);
        nb += 1;
        try bindings[layer].upload(plan, owners[layer].caches[0..streams]);
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        try frames[layer].preprocess(op, e, &scratch);
        try frames[layer].encode(op, e, &scratch, &bindings[layer], plan, owners[layer].caches[0..streams], false);
        try terminal(cb, e);
        try checkFrame(&files[layer], &frames[layer], false, false, count);
        try partials(&files[layer], &scratch, plan, count);
        try checkCaches(&files[layer], &owners[layer], "append", count);
    }
    for (&owners, &files) |*owner, *file| try owner.reset(file);
    const pair = queue.commandBuffer();
    const pair_e = pair.compute(.serial);
    for (&frames, &owners, &bindings) |*frame, *owner, *binding| {
        try frame.preprocess(op, pair_e, &scratch);
        try frame.encode(op, pair_e, &scratch, binding, plan, owner.caches[0..streams], false);
    }
    try terminal(pair, pair_e);
    for (&frames, &owners, &files) |*frame, *owner, *file| {
        try checkFrame(file, frame, false, false, count);
        try checkCaches(file, owner, "append", count);
    }
    count.two_layer_same_cb_cases += 1;
    const gated = queue.commandBuffer();
    const gated_e = gated.compute(.serial);
    for (&frames, &owners, &bindings) |*frame, *owner, *binding| try frame.encode(op, gated_e, &scratch, binding, plan, owner.caches[0..streams], true);
    try terminal(gated, gated_e);
    for (&frames, &files) |*frame, *file| {
        try checkFrame(file, frame, false, true, count);
        try frame.checkInputs(file, false, count);
    }
    if (count.negative_controls == 0) {
        const expected = try tensor(&files[0], "expected.out", .bf16, frames[0].get(.out).buffer.length());
        const wrong = try a.dupe(u8, expected.bytes);
        wrong[0] ^= 1;
        if (compare(frames[0].get(.out).buffer.contents()[0..wrong.len], wrong)) |_| return error.NegativeControlPassed else |err| if (err != error.AttentionBits) return err;
        count.negative_controls = 1;
    }
    var kept: [2]u32 = undefined;
    for (&bindings, &owners, 0..) |*binding, *owner, layer| kept[layer] = try binding.uploadKeep(plan, owner.caches[0..streams], case.keep);
    const keep_cb = queue.commandBuffer();
    const keep_e = keep_cb.compute(.serial);
    for (&bindings, &owners, kept) |*binding, *owner, keep| try op.keep(keep_e, &scratch, binding, keep, owner.caches[0..streams]);
    try terminal(keep_cb, keep_e);
    for (&owners, &files) |*owner, *file| try checkCaches(file, owner, "keep", count);
    count.kept_rows += kept[0] + kept[1];
    const one = [_]i32{-1};
    for (0..streams) |st| windows[st] = .{ .parents = &one, .start = case.starts[st] + @as(u32, @intCast(case.keep[st].len)), .capacity = case.capacities[st] };
    var next_plan = try q.Plan.init(a, windows[0..streams], .{});
    defer next_plan.deinit();
    try scratch.upload(next_plan);
    var next_frames: [2]Frame = undefined;
    var made_next: usize = 0;
    defer for (next_frames[0..made_next]) |*frame| frame.deinit();
    for (&next_frames, &files, &bindings, &owners) |*frame, *file, *binding, *owner| {
        frame.* = try Frame.init(device, file, next_plan.rows, true);
        made_next += 1;
        try binding.upload(next_plan, owner.caches[0..streams]);
    }
    const next_cb = queue.commandBuffer();
    const next_e = next_cb.compute(.serial);
    for (&next_frames, &bindings, &owners) |*frame, *binding, *owner| {
        try frame.preprocess(op, next_e, &scratch);
        try frame.encode(op, next_e, &scratch, binding, next_plan, owner.caches[0..streams], true);
    }
    try terminal(next_cb, next_e);
    for (&next_frames, &files, &owners) |*frame, *file, *owner| {
        try checkFrame(file, frame, true, true, count);
        try frame.checkInputs(file, true, count);
        try checkCaches(file, owner, "next.append", count);
    }
    count.next_rows += next_plan.rows * 2;
    count.rows += plan.rows * 2;
    count.cases += 1;
}
pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len == 2 and std.mem.eql(u8, args[1], "--probe-device")) {
        const device = try mtl.Device.init();
        defer device.deinit();
        std.debug.print("{{\"device\":\"{s}\",\"device_initialized\":true,\"dispatches_run\":false}}\n", .{std.mem.span(device.name())});
        return;
    }
    if (args.len != 4) return error.Usage;
    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, try std.fs.path.join(a, &.{ args[1], "manifest.json" }), a, .limited(1 << 20));
    const manifest = try std.json.parseFromSliceLeaky(Manifest, a, text, .{ .ignore_unknown_fields = true });
    if (!std.mem.eql(u8, manifest.schema, "tensorfold-qwen27-attention/1") or manifest.cases.len != 15) return error.FixtureManifest;
    const source = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], a, .limited(1 << 20));
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    if (!std.mem.startsWith(u8, args[3], "Apple M5 ") or !std.mem.eql(u8, args[3], std.mem.span(device.name()))) return error.DevicePinDiffers;
    const queue = try device.queue();
    defer queue.deinit();
    const library = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
    defer library.deinit();
    var op = try q.Attention.init(device, library, true);
    defer op.deinit();
    var counts: Counts = .{};
    for (manifest.cases) |case| {
        try runCase(init, device, queue, &op, args[1], case, &counts);
        std.debug.print("PASS {s}\n", .{case.name});
    }
    if (counts.cases != 15 or counts.rows != 602 or counts.kept_rows != 256 or counts.next_rows != 52 or counts.two_layer_same_cb_cases != 15 or counts.negative_controls != 1 or counts.raw_partial_bytes == 0 or counts.checked_bytes == 0) return error.FixtureCoverage;
    var manifest_sha: [32]u8 = undefined;
    var source_sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(text, &manifest_sha, .{});
    std.crypto.hash.sha2.Sha256.hash(source, &source_sha, .{});
    const report = try std.fmt.allocPrint(a, "{{\"status\":\"pass\",\"cases\":{d},\"rows\":{d},\"checked_bytes\":{d},\"raw_partial_bytes\":{d},\"kept_rows\":{d},\"next_rows\":{d},\"two_layer_same_cb_cases\":{d},\"negative_controls\":{d},\"fixture_manifest_sha256\":\"{s}\",\"metal_source_sha256\":\"{s}\"}}\n", .{ counts.cases, counts.rows, counts.checked_bytes, counts.raw_partial_bytes, counts.kept_rows, counts.next_rows, counts.two_layer_same_cb_cases, counts.negative_controls, std.fmt.bytesToHex(manifest_sha, .lower), std.fmt.bytesToHex(source_sha, .lower) });
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = "attention-result.json", .data = report });
}
