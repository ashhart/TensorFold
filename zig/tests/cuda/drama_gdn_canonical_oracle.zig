//! Verification driver: runs the production GDN controller on raw fixtures and writes outputs for a checker.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const drama = @import("drama");

const gdn = drama.backbone.gdn;
const Allocator = std.mem.Allocator;
const Sha256 = std.crypto.hash.sha2.Sha256;

const producer = "tf-drama-gdn-canonical-oracle";
pub const cases_schema = "drama-gdn-canonical-cases/1";
const result_schema = "drama-gdn-canonical-native/1";
const frozen_image_sha256 = "60ec0492bd5926c12bec45b1a555022d080bfe187bb9180d89aec36a0ac43b7f";
const frozen_decode = "drama_gdn_recur";
const frozen_prefill = "drama_gdn_recur_prefill";
pub const dv = 128;
const chunk = gdn.chunk_rows;
const gdn_cases = @import("drama_gdn_oracle_cases.zig");
const gdn_runner = @import("drama_gdn_oracle_runner.zig");
const workspaceCapacity = gdn_cases.workspaceCapacity;
const parseCases = gdn_cases.parseCases;
const LayerMem = gdn_runner.LayerMem;
const GeoMem = gdn_runner.GeoMem;
const OpRecord = gdn_runner.OpRecord;
const Runner = gdn_runner.Runner;
pub const layer_count = 2;
const passes = 2;
pub const guard = 4096; // bytes; a multiple of 256 keeps every placed buffer aligned like cuMemAlloc
pub const y_guard_rows = 2;
pub const fill = 0xff; // bf16 0xFFFF and f32 0xFFFFFFFF are NaN: no finite result has these bytes
const device_limit = 1 << 30;
pub const max_file = 1 << 28;
/// Workspace fields read for the gate probe when present as device addresses: FP32 [capacity rows][nv] each.
pub const probe_fields = [_][]const u8{ "g", "gc", "beta" };
pub const probe_available = blk: {
    for (probe_fields) |f| {
        if (!@hasField(gdn.Workspace, f)) break :blk false;
        if (@FieldType(gdn.Workspace, f) != u64) break :blk false;
    }
    break :blk true;
};

comptime {
    std.debug.assert(chunk == 64);
}

pub const Errors = struct {
    gpa: Allocator,
    json: std.ArrayList(u8) = .empty,
    count: usize = 0,

    pub fn add(e: *Errors, comptime f: []const u8, args: anytype) !void {
        const message = try std.fmt.allocPrint(e.gpa, f, args);
        defer e.gpa.free(message);
        if (e.count < 200) std.debug.print("error: {s}\n", .{message});
        try e.json.appendSlice(e.gpa, if (e.count == 0) "\n    " else ",\n    ");
        try jsonString(&e.json, e.gpa, message);
        e.count += 1;
    }
};

pub fn sha256Hex(bytes: []const u8) [64]u8 {
    var out: [32]u8 = undefined;
    Sha256.hash(bytes, &out, .{});
    return std.fmt.bytesToHex(out, .lower);
}

fn fileSha256(gpa: Allocator, io: std.Io, path: []const u8) ![64]u8 {
    const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    const buffer = try gpa.alloc(u8, 1 << 20);
    defer gpa.free(buffer);
    var hash = Sha256.init(.{});
    var offset: u64 = 0;
    while (true) {
        const got = try file.readPositionalAll(io, buffer, offset);
        hash.update(buffer[0..got]);
        offset += got;
        if (got < buffer.len) break;
    }
    return std.fmt.bytesToHex(hash.finalResult(), .lower);
}

fn jsonString(list: *std.ArrayList(u8), gpa: Allocator, s: []const u8) !void {
    try list.append(gpa, '"');
    for (s) |c| switch (c) {
        '"', '\\' => {
            try list.append(gpa, '\\');
            try list.append(gpa, c);
        },
        0...0x1f => try list.print(gpa, "\\u{x:0>4}", .{c}),
        else => try list.append(gpa, c),
    };
    try list.append(gpa, '"');
}

pub fn nowNs(io: std.Io) i128 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}

pub fn notFill(bytes: []const u8) usize {
    var n: usize = 0;
    for (bytes) |b| {
        if (b != fill) n += 1;
    }
    return n;
}

pub fn nonfiniteBf16(values: []const u16) usize {
    var n: usize = 0;
    for (values) |bits| {
        if ((bits & 0x7f80) == 0x7f80) n += 1;
    }
    return n;
}

pub fn unwrittenBf16(values: []const u16) usize {
    var n: usize = 0;
    for (values) |bits| {
        if (bits == 0xffff) n += 1;
    }
    return n;
}

pub fn nonfiniteF32(bytes: []align(16) const u8) usize {
    var n: usize = 0;
    for (std.mem.bytesAsSlice(u32, bytes)) |bits| {
        if ((bits & 0x7f800000) == 0x7f800000) n += 1;
    }
    return n;
}

pub fn bitsDifferent(comptime T: type, a: []align(16) const u8, b: []align(16) const u8) usize {
    var n: usize = 0;
    for (std.mem.bytesAsSlice(T, a), std.mem.bytesAsSlice(T, b)) |x, y| {
        if (x != y) n += 1;
    }
    return n;
}

pub fn int(v: usize) i32 {
    return @intCast(v);
}

pub fn sameBank(a: ?u1, b: ?u1) bool {
    if (a) |x| {
        if (b) |y| return x == y;
        return false;
    }
    return b == null;
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len != 5) {
        std.debug.print("usage: tf-drama-gdn-canonical-oracle GLUE_FATBIN FROZEN_FATBIN CASES_DIR OUTPUT_DIR\n", .{});
        return 2;
    }
    if (builtin.cpu.arch.endian() != .little) return error.UnsupportedHostEndian;
    const started = nowNs(io);

    // 1. fixtures and images, CPU only
    const cases_dir = try std.Io.Dir.cwd().openDir(io, args[3], .{});
    defer cases_dir.close(io);
    const out = try std.Io.Dir.cwd().openDir(io, args[4], .{});
    defer out.close(io);
    const cs = parseCases(arena, io, cases_dir) catch |err| switch (err) {
        error.BadCases => return 2,
        else => return err,
    };
    const glue_image = try std.Io.Dir.cwd().readFileAllocOptions(io, args[1], arena, .limited(max_file), .@"16", null);
    const frozen_image = try std.Io.Dir.cwd().readFileAllocOptions(io, args[2], arena, .limited(max_file), .@"16", null);
    const glue_sha = sha256Hex(glue_image);
    const frozen_sha = sha256Hex(frozen_image);
    if (!std.mem.eql(u8, &frozen_sha, frozen_image_sha256)) {
        std.debug.print("FROZEN_FATBIN sha256 {s} is not the frozen image {s}\n", .{ frozen_sha[0..], frozen_image_sha256 });
        return 2;
    }
    if (std.mem.eql(u8, &glue_sha, frozen_image_sha256)) {
        std.debug.print("GLUE_FATBIN is the frozen pre-chunk image\n", .{});
        return 2;
    }

    var errors: Errors = .{ .gpa = gpa };
    defer errors.json.deinit(gpa);
    const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, dv))); // backbone.zig's deltaNet scale

    // 2. device memory, once
    _ = try cuda.setAllocationLimit(device_limit);
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var name_buffer: [128]u8 = undefined;
    const device_name = try arena.dupe(u8, try ctx.name(&name_buffer));
    const capability = try ctx.capability();
    const driver_version = try driver.version();
    const usage_start = cuda.usage(true);

    var glue = try cuda.Module.load(&driver, glue_image);
    defer glue.unload();
    const kernels = try gdn.Kernels.init(&glue);
    var frozen = try cuda.Module.load(&driver, frozen_image);
    defer frozen.unload();
    const frozen_decode_fn = try frozen.function(frozen_decode);
    const frozen_prefill_fn = try frozen.function(frozen_prefill);

    const stream_bufs = try arena.alloc([2]cuda.DeviceBuffer, cs.streams.len);
    var streams_made: usize = 0;
    defer for (stream_bufs[0..streams_made]) |*p| {
        p[0].free();
        p[1].free();
    };
    var scratch_len: usize = 0;
    for (cs.streams, stream_bufs) |s, *p| {
        p[0] = try cuda.DeviceBuffer.fromHost(&driver, s.qkv.bytes);
        p[1] = cuda.DeviceBuffer.fromHost(&driver, s.ba.bytes) catch |e| {
            p[0].free();
            return e;
        };
        streams_made += 1;
        scratch_len = @max(scratch_len, s.qkv.bytes.len, s.ba.bytes.len);
    }

    const mems = try arena.alloc(?GeoMem, cs.geometries.len);
    @memset(mems, null);
    defer for (mems) |*slot| if (slot.*) |*m| {
        m.a_log.free();
        m.dt_bias.free();
        for (&m.layers) |*lm| {
            lm.region.free();
            lm.old.free();
        }
        m.ws.free();
        m.y.free();
        m.y_old.free();
        m.y_base.free();
        m.base_state.free();
        slot.* = null;
    };
    var geometry_json: std.ArrayList(u8) = .empty;
    defer geometry_json.deinit(gpa);
    for (cs.geometries, mems) |g, *slot| {
        if (!g.used) continue;
        const sb = g.stateBytes();
        const y_bytes = (g.max_rows + 2 * y_guard_rows) * g.rowBytes();
        var m: GeoMem = undefined;
        m.a_log = try cuda.DeviceBuffer.fromHost(&driver, g.a_log.bytes);
        m.dt_bias = try cuda.DeviceBuffer.fromHost(&driver, g.dt_bias.bytes);
        for (&m.layers) |*lm| {
            lm.bank_bytes = sb;
            lm.qkv_cache_bytes = g.qkvCacheBytes();
            lm.ba_cache_bytes = g.baCacheBytes();
            lm.region = try cuda.DeviceBuffer.alloc(&driver, LayerMem.total(sb, lm.qkv_cache_bytes, lm.ba_cache_bytes));
            try lm.region.fill8(fill, null);
            lm.old = try cuda.DeviceBuffer.alloc(&driver, sb);
            const o = lm.offsets();
            lm.state = .{ .banks = .{ lm.region.ptr + o[0], lm.region.ptr + o[1] }, .qkv_cache = lm.region.ptr + o[2], .ba_cache = lm.region.ptr + o[3] };
        }
        m.ws = try cuda.DeviceBuffer.alloc(&driver, guard + g.ws_bytes + guard);
        m.y = try cuda.DeviceBuffer.alloc(&driver, y_bytes);
        m.y_old = try cuda.DeviceBuffer.alloc(&driver, g.max_rows * g.rowBytes());
        m.y_base = try cuda.DeviceBuffer.alloc(&driver, g.max_rows * g.rowBytes());
        m.base_state = try cuda.DeviceBuffer.alloc(&driver, sb);
        slot.* = m;
        scratch_len = @max(scratch_len, y_bytes, sb, m.ws.len);
        try geometry_json.print(gpa, "{s}\n    {{\"name\": \"{s}\", \"nk\": {d}, \"nv\": {d}, \"dv\": {d}, \"width\": {d}, \"state_bytes\": {d}, \"qkv_cache_bytes\": {d}, \"ba_cache_bytes\": {d}, \"layer_region_bytes\": {d}, \"layers\": {d}, \"workspace_allocated_bytes\": {d}, \"workspace_largest_layout_bytes\": {d}, \"output_rows_allocated\": {d}, \"guard_bytes\": {d}, \"output_guard_rows\": {d}}}", .{
            if (geometry_json.items.len == 0) "" else ",", g.name, g.nk, g.nv, dv, g.width(), sb, g.qkvCacheBytes(), g.baCacheBytes(), m.layers[0].region.len, layer_count, m.ws.len, g.ws_bytes, g.max_rows, guard, y_guard_rows,
        });
    }
    try ctx.synchronize();
    const usage_after_setup = cuda.usage(false);
    const mem_after_setup = try ctx.memInfo();

    const scratch = try gpa.alignedAlloc(u8, .@"16", scratch_len);
    defer gpa.free(scratch);
    const scratch2 = try gpa.alignedAlloc(u8, .@"16", scratch_len);
    defer gpa.free(scratch2);

    var runner: Runner = .{
        .gpa = gpa,
        .io = io,
        .d = &driver,
        .stream = stream,
        .kernels = &kernels,
        .frozen_decode_fn = frozen_decode_fn,
        .frozen_prefill_fn = frozen_prefill_fn,
        .cs = &cs,
        .stream_bufs = stream_bufs,
        .out = out,
        .errors = &errors,
        .scale = scale,
        .scratch = scratch,
        .scratch2 = scratch2,
    };

    // 3. cases, two passes each
    var cases_json: std.ArrayList(u8) = .empty;
    defer cases_json.deinit(gpa);
    var path_buffer: [256]u8 = undefined;
    var max_ops: usize = 0;
    for (cs.cases) |c| max_ops = @max(max_ops, c.ops.len);
    const records = [passes][]OpRecord{ try gpa.alloc(OpRecord, max_ops), try gpa.alloc(OpRecord, max_ops) };
    defer for (records) |r| gpa.free(r);
    var total_ops: usize = 0;
    for (cs.cases) |c| {
        const m = &mems[c.geometry].?;
        const g = cs.geometries[c.geometry];
        try out.createDir(io, c.name, .default_dir);
        var ys: [layer_count]std.ArrayList(u8) = @splat(.empty);
        var old_ys: [layer_count]std.ArrayList(u8) = @splat(.empty);
        var base_ys: [layer_count]std.ArrayList(u8) = @splat(.empty);
        defer for (&ys, &old_ys, &base_ys) |*a, *b, *e| {
            a.deinit(gpa);
            b.deinit(gpa);
            e.deinit(gpa);
        };
        const case_start = nowNs(io);
        for (0..passes) |pass| try runner.runCase(c, m, records[pass][0..c.ops.len], pass, &ys, &old_ys, &base_ys);
        const case_ns = nowNs(io) - case_start;
        total_ops += c.ops.len;
        const usage_now = cuda.usage(false);
        if (usage_now.device != usage_after_setup.device or usage_now.host != usage_after_setup.host)
            try errors.add("{s}: counted allocations changed while running ({d} -> {d} device bytes)", .{ c.name, usage_after_setup.device, usage_now.device });

        // per layer concatenated outputs
        var files_json: std.ArrayList(u8) = .empty;
        defer files_json.deinit(gpa);
        for (0..layer_count) |layer| {
            for ([_]struct { []const u8, *std.ArrayList(u8) }{ .{ "y", &ys[layer] }, .{ "old.y", &old_ys[layer] }, .{ "decbase.y", &base_ys[layer] } }) |entry| {
                const kind, const list = entry;
                if (list.items.len == 0) continue;
                const name = try std.fmt.bufPrint(&path_buffer, "{s}/L{d}.{s}.bf16", .{ c.name, layer, kind });
                try out.writeFile(io, .{ .sub_path = name, .data = list.items, .flags = .{ .exclusive = true } });
                const file_sha = sha256Hex(list.items);
                try files_json.print(gpa, "{s}\"L{d}.{s}.bf16\": {{\"sha256\": \"{s}\", \"rows\": {d}}}", .{ if (files_json.items.len == 0) "" else ", ", layer, kind, file_sha[0..], list.items.len / g.rowBytes() });
            }
        }

        _, const ws_bytes = gdn.Workspace.layout(c.max_rows, g.nk, g.nv);
        try cases_json.print(gpa, "{s}\n    {{\"name\": \"{s}\", \"geometry\": \"{s}\", \"max_rows\": {d}, \"workspace_layout_bytes\": {d}, \"workspace_capacity_rows_contract\": {d}, \"init\": [", .{
            if (cases_json.items.len == 0) "" else ",", c.name, g.name, c.max_rows, ws_bytes, workspaceCapacity(c.max_rows),
        });
        for (c.init, 0..) |iv, i| try cases_json.print(gpa, "{s}\"{s}\"", .{ if (i == 0) "" else ", ", if (iv) |x| cs.inits[x].name else "zero" });
        try cases_json.print(gpa, "], \"wall_s\": {d:.6}, \"files\": {{{s}}}, \"ops\": [", .{ @as(f64, @floatFromInt(case_ns)) / 1e9, files_json.items });
        for (c.ops, 0..) |op, oi| {
            const a = records[0][oi];
            const b = records[1][oi];
            const repeat_equal = a.sameOutputs(b);
            if (!repeat_equal) try errors.add("{s} op {d}: pass 1 did not reproduce pass 0 (outputs, states, metadata or controls)", .{ c.name, oi });
            try cases_json.print(gpa, "{s}\n      {{\"index\": {d}, \"op\": \"{s}\", \"layer\": {d}", .{ if (oi == 0) "" else ",", oi, @tagName(op.kind), op.layer });
            if (op.kind == .reset) {
                try cases_json.print(gpa, ", \"state\": \"{s}\", \"poison_cache\": {}", .{ if (op.init) |x| cs.inits[x].name else "zero", op.poison_cache });
            } else {
                try cases_json.print(gpa, ", \"stream\": \"{s}\", \"start\": {d}, \"rows\": {d}, \"probe\": {}, \"y_sha256\": \"{s}\", \"old_y_sha256\": \"{s}\", \"old_state_sha256\": \"{s}\", \"y_unwritten\": {d}, \"y_nonfinite\": {d}, \"y_guard_before_changed\": {d}, \"y_guard_after_changed\": {d}, \"state_nonfinite\": {d}, \"ws_front_guard_changed\": {d}, \"ws_tail_changed\": {d}, \"inputs_unchanged\": {}, \"elapsed_s\": [{d:.9}, {d:.9}]", .{
                    cs.streams[op.stream].name, op.start, op.rows, op.probe, a.y_sha[0..], a.old_y_sha[0..], a.old_state_sha[0..], a.y_unwritten, a.y_nonfinite, a.y_guard_before_changed, a.y_guard_after_changed, a.state_nonfinite, a.ws_front_guard_changed, a.ws_tail_changed, a.inputs_unchanged and b.inputs_unchanged, @as(f64, @floatFromInt(a.elapsed_ns)) / 1e9, @as(f64, @floatFromInt(b.elapsed_ns)) / 1e9,
                });
                if (op.kind == .decode) try cases_json.print(gpa, ", \"decode_baseline\": {{\"y_sha256\": \"{s}\", \"state_sha256\": \"{s}\", \"y_bits_different\": {d}, \"state_bits_different\": {d}}}", .{ a.base_y_sha[0..], a.base_state_sha[0..], a.base_y_bits_different, a.base_state_bits_different });
                if (op.probe) {
                    try cases_json.print(gpa, ", \"gate_probe\": {{\"available\": {}, \"in_layout\": {}", .{ probe_available, a.probe_in_layout });
                    for (probe_fields, 0..) |f, fi| try cases_json.print(gpa, ", \"{s}\": {{\"sha256\": \"{s}\", \"nonfinite\": {d}}}", .{ f, a.probe_sha[fi][0..], a.probe_nonfinite[fi] });
                    try cases_json.appendSlice(gpa, "}");
                }
            }
            try cases_json.print(gpa, ", \"pending_before\": {d}, \"committed_bank_before\": {d}, \"pending_after\": {d}, \"committed_bank_after\": {d}, \"current_bank\": {s}, \"state_sha256\": \"{s}\", \"layer_guards_changed\": {d}, \"repeat_bit_exact\": {}}}", .{
                a.pending_before, a.committed_before, a.pending_after, a.committed_after, if (a.current_bank) |x| (if (x == 0) "0" else "1") else "null", a.state_sha[0..], a.layer_guards_changed, repeat_equal,
            });
        }
        try cases_json.appendSlice(gpa, "\n    ]}");
        std.debug.print("{s}: ops={d} wall_s={d:.3} errors_so_far={d}\n", .{ c.name, c.ops.len, @as(f64, @floatFromInt(case_ns)) / 1e9, errors.count });
    }
    const usage_end = cuda.usage(false);
    const mem_end = try ctx.memInfo();

    // 4. result.json
    var binary_json: std.ArrayList(u8) = .empty;
    defer binary_json.deinit(gpa);
    if (fileSha256(gpa, io, "/proc/self/exe")) |sha| {
        try binary_json.print(gpa, "\"{s}\"", .{sha[0..]});
    } else |err| {
        std.debug.print("binary hash unavailable: {s}\n", .{@errorName(err)});
        try binary_json.appendSlice(gpa, "null");
    }
    const source_sha = sha256Hex(@embedFile("drama_gdn_canonical_oracle.zig"));
    const evidence_valid = errors.count == 0;
    var mj: std.ArrayList(u8) = .empty;
    defer mj.deinit(gpa);
    try mj.print(gpa, "{{\n  \"schema\": \"{s}\",\n  \"producer\": \"{s}\",\n  \"glue_fatbin\": ", .{ result_schema, producer });
    try jsonString(&mj, gpa, args[1]);
    try mj.print(gpa, ",\n  \"glue_fatbin_sha256\": \"{s}\",\n  \"glue_fatbin_bytes\": {d},\n  \"frozen_fatbin\": ", .{ glue_sha[0..], glue_image.len });
    try jsonString(&mj, gpa, args[2]);
    try mj.print(gpa, ",\n  \"frozen_fatbin_sha256\": \"{s}\",\n  \"cases_dir\": ", .{frozen_sha[0..]});
    try jsonString(&mj, gpa, args[3]);
    try mj.print(gpa, ",\n  \"cases_sha256\": \"{s}\",\n  \"binary_sha256\": {s},\n  \"oracle_source_sha256\": \"{s}\",\n  \"device\": ", .{ cs.sha[0..], binary_json.items, source_sha[0..] });
    try jsonString(&mj, gpa, device_name);
    try mj.print(gpa, ",\n  \"capability\": {d},\n  \"driver_version\": {d},\n  \"scale_f32_bits\": {d},\n  \"chunk_rows\": {d},\n  \"passes\": {d},\n  \"layers\": {d}," ++
        "\n  \"gate_probe_fields_available\": {},\n  \"controller\": \"drama.backbone.gdn: State.resetMetadata/current, Workspace.layout/place, Kernels.init/prefill/decode; frozen controls: {s} (decode baseline, old decode) and {s} (old sequential prefill)\"," ++
        "\n  \"geometries\": [{s}\n  ],\n  \"cases\": [{s}\n  ]," ++
        "\n  \"allocations\": {{\"counted_device_bytes_start\": {d}, \"counted_device_bytes_after_setup\": {d}, \"counted_device_bytes_end\": {d}, \"counted_device_peak_bytes\": {d}, \"counted_pinned_host_bytes_end\": {d}, \"device_limit_bytes\": {d}, \"driver_free_bytes_after_setup\": {d}, \"driver_free_bytes_end\": {d}, \"driver_total_bytes\": {d}, \"host_scratch_bytes\": {d}}}," ++
        "\n  \"ops_per_pass\": {d},\n  \"wall_s\": {d:.6},\n  \"evidence_valid\": {},\n  \"errors\": [{s}\n  ]\n}}\n", .{
        capability,               driver_version,   @as(u32, @bitCast(scale)), chunk,               passes,                                             layer_count,
        probe_available,          frozen_decode,    frozen_prefill,            geometry_json.items, cases_json.items,                                   usage_start.device,
        usage_after_setup.device, usage_end.device, usage_end.peak,            usage_end.host,      @as(u64, device_limit),                             mem_after_setup.free,
        mem_end.free,             mem_end.total,    2 * scratch_len,           total_ops,           @as(f64, @floatFromInt(nowNs(io) - started)) / 1e9, evidence_valid,
        errors.json.items,
    });
    try out.writeFile(io, .{ .sub_path = "result.json", .data = mj.items, .flags = .{ .exclusive = true } });
    std.debug.print("cases={d} ops_per_pass={d} evidence_valid={} errors={d}\n", .{ cs.cases.len, total_ops, evidence_valid, errors.count });
    return if (evidence_valid) 0 else 3;
}
