//! Vulkan stage 1 bring-up: typed views, arena, BF16 cast/copy, quantized embedding and BF16 Q4
//! affine g64, checked on a real GPU. Exit 0 pass, 1 fail, 2 usage, 124 fence timeout.
//! Operator bring-up only: no model, no performance claim.
const std = @import("std");
const vk = @import("vulkan");
const shaders = @import("stage1_shaders");
const fx = @import("stage1_fixture.zig");
const c = vk.c;
const contract = @import("contract");
const cx = contract.spec; // toBf16, fromBf16, boundExact, the C constants

const usage = "usage: tf-stage1-probe [--vendor ID] [--timeout-ms MS] | --help\n" ++
    "  --help          show this text and exit without opening a loader or device\n" ++
    "  --vendor ID     refuse a device whose Vulkan vendorID differs (decimal or 0x hex; no default)\n" ++
    "  --timeout-ms MS bound on every fence wait, 1 to 600000 (default 10000); a timeout ends the process\n";

/// Milliseconds on the awake clock since main started, printed on every milestone.
const Elapsed = struct {
    io: std.Io,
    start: std.Io.Timestamp,

    fn ms(e: Elapsed) i64 {
        return e.start.durationTo(std.Io.Clock.awake.now(e.io)).toMilliseconds();
    }
};

const repeats = 3;

pub fn main(init: std.process.Init) !u8 {
    const clock: Elapsed = .{ .io = init.io, .start = std.Io.Clock.awake.now(init.io) };
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    for (args[1..]) |arg| if (std.mem.eql(u8, arg, "--help")) {
        std.debug.print(usage, .{});
        return 0;
    };
    var vendor: ?u32 = null;
    var timeout_ms: u64 = 10_000;
    var i: usize = 1;
    while (i < args.len) : (i += 2) {
        const value = if (i + 1 < args.len) args[i + 1] else "";
        if (std.mem.eql(u8, args[i], "--vendor")) {
            vendor = std.fmt.parseInt(u32, value, 0) catch null;
            if (vendor == null) break;
        } else if (std.mem.eql(u8, args[i], "--timeout-ms")) {
            timeout_ms = std.fmt.parseInt(u64, value, 10) catch 0;
            if (timeout_ms == 0 or timeout_ms > 600_000) break;
        } else break;
    } else return if (try probe(init.gpa, clock, vendor, timeout_ms * std.time.ns_per_ms)) 0 else 1;
    std.debug.print(usage, .{});
    return 2;
}

fn probe(a: std.mem.Allocator, clock: Elapsed, vendor: ?u32, timeout: u64) !bool {
    var rt = try vk.Runtime.open();
    defer rt.close();
    var dev = try vk.Device.open(&rt, .{ .features = .{ .storage16 = true, .int16 = true } });
    defer dev.close();
    const pr = dev.properties;
    const name = std.mem.sliceTo(&pr.deviceName, 0);
    const types = [_][]const u8{ "other", "integrated", "discrete", "virtual", "cpu" };
    const device_kind = if (pr.deviceType < types.len) types[pr.deviceType] else "unknown";
    const v = pr.apiVersion;
    const driver = .{ std.mem.sliceTo(&dev.v12.driverName, 0), std.mem.sliceTo(&dev.v12.driverInfo, 0) };
    std.debug.print("MILESTONE device-found elapsed_ms={d} name=\"{s}\" vendor=0x{x:0>4} device=0x{x:0>4} type={s} " ++
        "api={d}.{d}.{d} driver=\"{s}\" info=\"{s}\" subgroup={d} storage16={} int16={} " ++
        "max-storage-range={d} max-buffer-size={d} fence-timeout={d}ms\n", .{
        clock.ms(),
        name,
        pr.vendorID,
        pr.deviceID,
        device_kind,
        v >> 22 & 0x7f,
        v >> 12 & 0x3ff,
        v & 0xfff,
    } ++ driver ++ .{ dev.v11.subgroupSize, dev.features.storage16, dev.features.int16, dev.properties.limits.maxStorageBufferRange, dev.v13.maxBufferSize, timeout / std.time.ns_per_ms });
    if (pr.deviceType == c.VK_PHYSICAL_DEVICE_TYPE_CPU) {
        std.debug.print("FAIL device: a CPU (software) Vulkan device is never a GPU result\n", .{});
        return false;
    }
    if (vendor) |want| if (pr.vendorID != want) {
        std.debug.print("FAIL device: vendor 0x{x:0>4} is not the requested 0x{x:0>4}\n", .{ pr.vendorID, want });
        return false;
    };

    var cmds = try vk.Commands.init(&dev);
    defer cmds.deinit();
    var staging = try vk.Buffer.init(&dev, .staging, 64 << 10);
    defer staging.deinit();
    const sh = vk.ops.Shaders{
        .qmm_bf16 = &shaders.qmm_bf16,
        .dequant_bf16 = &shaders.dequant_bf16,
        .gather_u32 = &shaders.gather_u32,
        .gather_u16 = &shaders.gather_u16,
        .cast_bf16_f32 = &shaders.cast_bf16_f32,
        .cast_f32_bf16 = &shaders.cast_f32_bf16,
        .copy_u32 = &shaders.copy_u32,
        .copy_u16 = &shaders.copy_u16,
    };
    var o = try vk.ops.Ops.init(&dev, sh);
    defer o.deinit();
    var scratch = try vk.arena.Arena.init(&dev, 1 << 20);
    defer scratch.deinit();

    var passed: usize = 0;
    var total: usize = 0;
    inline for (.{ "cast-f32-bf16", "cast-bf16-f32" }) |gate| {
        total += 1;
        passed += @intFromBool(try castGate(gate, a, clock, &dev, &cmds, staging, &o, timeout));
    }
    inline for (.{ vk.tensor.Kind.u32, .u16 }) |kind| {
        total += 1;
        passed += @intFromBool(try copyGate(kind, a, clock, &dev, &cmds, staging, &o, timeout));
    }
    inline for (.{ vk.tensor.Kind.u32, .bf16 }) |kind| {
        total += 1;
        passed += @intFromBool(try gatherGate(kind, a, clock, &dev, &cmds, staging, &o, timeout));
    }
    total += 1;
    passed += @intFromBool(try checkIdsGate(clock));
    for (fx.embed_cases) |case| {
        total += 1;
        passed += @intFromBool(try embedGate(a, clock, &dev, &cmds, staging, &o, &scratch, case, timeout));
    }
    for (fx.cases, 0..) |case, n| {
        total += 1;
        passed += @intFromBool(try projectGate(a, clock, &dev, &cmds, staging, &o, &scratch, case, 0x57a6 + n, timeout));
    }
    std.debug.print("RESULT stage1 elapsed_ms={d} on \"{s}\": passed={d} total={d}\n", .{ clock.ms(), name, passed, total });
    return passed == total;
}

fn uploaded(_: std.mem.Allocator, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, bytes: []const u8, timeout: u64) !vk.Buffer {
    var b = try vk.Buffer.init(dev, .device, bytes.len);
    errdefer b.deinit();
    try vk.orExit(vk.upload(cmds, b, 0, staging, bytes, timeout));
    return b;
}

fn fetch(cmds: *vk.Commands, staging: vk.Buffer, b: vk.Buffer, out: []u8, timeout: u64) !void {
    try vk.orExit(vk.download(cmds, b, 0, staging, out, timeout));
}

/// Every stage 1 vector plus randoms: exact bf16 bits out, exact f32 back, guard tails untouched.
fn castGate(comptime gate: []const u8, a: std.mem.Allocator, clock: Elapsed, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, o: *vk.ops.Ops, timeout: u64) !bool {
    const to_bf16 = std.mem.eql(u8, gate, "cast-f32-bf16");
    var prng = std.Random.DefaultPrng.init(0x57a1);
    const rng = prng.random();
    const randoms = 128;
    const n = fx.bf16_vectors.len + randoms;
    var ins: std.ArrayList(u8) = .empty;
    defer ins.deinit(a);
    const wants = try a.alloc(u16, n); // the bf16 bits each element must produce or come from
    defer a.free(wants);
    for (0..n) |i| {
        const x: f32 = if (i < fx.bf16_vectors.len) fx.bf16_vectors[i].in else (rng.float(f32) * 2 - 1) * std.math.pow(f32, 2, @floatFromInt(@as(i32, @intCast(i % 41)) - 20));
        wants[i] = if (i < fx.bf16_vectors.len) fx.bf16_vectors[i].want else cx.toBf16(x);
        if (to_bf16) {
            try ins.appendSlice(a, &std.mem.toBytes(x));
        } else {
            try ins.appendSlice(a, &std.mem.toBytes(wants[i]));
        }
    }
    var in = try uploaded(a, dev, cmds, staging, ins.items, timeout);
    defer in.deinit();
    const out_esz: usize = if (to_bf16) 2 else 4;
    var out_buf = try vk.Buffer.init(dev, .device, (n + fx.tail) * out_esz);
    defer out_buf.deinit();
    const fill = try a.alloc(u8, (n + fx.tail) * out_esz);
    defer a.free(fill);
    @memset(fill, 0xa5);
    try vk.orExit(vk.upload(cmds, out_buf, 0, staging, fill, timeout));
    const in_kind: vk.tensor.Kind = if (to_bf16) .f32 else .bf16;
    const out_kind: vk.tensor.Kind = if (to_bf16) .bf16 else .f32;
    const in_view = try vk.tensor.View.init(&in, in_kind, 0, n);
    const out_view = try vk.tensor.View.init(&out_buf, out_kind, 0, n);
    if (to_bf16) {
        try vk.orExit(o.castF32ToBf16(cmds, out_view, in_view, timeout));
    } else {
        try vk.orExit(o.castBf16ToF32(cmds, out_view, in_view, timeout));
    }
    const words_out: usize = if (to_bf16) 1 else 2; // u16 words per output element
    const back = try a.alloc(u16, n * words_out);
    defer a.free(back);
    try fetch(cmds, staging, out_buf, std.mem.sliceAsBytes(back), timeout);
    for (0..n) |i| {
        const want_bits: u32 = @as(u32, wants[i]) << (if (to_bf16) @as(u5, 0) else 16);
        const got_bits: u32 = if (to_bf16) back[i] else @as(u32, back[2 * i]) | @as(u32, back[2 * i + 1]) << 16;
        if (got_bits != want_bits) {
            std.debug.print("FAIL {s}: element {d} read 0x{x:0>8}, want 0x{x:0>8}\n", .{ gate, i, got_bits, want_bits });
            return false;
        }
    }
    const tail_words = try a.alloc(u16, fx.tail * words_out);
    defer a.free(tail_words);
    try fetch(cmds, staging, out_buf, std.mem.sliceAsBytes(tail_words), timeout);
    for (tail_words) |g| if (g != 0xa5a5) {
        std.debug.print("FAIL {s}: guard tail clobbered\n", .{gate});
        return false;
    };
    std.debug.print("MILESTONE {s} elapsed_ms={d} elements={d} vectors={d} randoms={d} exact bits, tails unchanged\n", .{ gate, clock.ms(), n, fx.bf16_vectors.len, randoms });
    return true;
}

/// A contiguous same-kind copy at element offsets: exact bytes, both guard tails and the source unchanged.
fn copyGate(comptime kind: vk.tensor.Kind, a: std.mem.Allocator, clock: Elapsed, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, o: *vk.ops.Ops, timeout: u64) !bool {
    const gate = if (kind == .u32) "copy-u32" else "copy-u16";
    const elems: usize = if (kind == .u32) fx.copy_words else fx.copy_half_elements;
    const src_at: usize = if (kind == .u32) fx.copy_u32_src else fx.copy_u16_src;
    const dst_at: usize = if (kind == .u32) fx.copy_u32_dst else fx.copy_u16_dst;
    const esz = kind.size();
    var prng = std.Random.DefaultPrng.init(0x57a2 + esz);
    const rng = prng.random();
    const data = try a.alloc(u8, elems * esz);
    defer a.free(data);
    rng.bytes(data);
    const poison: u8 = 0xa5;
    var src = try poisonedBuf(a, dev, cmds, staging, src_at * esz + data.len, poison, timeout);
    defer src.buf.deinit();
    try vk.orExit(vk.upload(cmds, src.buf, src_at * esz, staging, data, timeout));
    var dst = try poisonedBuf(a, dev, cmds, staging, dst_at * esz + data.len, poison, timeout);
    defer dst.buf.deinit();
    const src_view = try vk.tensor.View.init(&src.buf, kind, src_at * esz, elems);
    const dst_view = try vk.tensor.View.init(&dst.buf, kind, dst_at * esz, elems);
    try vk.orExit(o.copy(cmds, dst_view, src_view, timeout));
    const back = try a.alloc(u8, dst.buf.len);
    defer a.free(back);
    try fetch(cmds, staging, dst.buf, back, timeout);
    if (std.mem.indexOfDiff(u8, back[dst_at * esz ..][0..data.len], data)) |at| {
        std.debug.print("FAIL {s}: first differing byte at element offset {d}\n", .{ gate, at / esz });
        return false;
    }
    for (back, 0..) |byte, i| {
        if (i >= dst_at * esz and i < dst_at * esz + data.len) continue;
        if (byte != poison) {
            std.debug.print("FAIL {s}: guard clobbered at byte {d}\n", .{ gate, i });
            return false;
        }
    }
    const src_back = try a.alloc(u8, src.buf.len);
    defer a.free(src_back);
    try fetch(cmds, staging, src.buf, src_back, timeout);
    if (std.mem.indexOfDiff(u8, src_back[src_at * esz ..][0..data.len], data) != null or
        std.mem.indexOfNone(u8, src_back[0 .. src_at * esz], &.{poison}) != null or
        std.mem.indexOfNone(u8, src_back[src_at * esz + data.len ..], &.{poison}) != null)
    {
        std.debug.print("FAIL {s}: the source view changed\n", .{gate});
        return false;
    }
    std.debug.print("MILESTONE {s} elapsed_ms={d} elements={d} src_off={d} dst_off={d} exact bytes, guards and source unchanged\n", .{ gate, clock.ms(), elems, src_at, dst_at });
    return true;
}

const Poisoned = struct { buf: vk.Buffer };

fn poisonedBuf(a: std.mem.Allocator, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, len: usize, poison: u8, timeout: u64) !Poisoned {
    var b = try vk.Buffer.init(dev, .device, len);
    errdefer b.deinit();
    const fill = try a.alloc(u8, len);
    defer a.free(fill);
    @memset(fill, poison);
    try vk.orExit(vk.upload(cmds, b, 0, staging, fill, timeout));
    return .{ .buf = b };
}

/// Selected and repeated valid ids gather exact table rows; an out-of-range id gathers to zero rows.
fn gatherGate(comptime kind: vk.tensor.Kind, a: std.mem.Allocator, clock: Elapsed, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, o: *vk.ops.Ops, timeout: u64) !bool {
    const gate = if (kind == .u32) "gather-u32" else "gather-bf16";
    const esz = kind.size();
    const rows: u32 = 6;
    const cols: usize = 10;
    const tokens: usize = 4;
    const ids = [_]u32{ 0, 5, 5, 2 }; // selected, with a repeat
    var prng = std.Random.DefaultPrng.init(0x57a3 + esz);
    const rng = prng.random();
    const cell = try a.alloc(u8, cols * esz);
    defer a.free(cell);
    const table = try a.alloc(u8, rows * cols * esz);
    defer a.free(table);
    for (0..rows) |r| {
        rng.bytes(cell);
        @memcpy(table[r * cols * esz ..][0 .. cols * esz], cell);
    }
    var table_buf = try uploaded(a, dev, cmds, staging, table, timeout);
    defer table_buf.deinit();
    const oor = [_]u32{rows}; // out of range: the host check refuses it
    var oor_buf = try uploaded(a, dev, cmds, staging, std.mem.sliceAsBytes(&oor), timeout);
    defer oor_buf.deinit();
    var ids_buf = try uploaded(a, dev, cmds, staging, std.mem.sliceAsBytes(ids[0..]), timeout);
    defer ids_buf.deinit();
    const out_bytes = try a.alloc(u8, tokens * cols * esz + fx.tail * esz);
    defer a.free(out_bytes);
    @memset(out_bytes, 0xa5);
    var out_buf = try uploaded(a, dev, cmds, staging, out_bytes, timeout);
    defer out_buf.deinit();
    const out_view = try vk.tensor.View.init(&out_buf, kind, 0, tokens * cols);
    const table_view = try vk.tensor.View.init(&table_buf, kind, 0, rows * cols);
    const ids_view = try vk.tensor.View.init(&ids_buf, .u32, 0, tokens);
    try vk.orExit(o.gather(cmds, out_view, table_view, ids_view, rows, timeout));
    try fetch(cmds, staging, out_buf, out_bytes, timeout);
    for (ids, 0..) |id, t| {
        const want = table[@as(usize, id) * cols * esz ..][0 .. cols * esz];
        const got = out_bytes[t * cols * esz ..][0 .. cols * esz];
        if (std.mem.indexOfDiff(u8, got, want)) |at| {
            std.debug.print("FAIL {s}: token {d} (id {d}) differs at byte {d}\n", .{ gate, t, id, at });
            return false;
        }
    }
    for (out_bytes[tokens * cols * esz ..]) |byte| if (byte != 0xa5) {
        std.debug.print("FAIL {s}: guard tail clobbered\n", .{gate});
        return false;
    };
    // the out-of-range id must gather zeros, never table row 0
    const oor_view = try vk.tensor.View.init(&oor_buf, .u32, 0, 1);
    try vk.orExit(o.gather(cmds, out_view, table_view, oor_view, rows, timeout));
    try fetch(cmds, staging, out_buf, out_bytes, timeout);
    for (out_bytes[0 .. cols * esz]) |byte| if (byte != 0) {
        std.debug.print("FAIL {s}: an out-of-range id gathered nonzero bytes\n", .{gate});
        return false;
    };
    std.debug.print("MILESTONE {s} elapsed_ms={d} rows={d} cols={d} tokens={d} rows exact incl repeat, out-of-range id gathers zeros, tails unchanged\n", .{ gate, clock.ms(), rows, cols, tokens });
    return true;
}

/// Host-side token validation: ids inside rows pass, a boundary id is refused before any upload.
fn checkIdsGate(clock: Elapsed) !bool {
    const rows: u64 = 8;
    vk.ops_q4.checkIds(&.{ 0, 7, 3, 7 }, rows) catch |e| {
        std.debug.print("FAIL check-ids: valid ids refused with {s}\n", .{@errorName(e)});
        return false;
    };
    const bad = [_]u32{ 0, 8 }; // rows itself is the first out-of-range id
    if (vk.ops_q4.checkIds(&bad, rows)) |_| {
        std.debug.print("FAIL check-ids: id {d} of {d} rows passed\n", .{ bad[1], rows });
        return false;
    } else |_| {}
    std.debug.print("MILESTONE check-ids elapsed_ms={d} valid ids pass, boundary id refused before upload\n", .{clock.ms()});
    return true;
}

/// The quantized embedding: packed rows gathered by ids then dequantized to bf16 within the
/// contract's bf16 output bound, with the table unchanged.
fn embedGate(a: std.mem.Allocator, clock: Elapsed, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, o: *vk.ops.Ops, scratch: *vk.arena.Arena, case: fx.EmbedCase, timeout: u64) !bool {
    var t = try fx.EmbedTable.init(a, case, 0x57a4);
    defer t.deinit();
    var w_buf = try uploaded(a, dev, cmds, staging, std.mem.sliceAsBytes(t.w), timeout);
    defer w_buf.deinit();
    var s_buf = try uploaded(a, dev, cmds, staging, std.mem.sliceAsBytes(t.scales), timeout);
    defer s_buf.deinit();
    var b_buf = try uploaded(a, dev, cmds, staging, std.mem.sliceAsBytes(t.biases), timeout);
    defer b_buf.deinit();
    var id_buf = try uploaded(a, dev, cmds, staging, std.mem.sliceAsBytes(t.ids), timeout);
    defer id_buf.deinit();
    const out_elems = case.tokens * case.k;
    var out = try poisoned16Buf(a, dev, cmds, staging, out_elems, timeout);
    defer out.buf.deinit();
    const out_view = try vk.tensor.View.init(&out.buf, .bf16, 0, out_elems);
    const id_view = try vk.tensor.View.init(&id_buf, .u32, 0, case.tokens);
    const w_view = try vk.tensor.View.init(&w_buf, .u32, 0, t.w.len - fx.tail);
    const s_view = try vk.tensor.View.init(&s_buf, .bf16, 0, t.scales.len - fx.tail);
    const b_view = try vk.tensor.View.init(&b_buf, .bf16, 0, t.biases.len - fx.tail);
    const mark = scratch.mark();
    defer scratch.reset(mark) catch {};
    try vk.orExit(vk.ops_q4.embedding(o, cmds, out_view, id_view, w_view, s_view, b_view, case.k, scratch, timeout));
    const bits = try a.alloc(u16, out_elems + fx.tail);
    defer a.free(bits);
    try fetch(cmds, staging, out.buf, std.mem.sliceAsBytes(bits), timeout);
    for (t.want, 0..) |want, i| {
        const got: f64 = cx.fromBf16(bits[i]);
        const err = @abs(got - want);
        const bound = cx.boundExact(fx.spec, want);
        if (!(std.math.isFinite(err) and err <= bound)) {
            std.debug.print("FAIL embedding case=\"{s}\": token {d} element {d} got {e}, want {e}, err {e} of bound {e}\n", .{ case.name, i / case.k, i % case.k, got, want, err, bound });
            return false;
        }
    }
    for (bits[out_elems..]) |g| if (g != fx.poison16) {
        std.debug.print("FAIL embedding case=\"{s}\": guard tail clobbered\n", .{case.name});
        return false;
    };
    const w_back = try a.alloc(u8, std.mem.sliceAsBytes(t.w).len);
    defer a.free(w_back);
    try fetch(cmds, staging, w_buf, w_back, timeout);
    if (std.mem.indexOfDiff(u8, w_back, std.mem.sliceAsBytes(t.w)) != null) {
        std.debug.print("FAIL embedding case=\"{s}\": the table changed\n", .{case.name});
        return false;
    }
    std.debug.print("MILESTONE embedding elapsed_ms={d} case=\"{s}\" tokens={d} rows={d} k={d} within bf16 bound, ids include a repeat, table and tails unchanged\n", .{ clock.ms(), case.name, case.tokens, case.rows, case.k });
    return true;
}

/// BF16 Q4 projection against FP64: contract-bounded outputs, three byte-identical repeats,
/// unchanged inputs and guards, and planted nonfinite outputs refused by the checker.
fn projectGate(a: std.mem.Allocator, clock: Elapsed, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, o: *vk.ops.Ops, scratch: *vk.arena.Arena, case: fx.Case, seed: u64, timeout: u64) !bool {
    const f = try fx.Fixture.init(a, case, seed);
    defer f.deinit();
    var x_buf = try uploaded(a, dev, cmds, staging, std.mem.sliceAsBytes(f.x), timeout);
    defer x_buf.deinit();
    var w_buf = try uploaded(a, dev, cmds, staging, std.mem.sliceAsBytes(f.w), timeout);
    defer w_buf.deinit();
    var s_buf = try uploaded(a, dev, cmds, staging, std.mem.sliceAsBytes(f.scales), timeout);
    defer s_buf.deinit();
    var b_buf = try uploaded(a, dev, cmds, staging, std.mem.sliceAsBytes(f.biases), timeout);
    defer b_buf.deinit();
    var out = try poisoned16Buf(a, dev, cmds, staging, f.out.len, timeout);
    defer out.buf.deinit();
    const x_view = try vk.tensor.View.init(&x_buf, .bf16, f.case.at.x * 2, f.q.x.len);
    const w_view = try vk.tensor.View.init(&w_buf, .u32, f.case.at.w * 4, f.w.len - f.case.at.w - fx.tail);
    const s_view = try vk.tensor.View.init(&s_buf, .bf16, f.case.at.scales * 2, f.q.scales.len);
    const b_view = try vk.tensor.View.init(&b_buf, .bf16, f.case.at.biases * 2, f.q.biases.len);
    const out_view = try vk.tensor.View.init(&out.buf, .bf16, f.case.at.out * 2, f.want.len);
    const bits = try a.alloc(u16, f.out.len);
    defer a.free(bits);
    var r: fx.Report = .{};
    const first = try a.dupe(u16, bits);
    defer a.free(first);
    const mark = scratch.mark();
    defer scratch.reset(mark) catch {};
    for (0..repeats) |rep| {
        try vk.orExit(vk.upload(cmds, out.buf, 0, staging, std.mem.sliceAsBytes(f.out), timeout));
        try vk.orExit(vk.ops_q4.project(o, cmds, out_view, x_view, w_view, s_view, b_view, case.shape.k, scratch, timeout));
        try fetch(cmds, staging, out.buf, std.mem.sliceAsBytes(bits), timeout);
        r = f.check(bits) catch |e| {
            const m = f.measure(bits);
            std.debug.print("FAIL project case=\"{s}\" repeat={d}: {s}; worst {d:.3} of bound at row {d} col {d}: " ++
                "got {e} want {e}\n", .{ case.name, rep, @errorName(e), m.worst, m.row, m.col, cx.fromBf16(bits[f.case.at.out + m.row * case.shape.n + m.col]), f.want[m.row * case.shape.n + m.col] });
            return false;
        };
        if (rep == 0) {
            @memcpy(first, bits);
        } else if (std.mem.indexOfDiff(u16, first, bits)) |at| {
            std.debug.print("FAIL project case=\"{s}\": repeat {d} differs at element {d}\n", .{ case.name, rep, at });
            return false;
        }
    }
    const back = try a.alloc(u8, @max(std.mem.sliceAsBytes(f.x).len, std.mem.sliceAsBytes(f.w).len, std.mem.sliceAsBytes(f.scales).len, std.mem.sliceAsBytes(f.biases).len));
    defer a.free(back);
    for ([_][]const u8{ std.mem.sliceAsBytes(f.x), std.mem.sliceAsBytes(f.w), std.mem.sliceAsBytes(f.scales), std.mem.sliceAsBytes(f.biases) }, [_]vk.Buffer{ x_buf, w_buf, s_buf, b_buf }, 0..) |bytes, buf, n| {
        try fetch(cmds, staging, buf, back[0..bytes.len], timeout);
        if (std.mem.indexOfDiff(u8, bytes, back[0..bytes.len])) |at| {
            std.debug.print("FAIL project case=\"{s}\": binding {d} changed at byte {d}\n", .{ case.name, n, at });
            return false;
        }
    }
    const planted_bits = [_]u16{ 0x7fc0, 0x7f80, 0xff80 };
    const planted = [_][]const u8{ "nan", "+inf", "-inf" };
    const at = f.case.at.out + r.row * case.shape.n + r.col;
    for (planted_bits, planted) |pb, pname| {
        bits[at] = pb;
        if (f.check(bits)) |_| {
            std.debug.print("FAIL project case=\"{s}\": a planted {s} output passed the check\n", .{ case.name, pname });
            return false;
        } else |e| if (e != error.OutOfBound) {
            std.debug.print("FAIL project case=\"{s}\": planted {s} gave {s}, not OutOfBound\n", .{ case.name, pname, @errorName(e) });
            return false;
        }
    }
    std.debug.print("MILESTONE project elapsed_ms={d} case=\"{s}\" worst={d:.4} of bound at row {d} col {d}, max_abs={e}, " ++
        "repeats byte-identical, inputs and guards unchanged, planted nan/+inf/-inf refused\n", .{ clock.ms(), case.name, r.worst, r.row, r.col, r.max_abs });
    return true;
}

const Poisoned16 = struct { buf: vk.Buffer };

fn poisoned16Buf(a: std.mem.Allocator, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, elems: usize, timeout: u64) !Poisoned16 {
    var out: Poisoned16 = .{ .buf = try vk.Buffer.init(dev, .device, (elems + fx.tail) * 2) };
    errdefer out.buf.deinit();
    const fill = try a.alloc(u16, elems + fx.tail);
    defer a.free(fill);
    @memset(fill, fx.poison16);
    try vk.orExit(vk.upload(cmds, out.buf, 0, staging, std.mem.sliceAsBytes(fill), timeout));
    return out;
}
