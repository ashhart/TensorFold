//! Prompt attention's real partials and gated mixer bytes against the hash-pinned parent, no checkpoint.
const std = @import("std");
const mtl = @import("metal");
const q = @import("qwen27");
const attn = q.attention;
const frozen = @import("prompt_causal_source.zig");
const dim = 256;
const capacity = 32768;
const options = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

fn emit(a: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
    for ([_]bool{ true, false }) |tensor| for ([_]bool{ false, true }) |paged| {
        const text = try q.simd_attention.grouped(a, tensor, true, 6, 4, paged);
        defer a.free(text);
        const path = try std.fmt.allocPrint(a, "{s}/{s}{s}.metal", .{ dir, if (tensor) "tensor" else "simd", if (paged) "-paged" else "" });
        defer a.free(path);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = text });
    };
}

fn opInit(a: std.mem.Allocator, device: mtl.Device, tensor: bool, side: usize, control: u8) !attn.Attention {
    const body = try q.simd_attention.grouped(a, tensor, true, 6, 4, false);
    defer a.free(body);
    const text = if (side == 0) try frozen.baseline(a, body) else if (control != 0) try frozen.planted(a, body, control == 2) else try a.dupe(u8, body);
    defer a.free(text);
    const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    defer lib.deinit();
    var op = try attn.Attention.init(device, lib);
    errdefer op.deinit();
    if (op.units != tensor) {
        if (op.prompt) |pipes| for (pipes) |pipe| pipe.deinit();
        op.prompt = null;
        op.prompt = try attn.Attention.promptPipelines(device, lib, tensor);
        op.units = tensor;
    }
    if (op.prompt == null) return error.PromptUnavailable;
    return op;
}

fn fill(xs: []u16, rng: std.Random) void {
    for (xs) |*x| x.* = @intCast(@as(u32, @bitCast(rng.float(f32) * 2 - 1)) >> 16);
}

/// One production append, pack, pick, prompt, merge and gate; the passed buffers include all partials.
fn run(queue: mtl.Queue, op: *attn.Attention, scratch: *attn.Scratch, binding: *attn.Binding, plan: attn.Plan, inputs: [5]mtl.Buffer, caches: []const attn.Cache) !void {
    const cb = queue.commandBuffer();
    const e = cb.compute(.concurrent);
    try op.encode(e, scratch, binding, plan, .{ .buffer = inputs[0] }, .{ .buffer = inputs[1] }, .{ .buffer = inputs[2] }, .{ .buffer = inputs[4] }, caches, .{ .buffer = inputs[3] }, true, false);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.debug.print("FAIL GPU: {s}\n", .{msg});
        return error.GpuFailed;
    }
}

/// PM/PL on every valid packed row; PO only where the merge reads it; final gated bf16 mixer bytes everywhere.
fn one(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, ops: *[2]attn.Attention, tensor: bool, rows: u32, start: u32, poison: bool) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const geometry = attn.planning.Geometry{ .shared_prefix = tensor, .solo_prompt = true };
    const heads = geometry.heads;
    const kvh = geometry.kv_heads;
    const limits = attn.Limits{ .rows = rows, .streams = 1, .keys = capacity, .shared_prefix = tensor, .geometry = geometry };
    var scratch = try attn.Scratch.init(device, limits);
    defer scratch.deinit();
    var binding = try attn.Binding.init(device, limits);
    defer binding.deinit();
    const parents = try a.alloc(i32, rows);
    defer a.free(parents);
    for (parents, 0..) |*p, i| p.* = if (i == 0) -1 else @intCast(i - 1);
    var plan = try attn.Plan.init(a, &.{.{ .parents = parents, .start = start, .capacity = capacity }}, geometry);
    defer plan.deinit();
    try scratch.upload(plan);
    const count = kvh * capacity * dim;
    var cache_buffers: [2]mtl.Buffer = undefined;
    for (&cache_buffers) |*b| b.* = try device.buffer(count * 2, options);
    defer for (cache_buffers) |b| b.deinit();
    const caches = [_]attn.Cache{.{ .keys = .{ .buffer = cache_buffers[0] }, .values = .{ .buffer = cache_buffers[1] }, .capacity = capacity }};
    try binding.upload(plan, &caches);
    var prng = std.Random.DefaultPrng.init(0xc615616 + start);
    const rng = prng.random();
    for (cache_buffers) |b| fill(b.slice(u16, count), rng);
    if (poison) cache_buffers[1].slice(u16, count)[(start + 64) * dim + 7] = 0x7f80;
    var inputs: [5]mtl.Buffer = undefined;
    const sizes = [_]usize{ rows * heads * dim, rows * kvh * dim, rows * kvh * dim, 2 * rows * heads * dim, rows * heads * dim };
    for (&inputs, sizes) |*b, n| b.* = try device.buffer(n * 2, options);
    defer for (inputs) |b| b.deinit();
    fill(inputs[0].slice(u16, sizes[0]), rng);
    fill(inputs[3].slice(u16, sizes[3]), rng);
    for (0..rows) |row| for (0..kvh) |h| for (0..2) |kv| {
        @memcpy(inputs[kv + 1].slice(u16, sizes[kv + 1])[(row * kvh + h) * dim ..][0..dim], cache_buffers[kv].slice(u16, count)[(h * capacity + start + row) * dim ..][0..dim]);
    };
    const fields = if (tensor) [3]attn.Field{ .poa, .pma, .pla } else [3]attn.Field{ .pob, .pmb, .plb };
    const rp = plan.tiles * 16;
    const chunks = (start + rows + 511) / 512;
    const n: usize = @as(usize, kvh) * chunks * rp;
    const old_o = try a.alloc(u32, n * dim);
    defer a.free(old_o);
    const old_m = try a.alloc(u32, n);
    defer a.free(old_m);
    const old_l = try a.alloc(u32, n);
    defer a.free(old_l);
    const old_out = try a.alloc(u16, sizes[4]);
    defer a.free(old_out);
    var finite_rows: usize = 0;
    var empty_rows: usize = 0;
    for (ops, 0..) |*op, side| {
        @memset(scratch.get(fields[0]).buffer.slice(u32, n * dim), 0xdeadbeef);
        @memset(scratch.get(fields[1]).buffer.slice(u32, n), 0xdeadbeef);
        @memset(scratch.get(fields[2]).buffer.slice(u32, n), 0xdeadbeef);
        @memset(inputs[4].slice(u16, sizes[4]), 0xdead);
        try run(queue, op, &scratch, &binding, plan, inputs, &caches);
        const pick = scratch.get(.pick).buffer.slice(u32, 12);
        if ((pick[7] != 0) != poison or (pick[4] != 0) == poison) return error.WrongMaskedSelection;
        const po = scratch.get(fields[0]).buffer.slice(u32, n * dim);
        const pm = scratch.get(fields[1]).buffer.slice(u32, n);
        const pl = scratch.get(fields[2]).buffer.slice(u32, n);
        const out = inputs[4].slice(u16, sizes[4]);
        if (side == 0) {
            @memcpy(old_o, po);
            @memcpy(old_m, pm);
            @memcpy(old_l, pl);
            @memcpy(old_out, out);
            continue;
        }
        for (0..kvh) |hk| for (0..chunks) |c| for (0..rows * geometry.group()) |row| {
            const at = (hk * chunks + c) * rp + row;
            if (old_m[at] != pm[at] or old_l[at] != pl[at]) {
                std.debug.print("FAIL partial h={d} chunk={d} row={d}: PM {x}/{x}, PL {x}/{x}\n", .{ hk, c, row, old_m[at], pm[at], old_l[at], pl[at] });
                return error.PartialDiffers;
            }
            if (std.math.isFinite(@as(f32, @bitCast(pm[at])))) {
                finite_rows += 1;
                if (!std.mem.eql(u32, old_o[at * dim ..][0..dim], po[at * dim ..][0..dim])) return error.NumeratorDiffers;
            } else {
                empty_rows += 1;
                if (pm[at] != 0xff800000 or pl[at] != 0) return error.EmptyPartial;
            }
        };
        if (!std.mem.eql(u16, old_out, out)) return error.MixerDiffers;
    }
    std.debug.print("PASS causal bytes: {s} rows={d} start={d} masked={s}, {d} finite partial rows, {d} empty rows, {d} mixer bytes\n", .{ if (tensor) "tensor" else "simd", rows, start, if (poison) "yes" else "no", finite_rows, empty_rows, sizes[4] * 2 });
}

pub fn main(init: std.process.Init) !void {
    const a = std.heap.page_allocator;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len == 3 and std.mem.eql(u8, args[1], "--emit-sources")) return emit(a, init.io, args[2]);
    var simd = false;
    var control: u8 = 0;
    for (args[1..]) |arg| {
        if (std.mem.eql(u8, arg, "--simd")) simd = true else if (std.mem.eql(u8, arg, "--plant-exit")) control = 1 else if (std.mem.eql(u8, arg, "--plant-cap")) control = 2 else return error.Argument;
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    const tensor = device.tensorUnits() and !simd;
    var ops = [2]attn.Attention{ try opInit(a, device, tensor, 0, 0), try opInit(a, device, tensor, 1, control) };
    defer for (&ops) |*op| op.deinit();
    for ([_]u32{ 0, 519, 8192, 30000 }) |start| for ([_]bool{ false, true }) |poison| {
        try one(a, device, queue, &ops, tensor, if (tensor) 2048 else 512, start, poison);
        if (control != 0) return error.ControlNotDetected;
    };
}
