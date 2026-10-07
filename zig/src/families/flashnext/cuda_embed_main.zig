//! One token through the captured _embed cubin, compared with the host dequant of the same row.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");

const embed = flash.embed;

fn shardPath(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) ![]u8 {
    const index = try std.fs.path.join(gpa, &.{ dir, "model.safetensors.index.json" });
    defer gpa.free(index);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, index, gpa, .limited(1 << 26));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    const file = parsed.value.object.get("weight_map").?.object.get(embed.weight_name).?.string;
    return std.fs.path.join(gpa, &.{ dir, file });
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 4) {
        std.debug.print("usage: flashnext-embed <model-dir> <kernel-dir> <token>\n", .{});
        return 2;
    }
    const token = try std.fmt.parseInt(u32, args[3], 10);
    const path = try shardPath(gpa, io, args[1]);
    defer gpa.free(path);
    var mapped = try embed.Mapped.open(gpa, io, path);
    defer mapped.close();
    const weight = mapped.weight;
    const scales = mapped.scales;
    const biases = mapped.biases;

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var table = try embed.Table.upload(&driver, weight, scales, biases);
    defer table.deinit();
    if (token >= table.rows) return error.UnexpectedTensor;
    const host = try gpa.alloc(u16, table.dims);
    defer gpa.free(host);
    try embed.dequant(weight.bytes, scales.bytes, biases.bytes, table.dims, token, host);

    var set = try cuda.aot.Set.load(gpa, io, &driver, ctx.device, args[2]);
    defer set.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var ids = try cuda.DeviceBuffer.fromHost(&driver, std.mem.asBytes(&token));
    defer ids.free();
    var out = try cuda.DeviceBuffer.alloc(&driver, table.dims * 2);
    defer out.free();
    try flash.Tri.embed(.{ .set = &set, .s = stream }, ids.ptr, table.w.ptr, table.s.ptr, table.b.ptr, out.ptr, table.dims, 1, 1);

    const gpu = try gpa.alloc(u8, table.dims * 2);
    defer gpa.free(gpu);
    try stream.synchronize();
    try out.download(0, gpu);

    var mismatch: usize = 0;
    for (host, 0..) |want, i| {
        const got = std.mem.readInt(u16, gpu[2 * i ..][0..2], .little);
        if (got != want) mismatch += 1;
    }
    std.debug.print("token {d} rows {d} dims {d} mismatch {d} head", .{ token, table.rows, table.dims, mismatch });
    for (0..@min(8, table.dims)) |i| std.debug.print(" {x:0>4}", .{std.mem.readInt(u16, gpu[2 * i ..][0..2], .little)});
    std.debug.print("\n", .{});
    if (mismatch != 0) return 1;

    const streams: usize = 4;
    var wide = try cuda.DeviceBuffer.alloc(&driver, streams * table.dims * 2);
    defer wide.free();
    try flash.Tri.embed(.{ .set = &set, .s = stream }, ids.ptr, table.w.ptr, table.s.ptr, table.b.ptr, wide.ptr, table.dims, streams, 1);
    const wide_bytes = try gpa.alloc(u8, streams * table.dims * 2);
    defer gpa.free(wide_bytes);
    try stream.synchronize();
    try wide.download(0, wide_bytes);
    const wide_u16 = try gpa.alloc(u16, streams * table.dims);
    defer gpa.free(wide_u16);
    for (wide_u16, 0..) |*v, i| v.* = std.mem.readInt(u16, wide_bytes[2 * i ..][0..2], .little);

    var copies_differ = false;
    for (1..streams) |s| {
        if (!std.mem.eql(u16, wide_u16[0..table.dims], wide_u16[s * table.dims ..][0..table.dims])) copies_differ = true;
    }
    std.debug.print("embed copies {d} identical {}\n", .{ streams, !copies_differ });

    const nc = table.dims / 256;
    var pss = try cuda.DeviceBuffer.alloc(&driver, nc * streams * 4);
    defer pss.free();
    const tri: flash.Tri = .{ .set = &set, .s = stream };
    try tri.hcWriteback(wide.ptr, wide.ptr, pss.ptr, wide.ptr, "*bf16", wide.ptr, wide.ptr, "*bf16", wide.ptr, "*bf16", table.dims, 1, table.dims, streams, 0, 1, 1, 1);
    const pss_bytes = try gpa.alloc(u8, nc * streams * 4);
    defer gpa.free(pss_bytes);
    try stream.synchronize();
    try pss.download(0, pss_bytes);
    const want = try gpa.alloc(f32, nc * streams);
    defer gpa.free(want);
    try flash.hc.partialSums(wide_u16, table.dims, streams, want);
    var max_ulp: u32 = 0;
    var streams_differ = false;
    for (0..nc) |c| {
        const first = std.mem.readInt(u32, pss_bytes[4 * c * streams ..][0..4], .little);
        for (0..streams) |s| {
            const bits = std.mem.readInt(u32, pss_bytes[4 * (c * streams + s) ..][0..4], .little);
            if (bits != first) streams_differ = true;
            const got: f32 = @bitCast(bits);
            const dist = flash.hc.ulps(got, want[c * streams + s]);
            if (dist > max_ulp) max_ulp = dist;
        }
    }
    std.debug.print("writeback streams identical {} max_ulp {d}\n", .{ !streams_differ, max_ulp });
    if (copies_differ or streams_differ or max_ulp > 4) return 1;

    const norm_mismatch = try runDown(gpa, &driver, &stream, &set, &mapped, wide.ptr, pss.ptr, wide_u16, pss_bytes, table.dims, streams, nc);
    return if (norm_mismatch == 0) 0 else 1;
}

fn runDown(gpa: std.mem.Allocator, driver: *cuda.Driver, stream: *cuda.Stream, set: *cuda.aot.Set, mapped: *const embed.Mapped, h: u64, pss: u64, wide_u16: []const u16, pss_bytes: []const u8, dims: usize, streams: usize, nc: usize) !usize {
    const prefix = "language_model.model.layers.0.attn_hyper_connection.";
    var name: [180]u8 = undefined;
    const down_w = try mapped.lookup(gpa, try std.fmt.bufPrint(&name, "{s}input_mix_weight_down.weight", .{prefix}), .u32);
    const down_s = try mapped.lookup(gpa, try std.fmt.bufPrint(&name, "{s}input_mix_weight_down.scales", .{prefix}), .bf16);
    const down_b = try mapped.lookup(gpa, try std.fmt.bufPrint(&name, "{s}input_mix_weight_down.biases", .{prefix}), .bf16);
    const inj_w = try mapped.lookup(gpa, try std.fmt.bufPrint(&name, "{s}block_inject_weight.weight", .{prefix}), .u32);
    const inj_s = try mapped.lookup(gpa, try std.fmt.bufPrint(&name, "{s}block_inject_weight.scales", .{prefix}), .bf16);
    const inj_b = try mapped.lookup(gpa, try std.fmt.bufPrint(&name, "{s}block_inject_weight.biases", .{prefix}), .bf16);
    const gamma = try mapped.lookup(gpa, try std.fmt.bufPrint(&name, "{s}hc_norm.weight", .{prefix}), .bf16);
    const up_w = try mapped.lookup(gpa, try std.fmt.bufPrint(&name, "{s}input_mix_weight_up.weight", .{prefix}), .u32);
    const up_s = try mapped.lookup(gpa, try std.fmt.bufPrint(&name, "{s}input_mix_weight_up.scales", .{prefix}), .bf16);
    const up_b = try mapped.lookup(gpa, try std.fmt.bufPrint(&name, "{s}input_mix_weight_up.biases", .{prefix}), .bf16);
    const n = down_w.dim(0) + inj_w.dim(0);
    const k8 = down_w.dim(1);
    const kg = k8 / 4;
    if (n != 324 or k8 != 1280 or gamma.dim(0) != streams * dims) return error.UnexpectedTensor;

    const stacked_w = try gpa.alloc(u8, down_w.bytes.len + inj_w.bytes.len);
    defer gpa.free(stacked_w);
    @memcpy(stacked_w[0..down_w.bytes.len], down_w.bytes);
    @memcpy(stacked_w[down_w.bytes.len..], inj_w.bytes);
    const stacked_s = try gpa.alloc(u8, down_s.bytes.len + inj_s.bytes.len);
    defer gpa.free(stacked_s);
    @memcpy(stacked_s[0..down_s.bytes.len], down_s.bytes);
    @memcpy(stacked_s[down_s.bytes.len..], inj_s.bytes);
    const stacked_b = try gpa.alloc(u8, down_b.bytes.len + inj_b.bytes.len);
    defer gpa.free(stacked_b);
    @memcpy(stacked_b[0..down_b.bytes.len], down_b.bytes);
    @memcpy(stacked_b[down_b.bytes.len..], inj_b.bytes);
    const tiled = try flash.hc.tileWords(gpa, stacked_w, n, k8);
    defer gpa.free(tiled);
    const scales_t = try flash.hc.transposeBf16(gpa, stacked_s, n, kg);
    defer gpa.free(scales_t);
    const biases_t = try flash.hc.transposeBf16(gpa, stacked_b, n, kg);
    defer gpa.free(biases_t);
    const scale = try gpa.alloc(f32, gamma.dim(0));
    defer gpa.free(scale);
    for (scale, 0..) |*o, i| o.* = @bitCast(@as(u32, std.mem.readInt(u16, gamma.bytes[2 * i ..][0..2], .little)) << 16);

    var w_b = try cuda.DeviceBuffer.fromHost(driver, tiled);
    defer w_b.free();
    var s_b = try cuda.DeviceBuffer.fromHost(driver, scales_t);
    defer s_b.free();
    var b_b = try cuda.DeviceBuffer.fromHost(driver, biases_t);
    defer b_b.free();
    var scale_b = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(scale));
    defer scale_b.free();
    var norm_b = try cuda.DeviceBuffer.alloc(driver, streams * dims * 2);
    defer norm_b.free();
    var out_b = try cuda.DeviceBuffer.alloc(driver, n * 2);
    defer out_b.free();
    const sk: usize = 32;
    var part_b = try cuda.DeviceBuffer.alloc(driver, sk * n * 4);
    defer part_b.free();
    const tri: flash.Tri = .{ .set = set, .s = stream.* };
    try tri.hcDown(h, pss, scale_b.ptr, norm_b.ptr, w_b.ptr, s_b.ptr, b_b.ptr, out_b.ptr, part_b.ptr, 1e-6, 1, n, k8 * 8, dims, nc, streams, sk);
    const low: usize = 320;
    var act_b = try cuda.DeviceBuffer.alloc(driver, low * 2);
    defer act_b.free();
    var xs_b = try cuda.DeviceBuffer.alloc(driver, (low / 32) * 4);
    defer xs_b.free();
    var inj_out = try cuda.DeviceBuffer.alloc(driver, streams * 2);
    defer inj_out.free();
    try tri.hcReduceAct(part_b.ptr, act_b.ptr, xs_b.ptr, inj_out.ptr, sk, 1, streams, low, n, 1);
    const up_n = up_w.dim(0);
    const up_k8 = up_w.dim(1);
    if (up_n != streams * dims or up_k8 != 40) return error.UnexpectedTensor;
    const up_tiled = try flash.hc.tileWords(gpa, up_w.bytes, up_n, up_k8);
    defer gpa.free(up_tiled);
    const up_scales = try flash.hc.transposeBf16(gpa, up_s.bytes, up_n, up_k8 / 4);
    defer gpa.free(up_scales);
    const up_biases = try flash.hc.transposeBf16(gpa, up_b.bytes, up_n, up_k8 / 4);
    defer gpa.free(up_biases);
    var up_wb = try cuda.DeviceBuffer.fromHost(driver, up_tiled);
    defer up_wb.free();
    var up_sb = try cuda.DeviceBuffer.fromHost(driver, up_scales);
    defer up_sb.free();
    var up_bb = try cuda.DeviceBuffer.fromHost(driver, up_biases);
    defer up_bb.free();
    var mixed_b = try cuda.DeviceBuffer.alloc(driver, dims * 2);
    defer mixed_b.free();
    var xsm_b = try cuda.DeviceBuffer.alloc(driver, (dims / 32) * 4);
    defer xsm_b.free();
    try tri.hcUpmix(act_b.ptr, xs_b.ptr, up_wb.ptr, up_sb.ptr, up_bb.ptr, norm_b.ptr, mixed_b.ptr, xsm_b.ptr, 1, up_n, up_k8 * 8, dims, streams);

    const got = try gpa.alloc(u8, streams * dims * 2);
    defer gpa.free(got);
    try stream.synchronize();
    try norm_b.download(0, got);
    const pss_f = try gpa.alloc(f32, nc * streams);
    defer gpa.free(pss_f);
    for (pss_f, 0..) |*o, i| o.* = @bitCast(std.mem.readInt(u32, pss_bytes[4 * i ..][0..4], .little));
    const want = try gpa.alloc(u16, streams * dims);
    defer gpa.free(want);
    try flash.hc.normed(wide_u16, pss_f, scale, dims, streams, 1e-6, want);
    var mismatch: usize = 0;
    for (want, 0..) |w, i| {
        if (std.mem.readInt(u16, got[2 * i ..][0..2], .little) != w) mismatch += 1;
    }
    var part4: [4]u8 = undefined;
    try part_b.download(0, &part4);
    const gpu_part: f32 = @bitCast(std.mem.readInt(u32, &part4, .little));
    const gpu_norm = try gpa.alloc(u16, 320);
    defer gpa.free(gpu_norm);
    for (gpu_norm, 0..) |*o, i| o.* = std.mem.readInt(u16, got[2 * i ..][0..2], .little);
    const host_part = try flash.hc.firstPartial(gpu_norm, stacked_w[0 .. k8 * 4], stacked_s[0 .. kg * 2], stacked_b[0 .. kg * 2], 10);
    const col_ulp = flash.hc.ulps(gpu_part, host_part);
    std.debug.print("hcdown normed mismatch {d} col0 gpu {d} host {d} ulp {d}\n", .{ mismatch, gpu_part, host_part, col_ulp });
    if (col_ulp > 8) return 1;

    const part_bytes = try gpa.alloc(u8, sk * n * 4);
    defer gpa.free(part_bytes);
    try part_b.download(0, part_bytes);
    const part_f = try gpa.alloc(f32, sk * n);
    defer gpa.free(part_f);
    for (part_f, 0..) |*o, i| o.* = @bitCast(std.mem.readInt(u32, part_bytes[4 * i ..][0..4], .little));
    const act_want = try gpa.alloc(u16, low);
    defer gpa.free(act_want);
    const inj_want = try gpa.alloc(u16, streams);
    defer gpa.free(inj_want);
    const xs_want = try gpa.alloc(f32, low / 32);
    defer gpa.free(xs_want);
    try flash.hc.reduceAct(part_f, sk, n, low, streams, act_want, inj_want, xs_want);
    const act_bytes = try gpa.alloc(u8, low * 2);
    defer gpa.free(act_bytes);
    const inj_bytes = try gpa.alloc(u8, streams * 2);
    defer gpa.free(inj_bytes);
    const xs_bytes = try gpa.alloc(u8, (low / 32) * 4);
    defer gpa.free(xs_bytes);
    try act_b.download(0, act_bytes);
    try inj_out.download(0, inj_bytes);
    try xs_b.download(0, xs_bytes);
    var act_mismatch: usize = 0;
    for (act_want, 0..) |w, i| {
        if (std.mem.readInt(u16, act_bytes[2 * i ..][0..2], .little) != w) act_mismatch += 1;
    }
    var inj_mismatch: usize = 0;
    for (inj_want, 0..) |w, i| {
        if (std.mem.readInt(u16, inj_bytes[2 * i ..][0..2], .little) != w) inj_mismatch += 1;
    }
    var xs_ulp: u32 = 0;
    for (xs_want, 0..) |w, i| {
        const got_xs: f32 = @bitCast(std.mem.readInt(u32, xs_bytes[4 * i ..][0..4], .little));
        const dist = flash.hc.ulps(got_xs, w);
        if (dist > xs_ulp) xs_ulp = dist;
    }
    std.debug.print("reduce act mismatch {d} inject mismatch {d} xs max_ulp {d}\n", .{ act_mismatch, inj_mismatch, xs_ulp });
    if (act_mismatch != 0 or inj_mismatch != 0 or xs_ulp > 4) return 1;

    const mixed_bytes = try gpa.alloc(u8, dims * 2);
    defer gpa.free(mixed_bytes);
    try mixed_b.download(0, mixed_bytes);
    const act_u16 = try gpa.alloc(u16, low);
    defer gpa.free(act_u16);
    for (act_u16, 0..) |*o, i| o.* = std.mem.readInt(u16, act_bytes[2 * i ..][0..2], .little);
    const xs_f = try gpa.alloc(f32, low / 32);
    defer gpa.free(xs_f);
    for (xs_f, 0..) |*o, i| o.* = @bitCast(std.mem.readInt(u32, xs_bytes[4 * i ..][0..4], .little));
    const norm_u16 = try gpa.alloc(u16, streams * dims);
    defer gpa.free(norm_u16);
    for (norm_u16, 0..) |*o, i| o.* = std.mem.readInt(u16, got[2 * i ..][0..2], .little);
    const host_mixed = try gpa.alloc(u16, dims);
    defer gpa.free(host_mixed);
    try flash.hc.mix(act_u16, xs_f, norm_u16, up_w.bytes, up_s.bytes, up_b.bytes, up_n, up_k8, dims, streams, host_mixed);
    var mixed_off: usize = 0;
    var max_steps: u32 = 0;
    for (host_mixed, 0..) |host_m, i| {
        const got_m = std.mem.readInt(u16, mixed_bytes[2 * i ..][0..2], .little);
        const dist = flash.hc.mixedSteps(got_m, host_m);
        if (dist != 0) mixed_off += 1;
        if (dist > max_steps) max_steps = dist;
    }
    std.debug.print("upmix dims {d} off {d} max_steps {d} head {x:0>4} host {x:0>4}\n", .{ dims, mixed_off, max_steps, std.mem.readInt(u16, mixed_bytes[0..2], .little), host_mixed[0] });
    if (max_steps > 2) return 1;
    const rest = try runProj(gpa, driver, stream, mapped, mixed_b, xsm_b, tri, h, inj_out.ptr, wide_u16);
    if (rest != 0) return 1;
    return mismatch;
}

const proj_names = [_][]const u8{ "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a" };
const proj_rows = [_]usize{ 10240, 6144, 48, 48 };

fn runProj(gpa: std.mem.Allocator, driver: *cuda.Driver, stream: *cuda.Stream, mapped: *const embed.Mapped, mixed: cuda.DeviceBuffer, xs: cuda.DeviceBuffer, tri: flash.Tri, h: u64, inj: u64, hidden: []const u16) !u32 {
    const prefix = "language_model.model.layers.0.linear_attn.";
    const k8: usize = 320;
    const kg: usize = 80;
    var name: [160]u8 = undefined;
    var n: usize = 0;
    var words_len: usize = 0;
    var scale_len: usize = 0;
    var ws: [4]core.safetensors.Tensor = undefined;
    var ss: [4]core.safetensors.Tensor = undefined;
    var bs: [4]core.safetensors.Tensor = undefined;
    for (proj_names, proj_rows, 0..) |part, rows, i| {
        ws[i] = try mapped.lookup(gpa, try std.fmt.bufPrint(&name, "{s}{s}.weight", .{ prefix, part }), .u32);
        ss[i] = try mapped.lookup(gpa, try std.fmt.bufPrint(&name, "{s}{s}.scales", .{ prefix, part }), .bf16);
        bs[i] = try mapped.lookup(gpa, try std.fmt.bufPrint(&name, "{s}{s}.biases", .{ prefix, part }), .bf16);
        if (ws[i].dim(0) != rows or ws[i].dim(1) != k8 or ss[i].dim(0) != rows or ss[i].dim(1) != kg) return error.UnexpectedTensor;
        n += rows;
        words_len += ws[i].bytes.len;
        scale_len += ss[i].bytes.len;
    }
    if (n != 16480) return error.UnexpectedTensor;
    const k = k8 * 8;
    const row = try gpa.alloc(u16, k);
    defer gpa.free(row);
    const words = try gpa.alloc(u8, words_len);
    defer gpa.free(words);
    const scales = try gpa.alloc(u8, scale_len);
    defer gpa.free(scales);
    const biases = try gpa.alloc(u8, scale_len);
    defer gpa.free(biases);
    var wo: usize = 0;
    var so: usize = 0;
    for (ws, ss, bs) |w, s, b| {
        @memcpy(words[wo..][0..w.bytes.len], w.bytes);
        wo += w.bytes.len;
        @memcpy(scales[so..][0..s.bytes.len], s.bytes);
        @memcpy(biases[so..][0..b.bytes.len], b.bytes);
        so += s.bytes.len;
    }
    var lane = try flash.qmm.pack(gpa, words, scales, biases, n, k8);
    defer lane.deinit(gpa);
    var w_b = try cuda.DeviceBuffer.fromHost(driver, lane.weight);
    defer w_b.free();
    var s_b = try cuda.DeviceBuffer.fromHost(driver, lane.scales);
    defer s_b.free();
    var b_b = try cuda.DeviceBuffer.fromHost(driver, lane.biases);
    defer b_b.free();
    var out_b = try cuda.DeviceBuffer.alloc(driver, n * 2);
    defer out_b.free();
    try flash.qmm.matmul(driver, stream.*, mixed.ptr, xs.ptr, w_b.ptr, s_b.ptr, b_b.ptr, out_b.ptr, 1, n, k);
    const row_bytes = try gpa.alloc(u8, k * 2);
    defer gpa.free(row_bytes);
    const xs_bytes = try gpa.alloc(u8, kg * 4);
    defer gpa.free(xs_bytes);
    const got_bytes = try gpa.alloc(u8, n * 2);
    defer gpa.free(got_bytes);
    try stream.synchronize();
    try mixed.download(0, row_bytes);
    try xs.download(0, xs_bytes);
    try out_b.download(0, got_bytes);
    for (row, 0..) |*o, i| o.* = std.mem.readInt(u16, row_bytes[2 * i ..][0..2], .little);
    const xs_f = try gpa.alloc(f32, kg);
    defer gpa.free(xs_f);
    for (xs_f, 0..) |*o, i| o.* = @bitCast(std.mem.readInt(u32, xs_bytes[4 * i ..][0..4], .little));
    const host = try gpa.alloc(u16, n);
    defer gpa.free(host);
    try flash.qmm.dotRow(row, xs_f, words, scales, biases, n, k, host);
    var off: usize = 0;
    var max_steps: u32 = 0;
    for (host, 0..) |want, i| {
        const got = std.mem.readInt(u16, got_bytes[2 * i ..][0..2], .little);
        const dist = flash.hc.mixedSteps(got, want);
        if (dist != 0) off += 1;
        if (dist > max_steps) max_steps = dist;
    }
    std.debug.print("proj n {d} off {d} max_steps {d} head {x:0>4} host {x:0>4}\n", .{ n, off, max_steps, std.mem.readInt(u16, got_bytes[0..2], .little), host[0] });
    if (max_steps != 0) return max_steps;
    const slices = try flash.gdn.memory(gpa, driver, stream, mapped, out_b, got_bytes) orelse return 1;
    defer gpa.free(slices);
    const wrote = try flash.mlp.connect(flash.Tri, gpa, driver, stream, mapped, tri, h, inj, hidden, slices);
    if (wrote != 0) return wrote;
    return flash.moe.experts(flash.Tri, gpa, driver, stream, mapped, tri, mixed.ptr);
}
