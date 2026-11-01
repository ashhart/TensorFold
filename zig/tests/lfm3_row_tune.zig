//! Diagnostic single-row launch sweep through unchanged shared projection arithmetic and production byte comparison.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const q = @import("tensorfold").lfm3;
const Row = @import("tensorfold").row_projection;

fn launch(e: mtl.ComputeEncoder, pipe: mtl.Pipeline, w: Row.Weights, x: mtl.Buffer, y: mtl.Buffer, rps: usize, groups: usize) void {
    e.setPipeline(pipe);
    e.setBuffer(x, 0, 0);
    e.setBuffer(w.w, w.w_off, 1);
    e.setBuffer(w.scales, w.s_off, 2);
    e.setBuffer(w.biases, w.b_off, 3);
    const dims = [_]i32{ @intCast(w.k), @intCast(w.n), @intCast(w.group), 1, 1, 1 };
    e.setBytes(std.mem.asBytes(&dims), 4);
    e.setBuffer(y, 0, 5);
    e.dispatchGroups(.of((w.n + rps * groups - 1) / (rps * groups), 1, 1), .of(groups * 32, 1, 1));
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const one = args.len == 5 and std.mem.eql(u8, args[4], "--rps1");
    const row_sweep = args.len == 5 and std.mem.startsWith(u8, args[4], "--rows=");
    if (args.len != 4 and !one and !row_sweep) return error.ExpectedModelModuleInput;
    const rows: usize = if (row_sweep) try std.fmt.parseInt(usize, args[4][7..], 10) else 1;
    if (rows != 1 and rows != 4 and rows != 8 and rows != 16 and rows != 32) return error.UnsupportedSweepRows;
    const columns: []const usize = if (one) &.{ 1, 4 } else &.{ 2, 4, 8 };
    const groups: []const usize = if (one) &.{ 1, 2, 4, 8, 16 } else &.{ 1, 2, 4, 8 };
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const model = try q.Model.load(init.gpa, init.io, args[1]);
    defer model.deinit();
    var name: [512]u8 = undefined;
    const weights = try model.checkpoint.get(try std.fmt.bufPrint(&name, "{s}.weight", .{args[2]}));
    const scales = try model.checkpoint.get(try std.fmt.bufPrint(&name, "{s}.scales", .{args[2]}));
    const biases = try model.checkpoint.get(try std.fmt.bufPrint(&name, "{s}.biases", .{args[2]}));
    const quant = try model.config.width(args[2]);
    if (weights.dtype != .u32 or scales.dtype != .bf16 or biases.dtype != .bf16 or scales.rank != 2)
        return error.ExpectedAffineProjection;
    const w = Row.Weights{ .w = weights.buffer, .w_off = weights.offset, .scales = scales.buffer, .s_off = scales.offset, .biases = biases.buffer, .b_off = biases.offset, .n = scales.shape[0], .k = scales.shape[1] * quant.group, .group = quant.group, .bits = quant.bits };
    try w.validate();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[3], init.gpa, .limited(4 << 20));
    defer init.gpa.free(bytes);
    if (bytes.len < rows * w.k * 2 or (!row_sweep and bytes.len != w.k * 2)) return error.ExpectedBf16InputRows;
    const input = try model.device.buffer(rows * w.k * 2, mtl.ResourceOptions.shared);
    defer input.deinit();
    @memcpy(input.contents()[0 .. rows * w.k * 2], bytes[0 .. rows * w.k * 2]);
    const output = try model.device.buffer(rows * w.n * 2, mtl.ResourceOptions.shared);
    defer output.deinit();
    var scratch = try q.state.Scratch.init(init.gpa, model.device, &model.config.layer_types, rows);
    defer scratch.deinit();
    try q.forward.project(model, &scratch, w, input, output, rows);
    const baseline = try init.gpa.dupe(u8, output.contents()[0 .. rows * w.n * 2]);
    defer init.gpa.free(baseline);
    if (row_sweep) return sweepRows(init.gpa, model, w, input, output, baseline, rows);
    var text = std.Io.Writer.Allocating.init(init.gpa);
    defer text.deinit();
    try text.writer.writeAll(sources.core_row_projection);
    for (columns) |rps| for (groups) |sg| {
        if (one and rps == 4 and sg != 2) continue;
        try text.writer.print("\ntemplate [[host_name(\"l3_tune_r{d}_s{d}\")]] [[kernel]] decltype(tf_row_projection<{d},{d},1,false,{d},false>) tf_row_projection<{d},{d},1,false,{d},false>;\n", .{ rps, sg, rps, sg, w.bits, rps, sg, w.bits });
    };
    const library = try mtl.Library.fromSource(model.device, text.written(), mtl.CompileOptions.mlx());
    defer library.deinit();
    for (columns) |rps| for (groups) |sg| {
        if (one and rps == 4 and sg != 2) continue;
        const symbol = try std.fmt.bufPrint(&name, "l3_tune_r{d}_s{d}", .{ rps, sg });
        const pipe = try mtl.Pipeline.init(model.device, library, try init.arena.allocator().dupeSentinel(u8, symbol, 0), false);
        defer pipe.deinit();
        var warm_seconds: f64 = 0;
        var warm_count: usize = 0;
        while (warm_seconds < 0.02 and warm_count < 128) : (warm_count += 1) {
            const cb = model.queue.commandBuffer();
            const e = cb.compute(.serial);
            for (0..32) |_| launch(e, pipe, w, input, output, rps, sg);
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure() != null) return error.GpuFailed;
            warm_seconds += cb.gpuSeconds();
        }
        var seconds: [5]f64 = undefined;
        for (&seconds) |*elapsed| {
            const cb = model.queue.commandBuffer();
            const e = cb.compute(.serial);
            for (0..32) |_| launch(e, pipe, w, input, output, rps, sg);
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure() != null) return error.GpuFailed;
            elapsed.* = cb.gpuSeconds() / 32;
        }
        const same = std.mem.eql(u8, baseline, output.contents()[0 .. rows * w.n * 2]);
        std.debug.print("ROW_TUNE rps={d} sg={d} bits={d} group={d} k={d} n={d} equal={} seconds={any}\n", .{ rps, sg, w.bits, w.group, w.k, w.n, same, seconds });
        if (!same) return error.ProjectionLaunchChangedBytes;
    };
}

const rows_source =
    \\template<int MB, int BITS>
    \\[[kernel, max_total_threads_per_threadgroup(64)]] void l3_tune_rows(
    \\ const device bfloat* X [[buffer(0)]], const device uint32_t* W [[buffer(1)]],
    \\ const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]],
    \\ constant int* dims [[buffer(4)]], device bfloat* OUT [[buffer(5)]],
    \\ uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    \\ uint2 tg [[threadgroup_position_in_grid]]) {
    \\ const int K=dims[0], N=dims[1], GS=dims[2], rows=dims[3];
    \\ const int first=int(tg.x)*MB, column=(int(tg.y)*2+int(sg))*4;
    \\ if(first>=rows || column>=N) return;
    \\ const device uint8_t* wr=(const device uint8_t*)W+size_t(column)*(K*BITS/8);
    \\ const device bfloat* sr=S+size_t(column)*(K/GS), *br=B+size_t(column)*(K/GS);
    \\ if(first+MB<=rows) {
    \\   float acc[MB][4];
    \\   tf_rp_rows<4,MB,BITS,false>(wr,sr,br,X+size_t(first)*K,K,GS,lane,acc);
    \\   if(lane==0) for(int r=0;r<MB;++r) for(int j=0;j<4;++j)
    \\     OUT[size_t(first+r)*N+column+j]=tf_rp_out<false>(acc[r][j]);
    \\ } else for(int r=first;r<rows;++r) {
    \\   float acc[1][4];
    \\   tf_rp_rows<4,1,BITS,false>(wr,sr,br,X+size_t(r)*K,K,GS,lane,acc);
    \\   if(lane==0) for(int j=0;j<4;++j) OUT[size_t(r)*N+column+j]=tf_rp_out<false>(acc[0][j]);
    \\ }
    \\}
;
fn launchRows(e: mtl.ComputeEncoder, pipe: mtl.Pipeline, w: Row.Weights, x: mtl.Buffer, y: mtl.Buffer, rows: usize, mb: usize) void {
    e.setPipeline(pipe);
    e.setBuffer(x, 0, 0);
    e.setBuffer(w.w, w.w_off, 1);
    e.setBuffer(w.scales, w.s_off, 2);
    e.setBuffer(w.biases, w.b_off, 3);
    const dims = [_]i32{ @intCast(w.k), @intCast(w.n), @intCast(w.group), @intCast(rows), 1, 1 };
    e.setBytes(std.mem.asBytes(&dims), 4);
    e.setBuffer(y, 0, 5);
    e.dispatchGroups(.of((rows + mb - 1) / mb, w.n / 8, 1), .of(64, 1, 1));
}
fn sweepRows(a: std.mem.Allocator, model: *q.Model, w: Row.Weights, x: mtl.Buffer, y: mtl.Buffer, baseline: []const u8, rows: usize) !void {
    var text = std.Io.Writer.Allocating.init(a);
    defer text.deinit();
    try text.writer.writeAll(sources.core_row_projection);
    try text.writer.writeAll(rows_source);
    for ([_]usize{ 2, 4, 8 }) |mb| try text.writer.print("\ntemplate [[host_name(\"l3_tune_mb{d}\")]] [[kernel]] decltype(l3_tune_rows<{d},{d}>) l3_tune_rows<{d},{d}>;\n", .{ mb, mb, w.bits, mb, w.bits });
    const lib = try mtl.Library.fromSource(model.device, text.written(), mtl.CompileOptions.mlx());
    defer lib.deinit();
    for ([_]usize{ 2, 4, 8 }) |mb| {
        const name = try std.fmt.allocPrintSentinel(a, "l3_tune_mb{d}", .{mb}, 0);
        defer a.free(name);
        const pipe = try mtl.Pipeline.init(model.device, lib, name, false);
        defer pipe.deinit();
        var warm: f64 = 0;
        var warm_count: usize = 0;
        while (warm < 0.02 and warm_count < 128) : (warm_count += 1) {
            const cb = model.queue.commandBuffer();
            const e = cb.compute(.serial);
            for (0..32) |_| launchRows(e, pipe, w, x, y, rows, mb);
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure() != null) return error.GpuFailed;
            warm += cb.gpuSeconds();
        }
        var timings: [5]f64 = undefined;
        for (&timings) |*elapsed| {
            const cb = model.queue.commandBuffer();
            const e = cb.compute(.serial);
            for (0..32) |_| launchRows(e, pipe, w, x, y, rows, mb);
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure() != null) return error.GpuFailed;
            elapsed.* = cb.gpuSeconds() / 32;
        }
        const same = std.mem.eql(u8, baseline, y.contents()[0..baseline.len]);
        std.debug.print("ROWS_TUNE rows={d} mb={d} bits={d} group={d} k={d} n={d} equal={} seconds={any}\n", .{ rows, mb, w.bits, w.group, w.k, w.n, same, timings });
        if (!same) return error.ProjectionBatchChangedBytes;
    }
}
