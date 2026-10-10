//! Native GDN fixture replay checks every byte with bounded buffers and no model or framework runtime.
const std = @import("std");
const mtl = @import("metal");

const Binding = struct { slot: u32, file: []const u8, bytes: usize };
const Case = struct { name: []const u8, function: []const u8, grid: [3]usize, group: [3]usize, inputs: []const Binding, outputs: []const Binding };
const Manifest = struct { version: u32, cases: []const Case };

fn path(a: std.mem.Allocator, dir: []const u8, name: []const u8) ![:0]const u8 {
    if (name.len == 0 or std.fs.path.isAbsolute(name) or std.mem.indexOf(u8, name, "..") != null) return error.BadPath;
    return std.fs.path.joinZ(a, &.{ dir, name });
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 3) return error.Usage;
    const dir = args[1];
    const src = try std.Io.Dir.cwd().readFileAlloc(io, args[2], a, .limited(1 << 20));
    const json = try std.Io.Dir.cwd().readFileAlloc(io, try path(a, dir, "manifest.json"), a, .limited(4 << 20));
    const manifest = try std.json.parseFromSliceLeaky(Manifest, a, json, .{ .ignore_unknown_fields = true });
    if (manifest.version != 1 or manifest.cases.len == 0 or manifest.cases.len > 256) return error.BadManifest;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
    defer lib.deinit();
    var checked: usize = 0;
    for (manifest.cases) |case| {
        const case_pool = mtl.objc.Pool.push();
        defer case_pool.pop();
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        const per_case = arena.allocator();
        if (case.inputs.len > 16 or case.outputs.len == 0 or case.outputs.len > 8) return error.BadManifest;
        const pipeline = try mtl.Pipeline.init(device, lib, case.function, false);
        defer pipeline.deinit();
        if (case.group[0] * case.group[1] * case.group[2] > pipeline.maxThreads()) return error.BadGroup;
        var inputs: [16]mtl.Buffer = undefined;
        var outputs: [8]mtl.Buffer = undefined;
        var expected: [8][]const u8 = undefined;
        var ni: usize = 0;
        var no: usize = 0;
        defer {
            for (inputs[0..ni]) |buffer| buffer.deinit();
            for (outputs[0..no]) |buffer| buffer.deinit();
        }
        for (case.inputs, 0..) |binding, i| {
            if (binding.slot > 15 or binding.bytes == 0 or binding.bytes > 64 << 20) return error.BadBinding;
            const bytes = try std.Io.Dir.cwd().readFileAlloc(io, try path(per_case, dir, binding.file), per_case, .limited(binding.bytes));
            if (bytes.len != binding.bytes) return error.BadBinding;
            inputs[i] = try device.buffer(bytes.len, mtl.ResourceOptions.shared);
            ni += 1;
            @memcpy(inputs[i].contents()[0..bytes.len], bytes);
        }
        for (case.outputs, 0..) |binding, i| {
            if (binding.slot > 15 or binding.bytes == 0 or binding.bytes > 64 << 20) return error.BadBinding;
            expected[i] = try std.Io.Dir.cwd().readFileAlloc(io, try path(per_case, dir, binding.file), per_case, .limited(binding.bytes));
            if (expected[i].len != binding.bytes) return error.BadBinding;
            outputs[i] = try device.buffer(binding.bytes, mtl.ResourceOptions.shared);
            no += 1;
            @memset(outputs[i].contents()[0..binding.bytes], 0xa5);
        }
        const cb = queue.commandBuffer();
        const encoder = cb.compute(.serial);
        encoder.setPipeline(pipeline);
        for (case.inputs, 0..) |binding, i| encoder.setBuffer(inputs[i], 0, binding.slot);
        for (case.outputs, 0..) |binding, i| encoder.setBuffer(outputs[i], 0, binding.slot);
        encoder.dispatchThreads(mtl.Size.of(case.grid[0], case.grid[1], case.grid[2]), mtl.Size.of(case.group[0], case.group[1], case.group[2]));
        encoder.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.GpuFailed;
        for (case.outputs, 0..) |binding, i| {
            const got = outputs[i].contents()[0..binding.bytes];
            const saved = try std.fmt.allocPrint(a, "{s}.actual", .{binding.file});
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try path(a, dir, saved), .data = got });
            if (!std.mem.eql(u8, got, expected[i])) {
                var bad: usize = 0;
                for (got, expected[i]) |g, w| bad += @intFromBool(g != w);
                var first: usize = 0;
                while (first < got.len and got[first] == expected[i][first]) first += 1;
                std.debug.print("FAIL {s} output {d}: {d} differing bytes; first byte {d} want {x} got {x}\n", .{ case.name, i, bad, first, expected[i][first], got[first] });
                return error.OperatorBits;
            }
            checked += binding.bytes;
        }
        std.debug.print("PASS {s}\n", .{case.name});
    }
    std.debug.print("{{\"native_runner_run\":true,\"operator_cases\":{d},\"checked_bytes\":{d},\"different_bytes\":0}}\n", .{ manifest.cases.len, checked });
}
