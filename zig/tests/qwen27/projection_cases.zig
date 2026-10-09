//! Replay actual input rows through the production prepared projection stacks.
const std = @import("std");
const q = @import("qwen27");
const mtl = @import("metal");

const Kind = enum { qkv, zba, delta_out, attention_q, attention_kv, attention_out, gu, down, head };
const Job = struct { kind: Kind, layer: usize, input: []const u8, output: []const u8, rows: u32 };

fn linear(m: *q.model.Model, job: Job) !q.projection.Linear {
    if (job.layer >= m.config.layers) return error.InvalidLayer;
    if (job.kind == .head) return m.weights.head;
    const layer = m.weights.layers[job.layer];
    return switch (job.kind) {
        .qkv, .zba, .delta_out => switch (layer.mixer) {
            .linear => |g| switch (job.kind) {
                .qkv => g.qkv,
                .zba => g.zba,
                else => g.out,
            },
            else => error.ExpectedLinearLayer,
        },
        .attention_q, .attention_kv, .attention_out => switch (layer.mixer) {
            .attention => |a| switch (job.kind) {
                .attention_q => a.q,
                .attention_kv => a.kv,
                else => a.out,
            },
            else => error.ExpectedAttentionLayer,
        },
        .gu => layer.gu,
        .down => layer.down,
        .head => unreachable,
    };
}

fn one(m: *q.model.Model, io: std.Io, job: Job) !void {
    const p = try linear(m, job);
    const data = try std.Io.Dir.cwd().readFileAlloc(io, job.input, m.allocator, .limited(2 << 20));
    defer m.allocator.free(data);
    if (data.len != @as(usize, job.rows) * p.k * 2) return error.InvalidInputRows;
    const input = try m.device.buffer(data.len, mtl.ResourceOptions.shared);
    defer input.deinit();
    @memcpy(input.contents()[0..data.len], data);
    const bytes = @as(usize, job.rows) * p.n * 2;
    const output = try m.device.buffer(bytes, mtl.ResourceOptions.shared);
    defer output.deinit();
    const cb = m.queue.commandBuffer();
    const e = cb.compute(.serial);
    try m.kernels.quant(e, p, .{ .buffer = input }, null, m.frame.get(.dims), .{ .buffer = output }, job.rows); // raw rows carry no producer sums: the kernel takes its own
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailed;
    const file = try std.Io.Dir.cwd().createFile(io, job.output, .{ .exclusive = true });
    defer file.close(io);
    try file.writeStreamingAll(io, output.contents()[0..bytes]);
    std.debug.print("projection {s} layer{d} rows{d} N{d} K{d} SK{d}\n", .{ @tagName(job.kind), job.layer, job.rows, p.n, p.k, p.slices });
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 3) return error.ExpectedModelCaseJson;
    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], a, .limited(1 << 20));
    const jobs = try std.json.parseFromSlice([]Job, a, text, .{ .allocate = .alloc_always });
    if (jobs.value.len == 0 or jobs.value.len > 128) return error.InvalidCaseCount;
    for (jobs.value, 0..) |job, i| {
        if (job.rows == 0 or job.rows > 16 or !std.fs.path.isAbsolute(job.input) or !std.fs.path.isAbsolute(job.output)) return error.InvalidJob;
        if (std.mem.eql(u8, job.input, job.output)) return error.AliasedJobPaths;
        for (jobs.value[0..i]) |before| if (std.mem.eql(u8, before.output, job.output)) return error.DuplicateOutput;
        std.Io.Dir.cwd().access(init.io, job.output, .{}) catch |err| switch (err) {
            error.FileNotFound => continue,
            else => return err,
        };
        return error.OutputAlreadyExists;
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const m = try q.model.Model.load(init.gpa, init.io, args[1], 16);
    defer m.deinit();
    for (jobs.value) |job| try one(m, init.io, job);
}
