//! The prompt-chunk kernels: each embedded file compiled as one library with MLX's options, kernels found by name.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const frags = @import("../../core/frags.zig");

pub const Kernels = struct {
    gpa: std.mem.Allocator,
    pipelines: std.StringHashMapUnmanaged(mtl.Pipeline) = .empty,

    /// The pipeline of an entry point (its full Metal function name).
    pub fn get(self: *const Kernels, name: []const u8) mtl.Pipeline {
        return self.pipelines.get(name) orelse std.debug.panic("no prefill kernel {s}", .{name});
    }

    pub fn deinit(self: *Kernels) void {
        var it = self.pipelines.iterator();
        while (it.next()) |e| {
            e.value_ptr.deinit();
            self.gpa.free(e.key_ptr.*);
        }
        self.pipelines.deinit(self.gpa);
    }
};

/// Every `[[kernel]] void NAME(` in a source.
fn names(gpa: std.mem.Allocator, text: []const u8) ![][]const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    errdefer out.deinit(gpa);
    const mark = "[[kernel]] void ";
    var at: usize = 0;
    while (std.mem.indexOfPos(u8, text, at, mark)) |i| {
        const start = i + mark.len;
        var end = start;
        while (end < text.len and (std.ascii.isAlphanumeric(text[end]) or text[end] == '_')) end += 1;
        try out.append(gpa, text[start..end]);
        at = end;
    }
    return out.toOwnedSlice(gpa);
}

const Job = struct {
    device: mtl.Device,
    file: sources.File,
    gpa: std.mem.Allocator,
    found: []?mtl.Pipeline = &.{},
    kernels: [][]const u8 = &.{},
    failed: bool = false,

    fn run(job: *Job) void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const text = frags.source(job.device, job.gpa, job.file.text) catch {
            job.failed = true;
            return;
        };
        defer job.gpa.free(text);
        const lib = mtl.Library.fromSource(job.device, text, mtl.CompileOptions.mlx()) catch {
            std.log.err("prefill kernels {s}: compile failed", .{job.file.name});
            job.failed = true;
            return;
        };
        defer lib.deinit();
        job.kernels = names(job.gpa, job.file.text) catch {
            job.failed = true;
            return;
        };
        job.found = job.gpa.alloc(?mtl.Pipeline, job.kernels.len) catch {
            job.failed = true;
            return;
        };
        // a templated entry has no function of its own name: only instantiated kernels make pipelines
        for (job.kernels, job.found) |n, *p| p.* = mtl.Pipeline.init(job.device, lib, n, false) catch null;
    }
};

/// Compile every prefill and MLX-op file on worker threads.
pub fn load(gpa: std.mem.Allocator, device: mtl.Device) !Kernels {
    var jobs: [sources.prefill.len]Job = undefined;
    for (&jobs, sources.prefill) |*j, f| j.* = .{ .device = device, .file = f, .gpa = gpa };
    var threads: [jobs.len]?std.Thread = @splat(null);
    for (&threads, &jobs) |*t, *j| t.* = std.Thread.spawn(.{}, Job.run, .{j}) catch null;
    for (threads, &jobs) |t, *j| if (t) |th| th.join() else j.run();
    // Every joined job retains its pipelines until ownership moves into the result.
    defer {
        for (&jobs) |*j| {
            for (j.found) |*found| if (found.*) |*p| p.deinit();
            gpa.free(j.found);
            gpa.free(j.kernels);
        }
    }
    var k = Kernels{ .gpa = gpa };
    errdefer k.deinit();
    for (&jobs) |*j| {
        if (j.failed) return error.KernelCompile;
        for (j.kernels, j.found) |n, *found| {
            if (found.*) |p| {
                // Keep the existing last-definition-wins behavior without leaking the old pipeline.
                if (k.pipelines.getPtr(n)) |previous| {
                    previous.deinit();
                    previous.* = p;
                    found.* = null;
                    continue;
                }
                const key = try gpa.dupe(u8, n);
                k.pipelines.put(gpa, key, p) catch |err| {
                    gpa.free(key);
                    return err;
                };
                found.* = null;
            }
        }
    }
    return k;
}

test "every prompt-chunk source compiles at this macOS's Metal language" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    var k = try load(std.testing.allocator, device);
    k.deinit();
}

test "entry-point names retain source slices and order" {
    const allocator = std.testing.allocator;
    const text = "[[kernel]] void alpha_1( [[kernel]] void beta2(";
    const found = try names(allocator, text);
    defer allocator.free(found);
    try std.testing.expectEqual(@as(usize, 2), found.len);
    try std.testing.expectEqualStrings("alpha_1", found[0]);
    try std.testing.expectEqualStrings("beta2", found[1]);
}

test "source without entry points returns an empty list" {
    const allocator = std.testing.allocator;
    const found = try names(allocator, "// header only");
    defer allocator.free(found);
    try std.testing.expectEqual(@as(usize, 0), found.len);
}
