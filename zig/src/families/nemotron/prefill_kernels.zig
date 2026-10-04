//! The prompt-chunk kernels: each embedded file compiled as one library with MLX's options, kernels found by name.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

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

/// `text` with its local `#include "../nax.h"` replaced by the header, as the prefill tools compile it.
fn inlined(gpa: std.mem.Allocator, text: []const u8) ![]u8 {
    const line = "#include \"../nax.h\"";
    const at = std.mem.indexOf(u8, text, line) orelse return gpa.dupe(u8, text);
    return std.mem.concat(gpa, u8, &.{ text[0..at], sources.nax, text[at + line.len ..] });
}

/// Every `[[kernel]] void NAME(` in a source.
fn names(gpa: std.mem.Allocator, text: []const u8) ![][]const u8 {
    var out: std.ArrayList([]const u8) = .empty;
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
        const text = inlined(job.gpa, job.file.text) catch {
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
    var k = Kernels{ .gpa = gpa };
    errdefer k.deinit();
    for (&jobs) |*j| {
        defer gpa.free(j.kernels);
        defer gpa.free(j.found);
        if (j.failed) return error.KernelCompile;
        for (j.kernels, j.found) |n, found| if (found) |p| try k.pipelines.put(gpa, try gpa.dupe(u8, n), p);
    }
    return k;
}
