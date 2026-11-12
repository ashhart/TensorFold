//! The joint schema head from a checkpoint directory: the config checked, every tensor's dtype and shape by name.
const std = @import("std");
const safetensors = @import("../safetensors.zig");
const config = @import("config.zig");
const tensors = @import("tensors.zig");
const Config = config.Config;
const Problem = config.Problem;

pub const config_file = "joint_head_config.json";
pub const weights_file = "joint_head.safetensors";

/// The mapped head: its tensors borrow the file's bytes until `close`.
pub const Head = struct {
    arena: std.heap.ArenaAllocator,
    file: safetensors.File,
    weights: tensors.Weights,

    pub fn close(h: *Head, io: std.Io) void {
        h.file.close(io);
        h.arena.deinit();
        h.* = undefined;
    }
};

/// `dir`'s head config and weights; on a refusal `problem` names the field, file or tensor.
pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, problem: *Problem) !Head {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    const text = std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, config_file }), a, .limited(1 << 16)) catch |e| {
        problem.set("{s}: {s}", .{ config_file, @errorName(e) });
        return e;
    };
    const cfg = try config.parse(gpa, text, problem);
    var file = safetensors.File.open(gpa, io, try std.fs.path.join(a, &.{ dir, weights_file })) catch |e| {
        problem.set("{s}: {s}", .{ weights_file, @errorName(e) });
        return e;
    };
    errdefer file.close(io);
    const weights = try check(a, &file, cfg, problem);
    return .{ .arena = arena, .file = file, .weights = weights };
}

/// The weights of `file` for `cfg`: a missing, extra, non-BF16 or misshaped tensor refused by name.
pub fn check(a: std.mem.Allocator, file: *const safetensors.File, cfg: Config, problem: *Problem) !tensors.Weights {
    var src: Source = .{ .file = file, .problem = problem };
    const w = try tensors.fill(a, cfg, &src);
    if (file.names.count() == tensors.count(cfg)) return w;
    // every expected name was found, so the surplus are names the head does not have
    var listing: tensors.Listing = .{ .a = a };
    _ = try tensors.fill(a, cfg, &listing);
    for (file.names.keys()) |name| {
        for (listing.names.items) |n| {
            if (std.mem.eql(u8, n, name)) break;
        } else {
            problem.set("{s}: unexpected tensor {s}", .{ weights_file, name });
            return error.ExtraTensor;
        }
    }
    unreachable;
}

const Source = struct {
    file: *const safetensors.File,
    problem: *Problem,

    pub fn take(s: *Source, name: []const u8, shape: []const usize) !tensors.Tensor {
        const t = s.file.get(name) orelse {
            s.problem.set("{s}: missing tensor {s}", .{ weights_file, name });
            return error.MissingTensor;
        };
        if (!t.is(.bf16, shape)) {
            s.problem.set("{s}: {s} is {t} {any}, want BF16 {any}", .{ weights_file, name, t.dtype, t.shape[0..t.rank], shape });
            return error.BadTensor;
        }
        return t;
    }
};
