//! The CUDA joint schema contract set: all thirteen kernels, both logits outputs, and their planted controls.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const g = @import("core").contract.gates;
const Fns = @import("qwen35").joint_schema.Fns;
const rows = @import("contract_js_rows.zig");
const spans = @import("contract_js_spans.zig");
const point = @import("contract_js_point.zig");
const heads = @import("contract_js_heads.zig");

const controls = struct {
    pub const bytes align(16) = @embedFile("fatbin_contract_controls").*;
};

pub const Set = struct {
    gpa: std.mem.Allocator,
    fns: Fns,
    ctl: cuda.Module,
    row_cases: [std.enums.values(rows.Which).len]rows.Row = undefined,
    span_cases: [2 * std.enums.values(spans.Which).len]spans.Spans = undefined,
    point_cases: [2 * std.enums.values(point.Which).len]point.Point = undefined,
    head_cases: [2 * std.enums.values(heads.Which).len]heads.Heads = undefined,
    used: [4]usize = @splat(0),

    /// Allocate stable cases and append each kernel and control to the normal CUDA runner's list.
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var fns = try Fns.load(gpu.d);
        errdefer fns.unload();
        var ctl = try cuda.Module.load(gpu.d, &controls.bytes);
        errdefer ctl.unload();
        set.* = .{ .gpa = gpu.gpa, .fns = fns, .ctl = ctl };
        errdefer set.freeCases();
        for (std.enums.values(rows.Which)) |which| {
            const x = &set.row_cases[set.used[0]];
            x.* = try .init(gpu, s, fns, ctl, which);
            set.used[0] += 1;
            try list.append(a, x.kernel());
        }
        for (std.enums.values(spans.Which)) |which| for ([_]bool{ false, true }) |control| {
            const x = &set.span_cases[set.used[1]];
            x.* = try .init(gpu, s, fns, ctl, which, control);
            set.used[1] += 1;
            try list.append(a, x.kernel());
        };
        for (std.enums.values(point.Which)) |which| for ([_]bool{ false, true }) |control| {
            const x = &set.point_cases[set.used[2]];
            x.* = try .init(gpu, s, fns, ctl, which, control);
            set.used[2] += 1;
            try list.append(a, x.kernel());
        };
        for (std.enums.values(heads.Which)) |which| for ([_]bool{ false, true }) |control| {
            const x = &set.head_cases[set.used[3]];
            x.* = try .init(gpu, s, fns, ctl, which, control);
            set.used[3] += 1;
            try list.append(a, x.kernel());
        };
        return set;
    }

    fn freeCases(set: *Set) void {
        for (set.row_cases[0..set.used[0]]) |*x| x.deinit();
        for (set.span_cases[0..set.used[1]]) |*x| x.deinit();
        for (set.point_cases[0..set.used[2]]) |*x| x.deinit();
        for (set.head_cases[0..set.used[3]]) |*x| x.deinit();
    }

    pub fn deinit(set: *Set) void {
        set.freeCases();
        set.ctl.unload();
        set.fns.unload();
        set.gpa.destroy(set);
    }
};
