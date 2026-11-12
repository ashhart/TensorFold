//! Flash Next EXL3's TP prefill launches under the contract: rotate and the reduction's combine, finish and place.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_flashnext_exl3_tppf_ref.zig");
const rows_ref = @import("contract_flashnext_exl3_tppf_rows_ref.zig");

const g = core.contract.gates;
const pk_ = flash.exl3.prefill_k;
const Fault = ref.Fault;

const images = struct {
    pub const controls align(16) = @embedFile("fatbin_contract_controls").*;
};

/// A case over a rig: its inputs uploaded, its outputs poisoned, the launch of `Case`'s kernel or of a control.
fn Dev(comptime Case: type, comptime launch: anytype) type {
    return struct {
        const Self = @This();
        rig: rig_.Rig,
        f: cuda.Function,
        case: Case,
        ptrs: [Case.n_in + Case.n_out]u64 = undefined,

        pub fn deinit(self: *Self) void {
            self.case.deinit();
            self.rig.deinit();
        }

        const vt = g.hooks(Self);

        pub fn kernel(self: *Self) g.Kernel {
            return .{ .spec = self.case.spec(), .ctx = self, .vt = &vt, .control = ref.controlGate(Case, self.case.fault) };
        }

        pub fn prepare(self: *Self, shape: core.contract.spec.Shape, fill: g.Fill, seed: u64) !bool {
            if (!try self.case.prepare(shape, fill, seed)) return false;
            self.rig.reset();
            var n: usize = 0;
            for (self.case.inputs()) |bytes| {
                self.ptrs[n] = try self.rig.input(bytes);
                n += 1;
            }
            for (self.case.outputLens()) |len| {
                self.ptrs[n] = try self.rig.output(len);
                n += 1;
            }
            return true;
        }

        pub fn run(self: *Self) !void {
            try launch(self.f, self.rig.s, &self.ptrs, &self.case);
            try self.rig.s.synchronize();
        }

        pub fn count(self: *Self) usize {
            return self.case.count();
        }

        pub fn reference(self: *Self, want: []f64, mag: []f64) !void {
            return self.case.reference(want, mag);
        }

        pub fn simulate(self: *Self, acc: core.contract.spec.Accum, order: g.Order, out: []f64) !void {
            return self.case.simulate(acc, order, out);
        }

        /// Guards and inputs as prepared, then the case's own check of the outputs.
        pub fn check(self: *Self, gate: g.Gate) !?[]const u8 {
            if (!try self.rig.intact()) return "a payload guard changed";
            const seen = try self.rig.inputBytes();
            var at: usize = 0;
            for (self.case.inputs()) |bytes| {
                if (!std.mem.eql(u8, seen[at..][0..bytes.len], bytes)) return "an immutable input changed";
                at += bytes.len;
            }
            return self.case.verify(try self.rig.outputBytes(), gate);
        }
    };
}

fn i32of(v: usize) i32 {
    return @intCast(v);
}

/// One launch a rank, all into one gather buffer (the engine's launches of its ranks, one device).
fn launchCombine(f: cuda.Function, s: cuda.Stream, p: []const u64, x: *const ref.Combine) !void {
    for (0..x.shape.k) |rank| {
        var a: cuda.Args = .{};
        for (p[0..4]) |ptr| a.add(ptr);
        for ([_]usize{ ref.slots, x.shape.n, x.shape.rows, x.sub, x.shape.k, rank, x.m }) |v| a.add(i32of(v));
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(x.shape.rows) }, .block = .{ .x = 256 } }, s, &a);
    }
}

fn launchFinish(f: cuda.Function, s: cuda.Stream, p: []const u64, x: *const ref.Finish) !void {
    var a: cuda.Args = .{};
    a.add(p[0]);
    a.add(p[1]);
    for ([_]usize{ x.shape.n, x.shape.rows, x.shape.k, x.m }) |v| a.add(i32of(v));
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(x.shape.rows) }, .block = .{ .x = 256 } }, s, &a);
}

fn launchPlace(f: cuda.Function, s: cuda.Stream, p: []const u64, x: *const rows_ref.Place) !void {
    var a: cuda.Args = .{};
    for (p[0..3]) |ptr| a.add(ptr);
    a.add(i32of(ref.slots));
    a.add(i32of(x.shape.n));
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(x.shape.rows) }, .block = .{ .x = 256 } }, s, &a);
}

fn launchRotate(f: cuda.Function, s: cuda.Stream, p: []const u64, x: *const rows_ref.Rotate) !void {
    var a: cuda.Args = .{};
    a.add(p[0]);
    a.add(i32of(x.shape.n));
    a.add(p[1]);
    a.add(p[2]);
    a.add(i32of(ref.slots));
    a.add(p[3]);
    a.add(p[4]);
    a.add(p[5]);
    a.add(p[6]);
    a.add(i32of(x.shape.n));
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(x.plan.n_members), .y = @intCast(x.shape.n / 128), .z = 2 }, .block = .{ .x = 32 } }, s, &a);
}

const CombineDev = Dev(ref.Combine, launchCombine);
const FinishDev = Dev(ref.Finish, launchFinish);
const PlaceDev = Dev(rows_ref.Place, launchPlace);
const RotateDev = Dev(rows_ref.Rotate, launchRotate);

const Plan = struct { fault: Fault, symbol: [:0]const u8 };

const combine_plan = [_]Plan{ .{ .fault = .none, .symbol = "" }, .{ .fault = .none, .symbol = "" }, .{ .fault = .flat, .symbol = "tf_ctl_tppf_combine_flat" }, .{ .fault = .nan_dropped, .symbol = "tf_ctl_tppf_combine_nan_dropped" }, .{ .fault = .writes_input, .symbol = "tf_ctl_tppf_combine_writes_input" }, .{ .fault = .bf16_adds, .symbol = "tf_ctl_tppf_combine_bf16_adds" }, .{ .fault = .f32_partials, .symbol = "tf_ctl_tppf_combine_f32_partials" } };
const finish_plan = [_]Plan{ .{ .fault = .none, .symbol = "" }, .{ .fault = .flat, .symbol = "tf_ctl_tppf_finish_flat" }, .{ .fault = .bf16_adds, .symbol = "tf_ctl_tppf_finish_bf16_adds" }, .{ .fault = .writes_input, .symbol = "tf_ctl_tppf_finish_writes_input" }, .{ .fault = .f32_partials, .symbol = "tf_ctl_tppf_finish_f32_partials" } };
const place_plan = [_]Plan{ .{ .fault = .none, .symbol = "" }, .{ .fault = .nan_dropped, .symbol = "tf_ctl_tppf_place_nan_dropped" }, .{ .fault = .writes_input, .symbol = "tf_ctl_tppf_place_writes_input" }, .{ .fault = .wrong, .symbol = "tf_ctl_tppf_place_wrong" } };
const rotate_plan = [_]Plan{ .{ .fault = .none, .symbol = "" }, .{ .fault = .nan_dropped, .symbol = "tf_ctl_tppf_rotate_nan_dropped" }, .{ .fault = .writes_input, .symbol = "tf_ctl_tppf_rotate_writes_input" }, .{ .fault = .wrong, .symbol = "tf_ctl_tppf_rotate_wrong" }, .{ .fault = .bf16_adds, .symbol = "tf_ctl_tppf_rotate_bf16_adds" } };

/// The four launches (combine in its two region layouts) and each one's planted faults, over the prefill image.
pub const Set = struct {
    a: std.mem.Allocator,
    k: pk_.Kernels,
    combines: [combine_plan.len]CombineDev,
    finishes: [finish_plan.len]FinishDev,
    places: [place_plan.len]PlaceDev,
    rotates: [rotate_plan.len]RotateDev,

    fn function(self: *const Set, controls: cuda.Module, comptime shipped: []const u8, plan: Plan) !cuda.Function {
        return if (plan.fault == .none) @field(self.k, shipped) else controls.function(plan.symbol);
    }

    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, controls: cuda.Module) !*Set {
        const self = try a.create(Set);
        errdefer a.destroy(self);
        self.a = a;
        self.k = try pk_.Kernels.load(gpu.d);
        errdefer self.k.unload();
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        for (combine_plan, &self.combines, 0..) |plan, *x, i| x.* = .{ .rig = rig, .f = try self.function(controls, "tp_combine", plan), .case = .{ .gpa = gpu.gpa, .blocks = i == 1, .fault = plan.fault } };
        for (finish_plan, &self.finishes) |plan, *x| x.* = .{ .rig = rig, .f = try self.function(controls, "tp_finish", plan), .case = .{ .gpa = gpu.gpa, .fault = plan.fault } };
        for (place_plan, &self.places) |plan, *x| x.* = .{ .rig = rig, .f = try self.function(controls, "tp_place", plan), .case = .{ .gpa = gpu.gpa, .fault = plan.fault } };
        for (rotate_plan, &self.rotates) |plan, *x| x.* = .{ .rig = rig, .f = try self.function(controls, "rotate", plan), .case = .{ .gpa = gpu.gpa, .fault = plan.fault } };
        return self;
    }

    pub fn destroy(self: *Set) void {
        for (&self.combines) |*x| x.deinit();
        for (&self.finishes) |*x| x.deinit();
        for (&self.places) |*x| x.deinit();
        for (&self.rotates) |*x| x.deinit();
        self.k.unload();
        self.a.destroy(self);
    }

    pub fn add(self: *Set, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !void {
        for (&self.rotates) |*x| try list.append(a, x.kernel());
        for (&self.combines) |*x| try list.append(a, x.kernel());
        for (&self.finishes) |*x| try list.append(a, x.kernel());
        for (&self.places) |*x| try list.append(a, x.kernel());
    }
};

/// tf-cuda-test exl3-tppf-contract: just these launches through the gates and their controls.
pub fn run(gpu: check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var controls = try cuda.Module.load(gpu.d, &images.controls);
    defer controls.unload();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(gpu.gpa);
    const set = try Set.create(gpu.gpa, gpu, s, controls);
    defer set.destroy();
    try set.add(gpu.gpa, &list);
    var t: g.Tally = .{};
    try g.runAll(gpu.gpa, list.items, 0x74707066, &t);
    std.debug.print("SUMMARY exl3 tppf {d} kernels: {d} pass gates 1-4, {d} fail; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.controls, t.caught, t.missed });
    try check.expect(t.problems() == 0, "exl3-tppf-contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}
