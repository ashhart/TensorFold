//! Unlimited-OCR-2's CUDA route under the contract: router block keys and picks, MoE gate and combine, greedy pick.
const std = @import("std");
const core = @import("core");
const uocr = @import("contract_uocr.zig");
const base = @import("contract_uocr_route_base.zig");
const Route = @import("contract_uocr_route_pick.zig").Route;
const gate = @import("contract_uocr_route_gate.zig");

const g = core.contract.gates;

const keys_name = "cuda uocr route tf_uocr_block_keys block 32 router 128";
const route_name = "cuda uocr route tf_uocr_route ranked 16 read 12 blocks to 4096";
const gate_name = "cuda uocr route tf_uocr_moe_gate experts 64 top 6 hidden 1280";
const pick_name = "cuda uocr route tf_uocr_argmax vocab 129280";
const pick_eos_name = "cuda uocr route tf_uocr_argmax vocab 129280, EOS barred";

/// A planted control: the kernel it stands in for, its symbol and its registration's name (gate 3 must go red).
const Control = struct { name: []const u8, symbol: [:0]const u8, label: []const u8 };

const route_controls = [_]Control{
    .{ .name = "route", .symbol = "tf_ctl_uocr_route_tie_high", .label = "control: cuda uocr route tf_uocr_route, a tie goes to the higher block" },
    .{ .name = "route", .symbol = "tf_ctl_uocr_route_gap_short", .label = "control: cuda uocr route tf_uocr_route, the local gap one block short" },
    .{ .name = "route", .symbol = "tf_ctl_uocr_route_last_page", .label = "control: cuda uocr route tf_uocr_route, a block's last page left out" },
};
const gate_control: Control = .{ .name = "moe_gate", .symbol = "tf_ctl_uocr_moe_gate_tie_high", .label = "control: cuda uocr route tf_uocr_moe_gate, a tie goes to the higher expert" };
const pick_control: Control = .{ .name = "argmax", .symbol = "tf_ctl_uocr_argmax_tie_high", .label = "control: cuda uocr route tf_uocr_argmax, a tie goes to the higher id" };

/// The group's registrations, on the heap: the gates hold pointers to them.
pub const Group = struct {
    gpa: std.mem.Allocator,
    keys: *base.BlockKeys,
    routes: [1 + route_controls.len]*Route,
    gates: [2]*gate.MoeGate,
    combine: *gate.Combine,
    picks: [3]*gate.Argmax,

    fn make(comptime T: type, a: std.mem.Allocator, v: T) !*T {
        const x = try a.create(T);
        x.* = v;
        return x;
    }

    pub fn init(ctx: uocr.Ctx, list: *std.ArrayList(g.Kernel)) !*Group {
        const a = ctx.gpu.gpa;
        const grp = try a.create(Group);
        errdefer a.destroy(grp);
        grp.gpa = a;
        const k = ctx.k.*;
        grp.keys = try make(base.BlockKeys, a, .{ .a = a, .rig = ctx.rig(), .k = k, .label = keys_name });
        grp.routes[0] = try make(Route, a, .{ .a = a, .rig = ctx.rig(), .k = k, .label = route_name });
        inline for (route_controls, 1..) |ctl, i| grp.routes[i] = try make(Route, a, .{ .a = a, .rig = ctx.rig(), .k = try ctx.with(ctl.name, ctl.symbol), .label = ctl.label, .control = .shapes });
        grp.gates[0] = try make(gate.MoeGate, a, .{ .a = a, .rig = ctx.rig(), .k = k, .label = gate_name });
        grp.gates[1] = try make(gate.MoeGate, a, .{ .a = a, .rig = ctx.rig(), .k = try ctx.with(gate_control.name, gate_control.symbol), .label = gate_control.label, .control = .shapes });
        grp.combine = try make(gate.Combine, a, .{ .a = a, .rig = ctx.rig(), .k = k });
        grp.picks[0] = try make(gate.Argmax, a, .{ .a = a, .rig = ctx.rig(), .k = k, .label = pick_name, .bar_eos = false });
        grp.picks[1] = try make(gate.Argmax, a, .{ .a = a, .rig = ctx.rig(), .k = k, .label = pick_eos_name, .bar_eos = true });
        grp.picks[2] = try make(gate.Argmax, a, .{ .a = a, .rig = ctx.rig(), .k = try ctx.with(pick_control.name, pick_control.symbol), .label = pick_control.label, .bar_eos = true, .control = .shapes });
        try list.append(a, grp.keys.kernel());
        for (grp.routes) |x| try list.append(a, x.kernel());
        for (grp.gates) |x| try list.append(a, x.kernel());
        try list.append(a, grp.combine.kernel());
        for (grp.picks) |x| try list.append(a, x.kernel());
        return grp;
    }

    pub fn deinit(grp: *Group) void {
        const a = grp.gpa;
        grp.keys.deinit();
        a.destroy(grp.keys);
        for (grp.routes) |x| {
            x.deinit();
            a.destroy(x);
        }
        for (grp.gates) |x| {
            x.deinit();
            a.destroy(x);
        }
        grp.combine.deinit();
        a.destroy(grp.combine);
        for (grp.picks) |x| {
            x.deinit();
            a.destroy(x);
        }
        a.destroy(grp);
    }
};
