//! Arithmetic test backend for the shared production graph; this is never registered as a native engine.
const std = @import("std");
const m = @import("math.zig");
const cfg = @import("../../src/families/qwen4/config.zig");
const graph = @import("hybrid_graph");
const State = @import("state.zig").State;
pub const Ops = struct {
    pub const Tensor = m.Tensor;
    a: std.mem.Allocator,
    c: *const cfg.Config,
    pub fn embed(o: *Ops, tokens: []const u32, hidden: usize, streams: usize) !Tensor {
        const w = try m.source(o.a, "model.embed_tokens.weight", o.c.vocab * hidden, false, false);
        const out = try m.tensor(o.a, tokens.len, hidden * streams);
        for (tokens, 0..) |token, row| for (0..streams) |s| {
            @memcpy(out.row(row)[s * hidden ..][0..hidden], w[token * hidden ..][0..hidden]);
        };
        return out;
    }
    pub fn name(o: *Ops, layer: usize, part: []const u8) ![]const u8 {
        return std.fmt.allocPrint(o.a, "model.layers.{d}.{s}", .{ layer, part });
    }
    fn hcName(o: *Ops, layer: usize, b: graph.Boundary) ![]const u8 {
        return if (b == .readout) "model.hyper_connection_mixer" else o.name(layer, if (b == .attention) "attn_hyper_connection" else "mlp_hyper_connection");
    }
    pub fn hcNorm(o: *Ops, layer: usize, b: graph.Boundary, h: Tensor, hidden: usize, eps: f32) !Tensor {
        return m.norm(o.a, h, try std.fmt.allocPrint(o.a, "{s}.hc_norm", .{try o.hcName(layer, b)}), hidden, eps);
    }
    pub fn hcProject(o: *Ops, layer: usize, b: graph.Boundary, p: graph.Projection, x: Tensor) !Tensor {
        const tail: []const u8 = switch (p) {
            .down => "input_mix_weight_down",
            .up => "input_mix_weight_up",
            .inject => "block_inject_weight",
            else => unreachable,
        };
        const n = switch (p) {
            .down => o.c.lowrank,
            .up => o.c.wide(),
            .inject => o.c.streams,
            else => unreachable,
        };
        return m.project(o.a, x, try std.fmt.allocPrint(o.a, "{s}.{s}", .{ try o.hcName(layer, b), tail }), n);
    }
    pub fn hcActivate(o: *Ops, x: Tensor, streams: usize) !Tensor {
        const out = try m.tensor(o.a, x.rows, x.width);
        for (x.bits, out.bits) |v, *y| y.* = m.bf(m.silu(m.round(m.f(v) / @as(f32, @floatFromInt(streams)))));
        return out;
    }
    pub fn hcInjection(o: *Ops, x: Tensor, streams: usize) !Tensor {
        const out = try m.tensor(o.a, x.rows, x.width);
        for (x.bits, out.bits) |v, *y| y.* = m.bf(2 * m.round(m.sig(m.round(m.f(v) / @as(f32, @floatFromInt(streams))))));
        return out;
    }
    pub fn hcMix(o: *Ops, up: Tensor, normed: Tensor, streams: usize) !Tensor {
        const out = try m.tensor(o.a, up.rows, o.c.hidden);
        for (0..up.rows) |r| for (0..out.width) |d| {
            var sum: f32 = 0;
            for (0..streams) |s| {
                const i = s * out.width + d;
                sum += m.round(m.round(m.sig(m.f(up.row(r)[i]))) * m.f(normed.row(r)[i]));
            }
            out.row(r)[d] = m.bf(sum / @as(f32, @floatFromInt(streams)));
        };
        return out;
    }
    pub fn writeback(o: *Ops, h: Tensor, branch: Tensor, injection: Tensor) !Tensor {
        const out = try m.tensor(o.a, h.rows, h.width);
        for (0..h.rows) |r| for (0..h.width) |i| {
            out.row(r)[i] = m.bf(m.f(h.row(r)[i]) + m.round(m.f(branch.row(r)[i % o.c.hidden]) * m.f(injection.row(r)[i / o.c.hidden])));
        };
        return out;
    }
    pub fn denseProject(o: *Ops, layer: usize, p: graph.Projection, x: Tensor) !Tensor {
        return m.project(o.a, x, try o.name(layer, switch (p) {
            .gate => "mlp.gate_proj",
            .value => "mlp.up_proj",
            .output => "mlp.down_proj",
            else => unreachable,
        }), if (p == .output) o.c.hidden else o.c.intermediate);
    }
    pub fn swiglu(o: *Ops, gate: Tensor, up: Tensor) !Tensor {
        const out = try m.tensor(o.a, gate.rows, gate.width);
        for (gate.bits, up.bits, out.bits) |g, u, *y| y.* = m.bf(m.round(m.silu(m.f(g))) * m.f(u));
        return out;
    }
    pub fn head(o: *Ops, x: Tensor) !Tensor {
        return m.project(o.a, x, if (o.c.tie_embeddings) "model.embed_tokens" else "lm_head", o.c.vocab);
    }
    pub fn routed(_: *Ops, _: usize, _: Tensor) !Tensor {
        return error.UnexpectedRoutedLayer;
    }
    pub fn delta(o: *Ops, layer: usize, state: *State, x: Tensor) !Tensor {
        return @import("mixers.zig").delta(o, layer, state, x);
    }
    pub fn indexed(o: *Ops, layer: usize, state: *State, x: Tensor) !Tensor {
        return @import("mixers.zig").indexed(o, layer, state, x);
    }
    pub fn ple(o: *Ops, layer: usize, state: *State, tokens: []const u32, h: Tensor) !Tensor {
        return @import("ple.zig").apply(o, layer, state, tokens, h);
    }
};
