//! Independent binary64 replay freezes the native stores while perturbing only the selected residual row.
const std = @import("std");
const core = @import("core");
const n = @import("../../../tests/qwen4/numeric_gpu.zig");
fn bf(x: u16) f64 {
    return @as(f32, @bitCast(@as(u32, x) << 16));
}
fn sig(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}
fn silu(x: f64) f64 {
    return x * sig(x);
}
pub fn Point(comptime T: type) type {
    return struct {
        const Self = @This();
        a: std.mem.Allocator,
        t: *T,
        offsets: std.ArrayList(f64) = .empty,
        frozen: bool = false,
        at: usize = 0,
        mem: std.mem.Allocator = undefined,
        pub fn deinit(o: *Self) void {
            o.offsets.deinit(o.a);
        }
        fn cut(o: *Self, x: f64, want: f64) !f64 {
            const at = o.at;
            o.at += 1;
            if (!o.frozen) try o.offsets.append(o.a, want - x);
            return x + o.offsets.items[at];
        }
        fn round(o: *Self, x: f64) !f64 {
            return o.cut(x, bf(core.quant.convert.bf16(@floatCast(x))));
        }
        fn float(o: *Self, x: f64) !f64 {
            return o.cut(x, @as(f32, @floatCast(x)));
        }
        fn alloc(o: *Self, len: usize) ![]f64 {
            return o.mem.alloc(f64, len);
        }
        fn saved(o: *Self, l: ?usize, scope: []const u8, suffix: []const u8) ![]const u16 {
            var name: [192]u8 = undefined;
            return n.view(u16, try o.t.tape.get(try @import("node_trace.zig").label(&name, l, scope, suffix), o.t.rows));
        }
        fn anchor(o: *Self, x: []f64, l: ?usize, scope: []const u8, suffix: []const u8) ![]f64 {
            const saved_ = try o.saved(l, scope, suffix);
            if (saved_.len != x.len) return error.ReferenceShape;
            for (x, saved_) |*v, s| v.* = try o.cut(v.*, bf(s));
            return x;
        }
        fn matrix(o: *Self, x: []const f64, w: anytype) ![]f64 {
            const k: usize = @intCast(w.params.k);
            const width: usize = @intCast(w.params.n);
            const out = try o.alloc(o.t.rows * width);
            for (out, 0..) |*v, at| {
                const row = at / width;
                const col = at % width;
                var sum: f64 = 0;
                for (0..k) |i| {
                    const wi = col * k + i;
                    const weight: f64 = if (w.params.format == .bf16) bf(std.mem.bytesAsSlice(u16, w.weight)[wi]) else blk: {
                        const code = (w.weight[wi / 2] >> @as(u3, @intCast((wi % 2) * 4))) & 15;
                        const sc = std.mem.bytesAsSlice(u16, w.scales)[wi / 64];
                        const bi = std.mem.bytesAsSlice(u16, w.biases)[wi / 64];
                        break :blk @as(f32, @mulAdd(f32, @floatFromInt(code), @floatCast(bf(sc)), @floatCast(bf(bi))));
                    };
                    sum += x[row * k + i] * weight;
                }
                v.* = sum;
            }
            return out;
        }
        fn norm(o: *Self, x: []const f64, w: anytype, group: usize) ![]f64 {
            const out = try o.alloc(x.len);
            const weights = n.vector(w.buffer);
            var start: usize = 0;
            while (start < x.len) : (start += group) {
                var square: f64 = 0;
                for (x[start..][0..group]) |v| square += v * v;
                const inv = 1 / @sqrt(square / @as(f64, @floatFromInt(group)) + o.t.e.c.eps);
                for (0..group) |i| out[start + i] = x[start + i] * inv * weights[(start + i) % weights.len];
            }
            return out;
        }
        const HcResult = struct { mix: []f64, inject: []f64 };
        fn hc(o: *Self, h: []const f64, l: ?usize, scope: []const u8, w: anytype) !HcResult {
            const c = &o.t.e.c;
            const normalized = try o.anchor(try o.norm(h, w.norm, c.hidden), l, scope, "norm");
            const down = try o.anchor(try o.matrix(normalized, w.down), l, scope, "down");
            const low = try o.alloc(down.len);
            for (low, down) |*v, x| v.* = try o.round(silu(try o.round(x / @as(f64, @floatFromInt(c.streams)))));
            const up = try o.anchor(try o.matrix(low, w.up), l, scope, "up");
            const mixed = try o.alloc(o.t.rows * c.hidden);
            for (mixed, 0..) |*v, i| {
                const row = i / c.hidden;
                const d = i % c.hidden;
                var sum: f64 = 0;
                for (0..c.streams) |s| {
                    const at = (row * c.streams + s) * c.hidden + d;
                    sum += try o.round((try o.round(sig(up[at]))) * normalized[at]);
                }
                v.* = sum / @as(f64, @floatFromInt(c.streams));
            }
            _ = try o.anchor(mixed, l, scope, "mix");
            var inject: []f64 = &.{};
            if (w.inject) |wi| {
                const raw = try o.anchor(try o.matrix(normalized, wi), l, scope, "raw_inject");
                inject = try o.alloc(raw.len);
                for (inject, raw) |*v, x| v.* = 2 * (try o.round(sig(try o.round(x / @as(f64, @floatFromInt(c.streams))))));
                _ = try o.anchor(inject, l, scope, "injection");
            }
            return .{ .mix = mixed, .inject = inject };
        }
        fn write(o: *Self, h: []const f64, branch: []const f64, inject: []const f64, l: usize, scope: []const u8) ![]f64 {
            const c = &o.t.e.c;
            const out = try o.alloc(h.len);
            for (out, h, 0..) |*v, x, i| v.* = x + (try o.round(branch[(i / c.wide()) * c.hidden + i % c.hidden] * inject[(i / c.wide()) * c.streams + (i % c.wide()) / c.hidden]));
            return o.anchor(out, l, scope, "residual");
        }
        fn pack(o: *Self, x: []const f64, matrices: anytype, l: usize, scope: []const u8) ![]f64 {
            var width: usize = 0;
            inline for (matrices) |m| width += @intCast(m.params.n);
            const out = try o.alloc(o.t.rows * width);
            var col: usize = 0;
            inline for (matrices) |m| {
                const part = try o.matrix(x, m);
                const cols: usize = @intCast(m.params.n);
                for (0..o.t.rows) |r| @memcpy(out[r * width + col ..][0..cols], part[r * cols ..][0..cols]);
                col += cols;
            }
            return o.anchor(out, l, scope, "projections");
        }
        fn gdn(o: *Self, x: []const f64, l: usize, w: anytype) ![]f64 {
            const c = &o.t.e.c;
            const raw = try o.pack(x, &.{ w.qkv, w.z, w.b, w.a }, l, "linear_attn");
            const ch = c.convWidth();
            const nv = c.value_heads;
            const nk = c.key_heads;
            const pw = ch + nv * 130;
            const weights = n.vector(w.conv);
            const normw = n.vector(w.norm.buffer);
            const alog = n.vector(w.alog);
            const dt = n.vector(w.dt);
            const states = n.view(f32, try o.t.tape.indexed(l, "linear_attn.recurrent", o.t.rows));
            const state = try o.alloc(nv * 128 * 128);
            @memset(state, 0);
            const y = try o.alloc(o.t.rows * nv * 128);
            for (0..o.t.rows) |row| {
                const front = try o.alloc(ch);
                for (front, 0..) |*v, d| {
                    var sum: f64 = 0;
                    for (0..4) |tap| if (row + tap >= 3) {
                        sum += raw[(row + tap - 3) * pw + d] * weights[d * 4 + tap];
                    };
                    if (w.conv_bf16) sum = try o.round(sum);
                    v.* = try o.round(silu(sum));
                }
                const qk = try o.alloc(2 * nk * 128);
                for (0..2 * nk) |h| {
                    var ss: f64 = 1e-6;
                    for (front[h * 128 ..][0..128]) |v| ss += v * v;
                    const inv = (if (h < nk) 1 / @sqrt(@as(f64, 128)) else @as(f64, 1)) / @sqrt(ss);
                    for (0..128) |i| qk[h * 128 + i] = try o.float(front[h * 128 + i] * inv);
                }
                for (0..nv) |head| {
                    const ar = raw[row * pw + ch + nv * 129 + head] + dt[head];
                    const decay = @exp(-@exp(@as(f64, alog[head])) * @log(1 + @exp(ar)));
                    const beta = try o.round(sig(raw[row * pw + ch + nv * 128 + head]));
                    const kh = head / (nv / nk);
                    for (0..128) |d| {
                        const s = state[(head * 128 + d) * 128 ..][0..128];
                        var pred: f64 = 0;
                        for (s, 0..) |v, i| pred += decay * v * qk[(nk + kh) * 128 + i];
                        const correction = beta * (front[2 * nk * 128 + head * 128 + d] - pred);
                        var sum: f64 = 0;
                        for (s, 0..) |*v, i| {
                            v.* = try o.cut(decay * v.* + qk[(nk + kh) * 128 + i] * correction, states[((row * nv + head) * 128 + d) * 128 + i]);
                            sum += v.* * qk[kh * 128 + i];
                        }
                        y[(row * nv + head) * 128 + d] = try o.round(sum);
                    }
                    const ys = y[(row * nv + head) * 128 ..][0..128];
                    var ss: f64 = 0;
                    for (ys) |v| ss += v * v;
                    const inv = 1 / @sqrt(ss / 128 + c.eps);
                    for (ys, 0..) |*v, d| {
                        var weighted = (try o.round(v.* * inv)) * normw[d];
                        if (w.norm.mid_bf16) weighted = try o.round(weighted);
                        const z = raw[row * pw + ch + head * 128 + d];
                        v.* = weighted * (if (c.gate == .sigmoid) sig(z) else silu(z));
                    }
                }
            }
            _ = try o.anchor(y, l, "linear_attn", "gated");
            return o.anchor(try o.matrix(y, w.out), l, "linear_attn", "output");
        }
        fn prepare(o: *Self, raw: []const f64, rows_: usize, heads: usize, stride: usize, offset: usize, hs: usize, w: anytype) ![]f64 {
            const c = &o.t.e.c;
            const d = c.head_dim;
            const half = c.rotary / 2;
            const out = try o.alloc(rows_ * heads * d);
            const nw = n.vector(w.buffer);
            for (0..rows_) |row| for (0..heads) |head| {
                const x = raw[row * stride + offset + head * hs ..][0..d];
                const y = out[(row * heads + head) * d ..][0..d];
                var ss: f64 = 0;
                for (x) |v| ss += v * v;
                const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(d)) + c.eps);
                for (x, y, 0..) |v, *z, i| z.* = try o.round(v * inv * nw[i]);
                for (0..half) |i| {
                    const freq: f32 = @floatCast(std.math.pow(f64, c.theta, -2 * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(c.rotary))));
                    const angle = @as(f32, @floatFromInt(row)) * freq;
                    const co = @as(f64, @cos(angle));
                    const si = @as(f64, @sin(angle));
                    const left = y[i];
                    const right = y[i + half];
                    y[i] = co * left - si * right;
                    y[i + half] = si * left + co * right;
                }
            };
            return out;
        }
        fn qsa(o: *Self, x: []const f64, l: usize, w: anytype) ![]f64 {
            const c = &o.t.e.c;
            const rows_ = o.t.rows;
            const hd = c.head_dim;
            const h = c.heads;
            const kv = c.kv_heads;
            const raw = try o.pack(x, &.{ w.q, w.k, w.v, w.index }, l, "self_attn");
            const stride = raw.len / rows_;
            const q = try o.anchor(try o.prepare(raw, rows_, h, stride, 0, 2 * hd, w.qn), l, "self_attn", "query");
            const k = try o.anchor(try o.prepare(raw, rows_, kv, stride, 2 * h * hd, hd, w.kn), l, "self_attn", "key");
            const v = try o.alloc(rows_ * kv * hd);
            for (0..rows_) |r| @memcpy(v[r * kv * hd ..][0 .. kv * hd], raw[r * stride + 2 * h * hd + kv * hd ..][0 .. kv * hd]);
            const counts = n.view(i32, try o.t.tape.indexed(l, "self_attn.selection_count", rows_));
            const sparse = n.view(i32, try o.t.tape.indexed(l, "self_attn.selection_sparse", rows_));
            const ids = n.view(i32, try o.t.tape.raw(l, "self_attn.selected_ids", rows_));
            const y = try o.alloc(rows_ * h * hd);
            @memset(y, 0);
            for (0..rows_) |row| for (0..h) |head| {
                const number: usize = @intCast(counts[row]);
                const selected = try o.mem.alloc(usize, number);
                const scores = try o.alloc(number);
                const kh = head / (h / kv);
                var maximum: f64 = -std.math.inf(f64);
                for (selected, scores, 0..) |*key, *s, j| {
                    key.* = if (sparse[row] != 0) @intCast(ids[row * o.t.ids.len + j]) else j;
                    s.* = 0;
                    for (0..hd) |d| s.* += q[(row * h + head) * hd + d] * k[(key.* * kv + kh) * hd + d];
                    s.* /= @sqrt(@as(f64, @floatFromInt(hd)));
                    maximum = @max(maximum, s.*);
                }
                var total: f64 = 0;
                for (scores) |s| total += @exp(s - maximum);
                for (selected, scores, 0..) |key, s, j| {
                    var tm: f64 = -std.math.inf(f64);
                    for (scores[(j / 512) * 512 .. @min(number, (j / 32 + 1) * 32)]) |v_| tm = @max(tm, v_);
                    const weight = if (o.t.e.gpu.probability_bf16) (try o.round(@exp(s - tm))) * @exp(tm - maximum) / total else @exp(s - maximum) / total;
                    for (0..hd) |d| y[(row * h + head) * hd + d] += weight * v[(key * kv + kh) * hd + d];
                }
            };
            _ = try o.anchor(y, l, "self_attn", "attention");
            for (y, 0..) |*v_, at| {
                const row = at / (h * hd);
                const head = (at / hd) % h;
                const d = at % hd;
                v_.* *= try o.round(sig(raw[row * stride + head * 2 * hd + hd + d]));
            }
            _ = try o.anchor(y, l, "self_attn", "gated");
            return o.anchor(try o.matrix(y, w.out), l, "self_attn", "output");
        }
        pub fn loss(o: *Self, layer: usize, row: usize, direction: []const f32, epsilon: f64) !f64 {
            var arena: std.heap.ArenaAllocator = .init(o.a);
            defer arena.deinit();
            o.mem = arena.allocator();
            o.at = 0;
            const t = o.t;
            const c = &t.e.c;
            const embed = try t.tape.get("embed", t.rows);
            const h0 = try o.alloc(t.rows * c.wide());
            for (h0, n.view(u16, embed)) |*v, b| v.* = bf(b);
            var h = h0;
            for (t.e.model.layers, 0..) |w, l| {
                const ahc = try o.hc(h, l, "attn_hyper_connection", w.attention);
                const branch = switch (w.mixer) {
                    .linear => |m| try o.gdn(ahc.mix, l, m),
                    .indexed => |m| try o.qsa(ahc.mix, l, m),
                };
                h = try o.write(h, branch, ahc.inject, l, "attn_hyper_connection");
                const mhc = try o.hc(h, l, "mlp_hyper_connection", w.mlp);
                const gate = try o.anchor(try o.matrix(mhc.mix, w.dense.gate), l, "mlp", "gate");
                const up = try o.anchor(try o.matrix(mhc.mix, w.dense.up), l, "mlp", "up");
                const act = try o.alloc(gate.len);
                for (act, gate, up) |*v, g, u| v.* = (try o.round(silu(g))) * u;
                _ = try o.anchor(act, l, "mlp", "act");
                const output = try o.anchor(try o.matrix(act, w.dense.down), l, "mlp", "pocket_output");
                if (l == layer) for (output[row * c.hidden ..][0..c.hidden], direction) |*v, d| {
                    v.* += epsilon * d;
                };
                h = try o.write(h, output, mhc.inject, l, "mlp_hyper_connection");
            }
            const mixed = try o.hc(h, null, "model.hyper_connection_mixer", t.e.model.readout);
            const logits = try o.anchor(try o.matrix(mixed.mix, t.e.model.head), null, "", "head");
            var sum: f64 = 0;
            for (t.seq.start - 1..t.rows) |r| {
                const scores = logits[r * c.vocab ..][0..c.vocab];
                var mx: f64 = -std.math.inf(f64);
                for (scores) |v| mx = @max(mx, v);
                var total: f64 = 0;
                for (scores) |v| total += @exp(v - mx);
                sum += mx + @log(total) - scores[t.seq.ids[r + 1]];
            }
            o.frozen = true;
            if (o.at != o.offsets.items.len) return error.ReferenceCuts;
            return sum / @as(f64, @floatFromInt(t.seq.ids.len - t.seq.start));
        }
    };
}
