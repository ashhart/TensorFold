//! One host-checkable arithmetic or stream-isolation mutation for every modified Mini DFlash 2 Metal kernel.
const std = @import("std");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;

pub const Kind = enum {
    conv,
    ctx_kv,
    block_qkv,
    attend_ring,
    attend_reach,
    attend_masked,
    attend_merge,
    walk,
};

const Edit = struct { from: []const u8, to: []const u8 };

fn edit(kind: Kind) Edit {
    return switch (kind) {
        .conv => .{
            .from = "if (r % a.block >= 1) y = float(bfloat(y + float(bfloat(c1 * float(X[(size_t)(r - 1) * a.width + ch])))));",
            .to = "if (r >= 1) y = float(bfloat(y + float(bfloat(c1 * float(X[(size_t)(r - 1) * a.width + ch])))));",
        },
        .ctx_kv => .{
            .from = "device bfloat* kd = DF2_RING(K, s) + (slot * DF2_KVH + head) * DF2_D;",
            .to = "device bfloat* kd = K0 + (slot * DF2_KVH + head) * DF2_D;",
        },
        .block_qkv => .{
            .from = "const uint pos = a.pos0[stream] + (row % a.block);",
            .to = "const uint pos = a.pos0[0] + (row % a.block);",
        },
        .attend_ring => .{
            .from = "const device bfloat* K_ring = DF2_RING(K, s);",
            .to = "const device bfloat* K_ring = K0;",
        },
        .attend_reach => .{
            .from = "const uint reach = st.anchor + (row % a.block) + 1;",
            .to = "const uint reach = st.anchor + row + 1;",
        },
        .attend_masked => .{
            .from = "const float4 x = seen ? float4(((const device bfloat4*)vsrc)[lane]) : float4(0.0f);",
            .to = "const float4 x = float4(((const device bfloat4*)vsrc)[lane]);",
        },
        .attend_merge => .{
            .from = "const uint chunks = streams[s].chunks;",
            .to = "const uint chunks = streams[0].chunks;",
        },
        .walk => .{
            .from = "const size_t row_base = (size_t)stream * a.stride + a.row0;",
            .to = "const size_t row_base = a.row0;",
        },
    };
}

pub fn name(kind: Kind) []const u8 {
    return switch (kind) {
        inline else => |k| "control: metal inkling2 dflash2_batch " ++ @tagName(k),
    };
}

pub fn source(a: std.mem.Allocator, production: []const u8, kind: Kind) ![]u8 {
    const e = edit(kind);
    if (std.mem.count(u8, production, e.from) != 1) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, production, e.from, e.to);
}

test "every DFlash2 control applies once and only mutates its isolation anchor" {
    const production = @import("inkling2_metal").inkling2_dflash2;
    const a = std.testing.allocator;
    for (std.enums.values(Kind)) |kind| {
        const e = edit(kind);
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, production, e.from));
        const changed = try source(a, production, kind);
        defer a.free(changed);
        const at = std.mem.indexOf(u8, production, e.from).?;
        try std.testing.expectEqualStrings(production[0..at], changed[0..at]);
        try std.testing.expectEqualStrings(e.to, changed[at..][0..e.to.len]);
        try std.testing.expectEqualStrings(production[at + e.from.len ..], changed[at + e.to.len ..]);
        try std.testing.expectError(error.ControlEditMissing, source(a, changed, kind));
        try std.testing.expectError(error.ControlEditMissing, source(a, "", kind));
    }
}

test "repeated DFlash2 control anchors are refused" {
    const production = @import("inkling2_metal").inkling2_dflash2;
    const a = std.testing.allocator;
    const twice = try std.mem.concat(a, u8, &.{ production, production });
    defer a.free(twice);
    for (std.enums.values(Kind)) |kind| try std.testing.expectError(error.ControlEditMissing, source(a, twice, kind));
}

test "DFlash2 batch, attention, and walk probe registration rolls back cleanly on OOM" {
    const production = @import("inkling2_metal").inkling2_dflash2;
    const StubRig = struct {
        pub fn outputBytes(_: @This()) ![]const u8 {
            return &.{};
        }
        pub fn inputBytes(_: @This()) ![]const u8 {
            return &.{};
        }
        pub fn intact(_: @This()) !bool {
            return true;
        }
        pub fn overlap(_: @This()) bool {
            return false;
        }
    };
    const FakeProbe = struct {
        a: std.mem.Allocator,
        payload: []u8,
        deinited: *usize,
        label: []const u8,
        ctl_gate: ?g.Gate,
        rig: StubRig = .{},

        fn init(a: std.mem.Allocator, kind: ?Kind, ctl_gate: ?g.Gate, label: []const u8, deinited: *usize) !@This() {
            if (kind) |k| {
                const text = try source(a, production, k);
                defer a.free(text);
                return .{ .a = a, .payload = try a.alloc(u8, 24), .deinited = deinited, .label = label, .ctl_gate = ctl_gate };
            }
            return .{ .a = a, .payload = try a.alloc(u8, 24), .deinited = deinited, .label = label, .ctl_gate = ctl_gate };
        }
        fn deinit(self: *@This()) void {
            self.deinited.* += 1;
            self.a.free(self.payload);
        }
        pub fn prepare(_: *@This(), _: c.Shape, _: g.Fill, _: u64) !bool {
            return false;
        }
        pub fn run(_: *@This()) !void {}
        pub fn count(_: *@This()) usize {
            return 0;
        }
        pub fn reference(_: *@This(), _: []f64, _: []f64) !void {}
        fn depth(_: c.Shape) usize {
            return 1;
        }
        const vt = g.hooks(@This());
        fn kernel(self: *@This()) g.Kernel {
            return .{
                .ctx = self,
                .vt = &vt,
                .control = self.ctl_gate,
                .spec = .{ .name = self.label, .accum = .none, .depth = depth, .out = .bf16, .served = &.{} },
            };
        }
    };
    const Case = struct { kind: ?Kind, gate: ?g.Gate, label: []const u8, use_walks: bool };
    const cases = [_]Case{
        .{ .kind = null, .gate = null, .label = "metal inkling2 dflash2 conv (block isolation)", .use_walks = false },
        .{ .kind = .conv, .gate = .shapes, .label = name(.conv), .use_walks = false },
        .{ .kind = .ctx_kv, .gate = .shapes, .label = name(.ctx_kv), .use_walks = false },
        .{ .kind = .block_qkv, .gate = .shapes, .label = name(.block_qkv), .use_walks = false },
        .{ .kind = null, .gate = null, .label = "metal inkling2 dflash2 attention (1 stream, wrapped ring) +merge", .use_walks = false },
        .{ .kind = null, .gate = null, .label = "metal inkling2 dflash2 attention (3 streams, varied lengths) +merge", .use_walks = false },
        .{ .kind = .attend_masked, .gate = .nonfinite, .label = name(.attend_masked), .use_walks = false },
        .{ .kind = .attend_ring, .gate = .shapes, .label = name(.attend_ring), .use_walks = false },
        .{ .kind = .attend_reach, .gate = .shapes, .label = name(.attend_reach), .use_walks = false },
        .{ .kind = .attend_merge, .gate = .shapes, .label = name(.attend_merge), .use_walks = false },
        .{ .kind = null, .gate = null, .label = "metal inkling2 dflash2 walk (3 streams)", .use_walks = true },
        .{ .kind = .walk, .gate = .shapes, .label = name(.walk), .use_walks = true },
    };
    const runner = struct {
        fn add(a: std.mem.Allocator, owners: *std.ArrayList(*FakeProbe), kernels: *std.ArrayList(g.Kernel), case: Case, deinited: *usize) !void {
            const p = try a.create(FakeProbe);
            errdefer a.destroy(p);
            p.* = try FakeProbe.init(a, case.kind, case.gate, case.label, deinited);
            errdefer p.deinit();
            try contract.register(a, owners, kernels, p, p.kernel());
        }
        fn run(case: Case, fail_at: ?usize, expect_owners_grown: bool) !usize {
            var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{
                .fail_index = fail_at orelse std.math.maxInt(usize),
                .resize_fail_index = 0,
            });
            const fa = failing.allocator();
            var existing_deinits: usize = 0;
            var candidate_deinits: usize = 0;
            {
                const WalkSuite = struct {
                    a: std.mem.Allocator,
                    kernels: *std.ArrayList(g.Kernel),
                    walks: std.ArrayList(*FakeProbe) = .empty,
                    fn deinit(s: *@This()) void {
                        for (s.walks.items) |w| {
                            w.deinit();
                            s.a.destroy(w);
                        }
                        s.walks.deinit(s.a);
                    }
                };
                const ProbeSuite = struct {
                    a: std.mem.Allocator,
                    kernels: *std.ArrayList(g.Kernel),
                    probes: std.ArrayList(*FakeProbe) = .empty,
                    fn deinit(s: *@This()) void {
                        for (s.probes.items) |p| {
                            p.deinit();
                            s.a.destroy(p);
                        }
                        s.probes.deinit(s.a);
                    }
                };
                var kernels: std.ArrayList(g.Kernel) = .empty;
                var walk_suite: WalkSuite = .{ .a = fa, .kernels = &kernels };
                var probe_suite: ProbeSuite = .{ .a = fa, .kernels = &kernels };
                defer {
                    walk_suite.deinit();
                    probe_suite.deinit();
                    kernels.deinit(fa);
                }
                const owners = if (case.use_walks) &walk_suite.walks else &probe_suite.probes;
                try owners.ensureTotalCapacityPrecise(fa, 1);
                try kernels.ensureTotalCapacityPrecise(fa, 1);
                const base_case: Case = .{ .kind = null, .gate = null, .label = "existing", .use_walks = case.use_walks };
                try add(fa, owners, &kernels, base_case, &existing_deinits);
                const existing = owners.items[0];
                if (fail_at != null) {
                    try std.testing.expectError(error.OutOfMemory, add(fa, owners, &kernels, case, &candidate_deinits));
                    try std.testing.expect(failing.has_induced_failure);
                    try std.testing.expectEqual(@as(usize, 1), candidate_deinits);
                    try std.testing.expectEqual(@as(usize, 0), existing_deinits);
                    try std.testing.expectEqual(@as(usize, 1), owners.items.len);
                    try std.testing.expectEqual(@as(usize, 1), kernels.items.len);
                    if (expect_owners_grown) {
                        try std.testing.expect(owners.capacity > 1);
                    } else {
                        try std.testing.expectEqual(@as(usize, 1), owners.capacity);
                    }
                    try std.testing.expectEqual(@as(usize, 1), kernels.capacity);
                    try std.testing.expectEqual(existing, owners.items[0]);
                    try std.testing.expectEqual(@as(*anyopaque, existing), kernels.items[0].ctx);
                } else {
                    try add(fa, owners, &kernels, case, &candidate_deinits);
                    try std.testing.expectEqual(@as(usize, 2), owners.items.len);
                    try std.testing.expectEqual(@as(usize, 2), kernels.items.len);
                }
            }
            try std.testing.expectEqual(@as(usize, 1), existing_deinits);
            try std.testing.expectEqual(@as(usize, 1), candidate_deinits);
            try std.testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
            try std.testing.expectEqual(failing.allocations, failing.deallocations);
            return failing.alloc_index;
        }
    };
    for (cases) |case| {
        const total = try runner.run(case, null, false);
        _ = try runner.run(case, total - 2, false);
        _ = try runner.run(case, total - 1, true);
    }
}
