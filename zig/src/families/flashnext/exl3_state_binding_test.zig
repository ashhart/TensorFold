const std = @import("std");
const binding = @import("exl3_state_binding.zig");

fn requestFor(width: usize, output: bool) binding.Request {
    var request: binding.Request = .{
        .width = width,
        .chain = 10,
        .chain_to = 11,
        .geometry = .{ .grid = .{ 16, 8, 1 }, .block = .{ 128, 1, 1 } },
        .nonstate = .{ .gq = 100, .gk = 200, .gv = 300, .ggb = 400, .rows = @intCast(width), .ys = 500 },
        .count = 2,
    };
    request.layers[0] = .{ .layer = 0, .state_in = 1000, .state_out = if (output) 2000 else 0 };
    request.layers[1] = .{ .layer = 2, .state_in = 3000, .state_out = if (output) 4000 else 0 };
    return request;
}

fn flip(request: binding.Request) binding.Request {
    var target = request;
    for (target.layers[0..target.count]) |*mapping| std.mem.swap(u64, &mapping.state_in, &mapping.state_out);
    return target;
}

const FakeNode = struct {
    function: usize,
    geometry: binding.Geometry,
    values: binding.Parameters,
    pointers: [8]?*anyopaque = @splat(null),
    kernel: bool = true,
    extra: bool = false,

    fn header(self: *FakeNode) binding.Header {
        self.pointers[0] = &self.values.nonstate.gq;
        self.pointers[1] = &self.values.nonstate.gk;
        self.pointers[2] = &self.values.nonstate.gv;
        self.pointers[3] = &self.values.nonstate.ggb;
        self.pointers[4] = &self.values.state_in;
        const output = self.values.state_out != 0;
        if (output) self.pointers[5] = &self.values.state_out;
        self.pointers[if (output) 6 else 5] = &self.values.nonstate.rows;
        self.pointers[if (output) 7 else 6] = &self.values.nonstate.ys;
        return .{ .function = self.function, .geometry = self.geometry, .params = &self.pointers, .extra = self.extra };
    }
};

const Fake = struct {
    template: usize = 42,
    records: [4]FakeNode = undefined,
    ids: [4]usize = .{ 0, 1, 2, 3 },
    count: usize = 2,
    calls: usize = 0,
    fail_at: ?usize = null,
    query_failure: bool = false,
    writes: [2]binding.Parameters = undefined,

    fn init(request: binding.Request) Fake {
        var fake: Fake = .{};
        for (request.layers[0..request.count], 0..) |mapping, index| {
            fake.records[index] = .{
                .function = if (mapping.state_out == 0) request.chain else request.chain_to,
                .geometry = request.geometry,
                .values = .{ .nonstate = request.nonstate, .state_in = mapping.state_in, .state_out = mapping.state_out },
            };
        }
        return fake;
    }

    pub fn nodes(self: *Fake, template: usize) ![]usize {
        if (template != self.template) return error.WrongTemplate;
        return self.ids[0..self.count];
    }

    pub fn freeNodes(_: *Fake, _: []usize) void {}

    pub fn kernel(self: *Fake, node: usize) !?binding.Header {
        if (self.query_failure) return error.PriorAsyncFailure;
        if (!self.records[node].kernel) return null;
        return self.records[node].header();
    }

    pub fn set(self: *Fake, node: usize, function: usize, geometry: binding.Geometry, parameters: binding.Parameters) !void {
        const call = self.calls;
        self.calls += 1;
        if (self.fail_at == call) return error.SetterFailed;
        try std.testing.expectEqual(self.records[node].function, function);
        try std.testing.expectEqualDeep(self.records[node].geometry, geometry);
        self.writes[node] = parameters;
    }
};

test "exl3: state binding both widths and orientations preserve typed nonstate values" {
    for ([_]usize{ 1, 2 }) |width| {
        const request = requestFor(width, true);
        var fake = Fake.init(request);
        var state = try binding.Binding.init(fake.template, request);
        try state.load(&fake);
        const target = flip(request);
        try state.apply(&fake, target);
        for (target.layers[0..target.count], fake.writes) |mapping, written| {
            try std.testing.expectEqualDeep(request.nonstate, written.nonstate);
            try std.testing.expectEqual(mapping.state_in, written.state_in);
            try std.testing.expectEqual(mapping.state_out, written.state_out);
        }
        fake.records[0].values.nonstate.rows = 99;
        try state.apply(&fake, request);
        for (request.layers[0..request.count], fake.writes) |mapping, written| {
            try std.testing.expectEqualDeep(request.nonstate, written.nonstate);
            try std.testing.expectEqual(mapping.state_in, written.state_in);
            try std.testing.expectEqual(mapping.state_out, written.state_out);
        }
        try std.testing.expectEqual(@as(usize, 4), fake.calls);
    }
}

test "exl3: state binding schema7 reads c_int rows and u64 ys without a state output" {
    for ([_]usize{ 1, 2 }) |width| {
        const request = requestFor(width, false);
        var fake = Fake.init(request);
        var state = try binding.Binding.init(fake.template, request);
        try state.load(&fake);
        try state.apply(&fake, request);
        try std.testing.expectEqualDeep(request.nonstate, fake.writes[0].nonstate);
        var changed = request;
        changed.layers[0].state_in += 1;
        try std.testing.expectError(error.UnsupportedBinding, state.apply(&fake, changed));
        try std.testing.expectEqual(@as(usize, 2), fake.calls);
    }
}

test "exl3: state binding wrong functions are never inspected and missing duplicate mappings decline" {
    const request = requestFor(2, true);
    var fake = Fake.init(request);
    fake.records[0].function = 999;
    fake.records[0].extra = true;
    var state = try binding.Binding.init(fake.template, request);
    try std.testing.expectError(error.UnsupportedBinding, state.load(&fake));
    fake = Fake.init(request);
    fake.count = 1;
    try std.testing.expectError(error.UnsupportedBinding, state.load(&fake));
    fake = Fake.init(request);
    fake.records[1] = fake.records[0];
    try std.testing.expectError(error.UnsupportedBinding, state.load(&fake));
    var duplicate = request;
    duplicate.layers[1].layer = duplicate.layers[0].layer;
    try std.testing.expectError(error.UnsupportedBinding, binding.Binding.init(42, duplicate));
    duplicate = request;
    duplicate.layers[1].state_in = duplicate.layers[0].state_out;
    try std.testing.expectError(error.UnsupportedBinding, binding.Binding.init(42, duplicate));
    try std.testing.expectEqual(@as(usize, 0), fake.calls);
}

test "exl3: state binding full validation precedes every setter" {
    const request = requestFor(1, true);
    var fake = Fake.init(request);
    var state = try binding.Binding.init(fake.template, request);
    try state.load(&fake);
    var target = flip(request);
    target.layers[1].state_out += 1;
    try std.testing.expectError(error.UnsupportedBinding, state.apply(&fake, target));
    target = flip(request);
    target.nonstate.gk += 1;
    try std.testing.expectError(error.UnsupportedBinding, state.apply(&fake, target));
    target = flip(request);
    target.geometry.block[0] += 1;
    try std.testing.expectError(error.UnsupportedBinding, state.apply(&fake, target));
    target = flip(request);
    target.chain_to += 1;
    try std.testing.expectError(error.UnsupportedBinding, state.apply(&fake, target));
    try std.testing.expectEqual(@as(usize, 0), fake.calls);
    fake = Fake.init(request);
    fake.records[1].values.nonstate.ys += 1;
    state = try binding.Binding.init(fake.template, request);
    try std.testing.expectError(error.UnsupportedBinding, state.load(&fake));
    fake = Fake.init(request);
    fake.query_failure = true;
    try std.testing.expectError(error.PriorAsyncFailure, state.load(&fake));
}

test "exl3: state binding partial update poisons all future uses" {
    const request = requestFor(2, true);
    var fake = Fake.init(request);
    var state = try binding.Binding.init(fake.template, request);
    try state.load(&fake);
    fake.fail_at = 1;
    try std.testing.expectError(error.SetterFailed, state.apply(&fake, flip(request)));
    try std.testing.expect(state.poisoned);
    try std.testing.expectError(error.StateBindingPoisoned, state.apply(&fake, request));
    try std.testing.expectEqual(@as(usize, 2), fake.calls);
}

const Peer = struct {
    const Vote = struct { state: enum { ready, failed }, cohort: u64 };
    const Decision = struct { choice: enum { replay, capture, eager }, cohort: u64 };
    choice: Decision = .{ .choice = .replay, .cohort = 7 },
    observed: ?Vote = null,
    disconnected: bool = false,

    pub fn graphAgree(self: *Peer, _: u64, vote: Vote) !Decision {
        self.observed = vote;
        if (self.disconnected) return error.RankDisconnected;
        return self.choice;
    }
};

test "exl3: state binding peer failure disagreement and disconnect poison successful local updates" {
    const request = requestFor(2, true);
    for (0..4) |scenario| {
        var fake = Fake.init(request);
        var state = try binding.Binding.init(fake.template, request);
        try state.load(&fake);
        var peer: Peer = .{};
        switch (scenario) {
            0 => peer.choice.choice = .capture,
            1 => peer.choice.cohort = 8,
            2 => peer.disconnected = true,
            3 => fake.fail_at = 1,
            else => unreachable,
        }
        if (scenario == 2) {
            try std.testing.expectError(error.RankDisconnected, state.applyTogether(&fake, flip(request), &peer, 3, 7));
        } else try std.testing.expectError(error.StateRebindFailed, state.applyTogether(&fake, flip(request), &peer, 3, 7));
        try std.testing.expect(state.poisoned);
        try std.testing.expectEqual(scenario == 3, peer.observed.?.state == .failed);
    }
    var fake = Fake.init(request);
    var state = try binding.Binding.init(fake.template, request);
    try state.load(&fake);
    var peer: Peer = .{};
    try state.applyTogether(&fake, flip(request), &peer, 3, 7);
    try std.testing.expect(!state.poisoned);
}

test "exl3: state binding template replacement requires new metadata" {
    const request = requestFor(1, true);
    var fake = Fake.init(request);
    var state = try binding.Binding.init(fake.template, request);
    try state.load(&fake);
    state.invalidate();
    fake.template = 43;
    try std.testing.expectError(error.StateBindingPoisoned, state.load(&fake));
    try std.testing.expectError(error.StateBindingPoisoned, state.apply(&fake, flip(request)));
    state = try binding.Binding.init(fake.template, request);
    try std.testing.expectError(error.UnsupportedBinding, state.apply(&fake, flip(request)));
    try state.load(&fake);
    try state.apply(&fake, flip(request));
}

test "exl3: state binding keep hash excludes only layer state_out" {
    const Layer = struct { proj: u64 = 3, state_out: u64 = 4, hist: u64 = 5 };
    const Keep = struct { layers: [2]Layer = @splat(.{}), mixer: u64 = 8 };
    const original: Keep = .{};
    var changed = original;
    changed.layers[0].state_out += 1;
    try std.testing.expectEqual(binding.keepHash(original), binding.keepHash(changed));
    changed.layers[1].hist += 1;
    try std.testing.expect(binding.keepHash(original) != binding.keepHash(changed));
    changed = original;
    changed.mixer += 1;
    try std.testing.expect(binding.keepHash(original) != binding.keepHash(changed));
}
