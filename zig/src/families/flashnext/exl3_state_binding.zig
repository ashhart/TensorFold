const std = @import("std");

pub const max_layers = 48;
pub const Mapping = struct { layer: usize, state_in: u64, state_out: u64 };
pub const Geometry = struct {
    grid: [3]u32,
    block: [3]u32,
    shared: u32 = 0,
};
pub const Nonstate = struct { gq: u64, gk: u64, gv: u64, ggb: u64, rows: c_int, ys: u64 };
pub const Request = struct {
    width: usize,
    chain: usize,
    chain_to: usize,
    geometry: Geometry,
    nonstate: Nonstate,
    layers: [max_layers]Mapping = undefined,
    count: usize = 0,

    pub fn validate(self: Request) !void {
        if (self.width < 1 or self.width > 2 or self.nonstate.rows != self.width or
            self.chain == 0 or self.chain_to == 0 or self.chain == self.chain_to or
            self.count == 0 or self.count > max_layers) return error.UnsupportedBinding;
        for (self.layers[0..self.count], 0..) |mapping, index| {
            if (mapping.state_in == 0 or mapping.state_in == mapping.state_out) return error.UnsupportedBinding;
            for (self.layers[0..index]) |previous| {
                if (previous.layer == mapping.layer or previous.state_in == mapping.state_in or
                    previous.state_out == mapping.state_in or
                    (mapping.state_out != 0 and (previous.state_in == mapping.state_out or previous.state_out == mapping.state_out)))
                    return error.UnsupportedBinding;
            }
        }
    }

    pub fn compatible(self: Request, target: Request) bool {
        target.validate() catch return false;
        if (self.width != target.width or self.chain != target.chain or self.chain_to != target.chain_to or
            !std.meta.eql(self.geometry, target.geometry) or !std.meta.eql(self.nonstate, target.nonstate) or
            self.count != target.count) return false;
        for (self.layers[0..self.count], target.layers[0..target.count]) |before, after| {
            if (before.layer != after.layer) return false;
            const same = before.state_in == after.state_in and before.state_out == after.state_out;
            const flipped = before.state_out != 0 and before.state_in == after.state_out and before.state_out == after.state_in;
            if (!same and !flipped) return false;
        }
        return true;
    }
};

pub const Header = struct {
    function: usize,
    geometry: Geometry,
    params: ?[*]?*anyopaque,
    extra: bool = false,
    alternate_kernel: bool = false,
};

pub const Parameters = struct {
    nonstate: Nonstate,
    state_in: u64,
    state_out: u64,
};

fn argument(comptime Value: type, params: [*]?*anyopaque, index: usize) !Value {
    const pointer = params[index] orelse return error.UnsupportedBinding;
    var value: Value = undefined;
    @memcpy(std.mem.asBytes(&value), @as([*]const u8, @ptrCast(pointer))[0..@sizeOf(Value)]);
    return value;
}

pub const Binding = struct {
    template: usize,
    request: Request,
    nodes: [max_layers]usize = undefined,
    parameters: [max_layers]Parameters = undefined,
    loaded: bool = false,
    poisoned: bool = false,

    pub fn init(template: usize, request: Request) !Binding {
        try request.validate();
        if (template == 0) return error.UnsupportedBinding;
        return .{ .template = template, .request = request };
    }

    pub fn invalidate(self: *Binding) void {
        self.loaded = false;
        self.poisoned = true;
        self.template = 0;
    }

    pub fn load(self: *Binding, backend: anytype) !void {
        if (self.poisoned or self.template == 0) return error.StateBindingPoisoned;
        if (self.loaded) return;
        var matched: [max_layers]bool = @splat(false);
        const nodes = try backend.nodes(self.template);
        defer backend.freeNodes(nodes);
        for (nodes) |node| {
            const header = (try backend.kernel(node)) orelse continue;
            if (header.function != self.request.chain and header.function != self.request.chain_to) continue;
            if (header.extra or header.alternate_kernel or !std.meta.eql(header.geometry, self.request.geometry)) return error.UnsupportedBinding;
            const params = header.params orelse return error.UnsupportedBinding;
            const output = header.function == self.request.chain_to;
            const values: Parameters = .{
                .nonstate = .{
                    .gq = try argument(u64, params, 0),
                    .gk = try argument(u64, params, 1),
                    .gv = try argument(u64, params, 2),
                    .ggb = try argument(u64, params, 3),
                    .rows = try argument(c_int, params, if (output) 6 else 5),
                    .ys = try argument(u64, params, if (output) 7 else 6),
                },
                .state_in = try argument(u64, params, 4),
                .state_out = if (output) try argument(u64, params, 5) else 0,
            };
            if (!std.meta.eql(values.nonstate, self.request.nonstate)) return error.UnsupportedBinding;
            var found: ?usize = null;
            for (self.request.layers[0..self.request.count], 0..) |mapping, index| {
                if (values.state_in == mapping.state_in and values.state_out == mapping.state_out) {
                    if (matched[index] or found != null) return error.UnsupportedBinding;
                    found = index;
                }
            }
            const index = found orelse return error.UnsupportedBinding;
            matched[index] = true;
            self.nodes[index] = node;
            self.parameters[index] = values;
        }
        for (matched[0..self.request.count]) |present| if (!present) return error.UnsupportedBinding;
        self.loaded = true;
    }

    pub fn apply(self: *Binding, backend: anytype, target: Request) !void {
        if (self.poisoned) return error.StateBindingPoisoned;
        if (!self.loaded or !self.request.compatible(target)) return error.UnsupportedBinding;
        errdefer self.poisoned = true;
        for (target.layers[0..target.count], 0..) |mapping, index| {
            var parameters = self.parameters[index];
            parameters.state_in = mapping.state_in;
            parameters.state_out = mapping.state_out;
            try backend.set(self.nodes[index], if (mapping.state_out == 0) target.chain else target.chain_to, target.geometry, parameters);
        }
    }

    pub fn applyTogether(self: *Binding, backend: anytype, target: Request, peer: anytype, signature: u64, cohort: u64) !void {
        var succeeded = true;
        self.apply(backend, target) catch {
            succeeded = false;
        };
        errdefer self.poisoned = true;
        const agreement = try peer.graphAgree(signature, .{ .state = if (succeeded) .ready else .failed, .cohort = cohort });
        if (!succeeded or agreement.choice != .replay or agreement.cohort != cohort) return error.StateRebindFailed;
    }
};

pub fn normalized(key: anytype) @TypeOf(key) {
    var candidate = key;
    candidate.seq[0][4] = 0;
    candidate.keep = key.keep_nonstate;
    return candidate;
}

pub fn keepHash(keep: anytype) u64 {
    var hash = std.hash.Wyhash.init(0x73746174656b6565);
    for (keep.layers) |layer| {
        inline for (@typeInfo(@TypeOf(layer)).@"struct".field_names) |field| {
            if (comptime !std.mem.eql(u8, field, "state_out")) std.hash.autoHash(&hash, @field(layer, field));
        }
    }
    std.hash.autoHash(&hash, keep.mixer);
    return hash.final();
}
