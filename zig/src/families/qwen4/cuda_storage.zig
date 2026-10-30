//! Weight placement over one CUDA driver and the same stream that owns this engine's inference commands.
pub fn Backend(comptime cuda: type) type {
    return struct {
        const Self = @This();
        pub const Buffer = cuda.DeviceBuffer;
        driver: *const cuda.Driver,
        stream: cuda.Stream,
        context: ?*const cuda.Context = null,
        /// Every entry can run on a different lane worker; copy-source custody uses this same current context.
        pub fn enter(self: *Self) !void {
            const context = self.context orelse return error.Qwen4ContextMissing;
            if (context.d != self.driver or self.stream.d != self.driver) return error.Qwen4DeviceContext;
            try context.makeCurrent();
        }
        pub fn allocationBytes(n: usize) !u64 {
            return n;
        }
        pub fn allocate(self: *Self, n: usize) !Buffer {
            try self.enter();
            if (self.stream.d != self.driver) return error.Qwen4DeviceContext;
            return Buffer.alloc(self.driver, n);
        }
        pub fn zero(self: *Self, b: Buffer, offset: usize, n: usize) !void {
            try self.enter();
            if (self.stream.d != self.driver or b.d != self.driver) return error.Qwen4DeviceContext;
            if (offset > b.len or n > b.len - offset) return error.Qwen4StateRange;
            if (n == 0) return;
            try self.driver.check(self.driver.api.cuMemsetD8Async(try b.at(offset), 0, n, self.stream.handle), "qwen4 state zero");
        }
        pub fn copy(self: *Self, to: Buffer, to_offset: usize, from: Buffer, from_offset: usize, n: usize) !void {
            try self.enter();
            if (self.stream.d != self.driver or to.d != self.driver or from.d != self.driver) return error.Qwen4DeviceContext;
            if (to_offset > to.len or n > to.len - to_offset or from_offset > from.len or n > from.len - from_offset) return error.Qwen4StateRange;
            if (n == 0) return;
            try self.driver.check(self.driver.api.cuMemcpyDtoDAsync_v2(try to.at(to_offset), try from.at(from_offset), n, self.stream.handle), "qwen4 state copy");
        }
        pub fn upload(self: *Self, bytes: []const u8) !Buffer {
            try self.enter();
            if (self.stream.d != self.driver) return error.Qwen4DeviceContext;
            return Buffer.fromHost(self.driver, bytes);
        }
        /// The shared copy-source owner retains a pinned copy until this inference stream finishes reading it.
        pub fn write(self: *Self, to: Buffer, offset: usize, bytes: []const u8) !void {
            try self.enter();
            if (self.stream.d != self.driver or to.d != self.driver) return error.Qwen4DeviceContext;
            try to.uploadAsync(offset, bytes, self.stream.handle);
        }
        pub fn release(_: *Self, buffer: Buffer) void {
            // The owner releases only after its entry/drain proved this thread's context and completion.
            var owned = buffer;
            owned.free();
        }
        pub fn drain(self: *Self) !void {
            try self.enter();
            if (self.stream.d != self.driver) return error.Qwen4DeviceContext;
            try self.stream.synchronize();
        }
    };
}
