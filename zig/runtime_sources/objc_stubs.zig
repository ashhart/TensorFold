//! Link-only non-macOS stubs for the device-free source inventory; none is callable in a passing test.
export fn objc_getClass(name: [*:0]const u8) ?*anyopaque {
    _ = name;
    @panic("Objective-C is unavailable in the host-only Metal source inventory");
}

export fn sel_registerName(name: [*:0]const u8) ?*anyopaque {
    _ = name;
    @panic("Objective-C is unavailable in the host-only Metal source inventory");
}

export fn objc_msgSend(receiver: ?*anyopaque, selector: ?*anyopaque) ?*anyopaque {
    _ = receiver;
    _ = selector;
    @panic("Objective-C is unavailable in the host-only Metal source inventory");
}
