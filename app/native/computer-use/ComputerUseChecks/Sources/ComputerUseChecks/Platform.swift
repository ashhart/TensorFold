#if !canImport(CoreGraphics)
typealias CGKeyCode = UInt16

// Linux only needs the flag values; Foundation supplies CGPoint and CGSize on both platforms.
struct CGEventFlags: OptionSet, Sendable {
    let rawValue: UInt64
    init(rawValue: UInt64) { self.rawValue = rawValue }

    static let maskShift = Self(rawValue: 1 << 17)
    static let maskControl = Self(rawValue: 1 << 18)
    static let maskAlternate = Self(rawValue: 1 << 19)
    static let maskCommand = Self(rawValue: 1 << 20)
    static let maskSecondaryFn = Self(rawValue: 1 << 23)
}
#endif
