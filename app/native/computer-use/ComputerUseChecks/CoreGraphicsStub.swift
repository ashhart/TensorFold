// This file is empty on a Mac, because canImport(CoreGraphics) is true and the real types are used.
// On Linux that import does not exist, so the block below defines the same names the parser already uses.
#if !canImport(CoreGraphics)
typealias CGFloat = Double
typealias CGKeyCode = UInt16

struct CGPoint: Equatable {
    var x: CGFloat
    var y: CGFloat
}

struct CGSize: Equatable {
    var width: CGFloat
    var height: CGFloat
}

// Same bit positions as Apple's CGEventFlags. The tests compare flags, not the raw numbers.
struct CGEventFlags: OptionSet {
    var rawValue: UInt64
    static let maskShift = CGEventFlags(rawValue: 1 << 17)
    static let maskControl = CGEventFlags(rawValue: 1 << 18)
    static let maskAlternate = CGEventFlags(rawValue: 1 << 19)
    static let maskCommand = CGEventFlags(rawValue: 1 << 20)
    static let maskSecondaryFn = CGEventFlags(rawValue: 1 << 23)
}
#endif
