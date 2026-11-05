#if canImport(CoreGraphics)
import CoreGraphics
#endif

// The parser refuses phone taps and typing before any Mac API runs.
func phoneLocalPoint(_ point: CGPoint, size: CGSize) -> Bool {
    point.x.isFinite && point.y.isFinite && size.width.isFinite && size.height.isFinite && size.width >= 1 && size.height >= 1 && point.x >= 0 && point.y >= 0 && point.x < size.width && point.y < size.height
}

func phoneTypeAllowed(_ text: String) -> Bool {
    text.unicodeScalars.allSatisfy { scalar in
        scalar == "\n" || scalar == "\t" || scalar.value >= 32 && scalar.value != 127 && !(0xE000...0xF8FF).contains(scalar.value)
    }
}
