import CoreGraphics
import Foundation
import ImageIO
import Vision

// This standalone mode accepts user-selected bytes on stdin only. It never captures a
// screen, requests Accessibility privileges, opens a path, or sends a network request.
let attachmentTextMaxBytes = 20 * 1024 * 1024
let attachmentTextMaxPages = 8
let attachmentTextMaxPixels = 24 * 1024 * 1024
let attachmentTextMaxCharacters = 199_800
let attachmentTextReviewWarning = "Review every line against the original, especially handwriting, names and numbers. OCR may omit or misread text even when its confidence score is high."

struct AttachmentOcrPage {
    let text: String
    let confidence: Double?
    let warnings: [String]
    let truncated: Bool
}

func attachmentOcrCapability() -> [String: Any] {
    return ["available": true, "method": "apple-vision", "maxBytes": attachmentTextMaxBytes, "maxPages": attachmentTextMaxPages]
}

func readAttachmentInput() throws -> Data {
    var data = Data()
    while let chunk = try FileHandle.standardInput.read(upToCount: min(64 * 1024, attachmentTextMaxBytes + 1 - data.count)), !chunk.isEmpty {
        data.append(chunk)
        if data.count > attachmentTextMaxBytes { throw HelperError("attachment_too_large", "Text extraction accepts files up to 20 MiB. Choose a smaller file or fewer PDF pages.") }
    }
    guard !data.isEmpty else { throw HelperError("attachment_empty", "The selected file is empty.") }
    return data
}

func recognizeAttachmentImage(_ image: CGImage, orientation: CGImagePropertyOrientation = .up) throws -> AttachmentOcrPage {
    guard image.width > 0, image.height > 0, image.width <= attachmentTextMaxPixels / image.height else {
        throw HelperError("attachment_pixels", "The image exceeds the 24-megapixel text-extraction limit.")
    }
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false
    request.automaticallyDetectsLanguage = true
    request.preferBackgroundProcessing = true
    if VNRecognizeTextRequest.supportedRevisions.contains(VNRecognizeTextRequestRevision3) { request.revision = VNRecognizeTextRequestRevision3 }
    do { try VNImageRequestHandler(cgImage: image, orientation: orientation, options: [:]).perform([request]) }
    catch { throw HelperError("attachment_ocr", "Apple Vision could not read this image. Choose a clearer image or type a transcript.") }
    let observations = request.results ?? []
    let candidates = observations.prefix(1000).compactMap { $0.topCandidates(1).first }
    let text = candidates.map { $0.string }.joined(separator: "\n")
    let confidence = candidates.map { Double($0.confidence) }.min()
    var warnings: [String] = []
    if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { warnings.append("No readable text was found. Nothing will be sent unless you type a transcript and accept it.") }
    if candidates.contains(where: { $0.confidence < 0.65 }) { warnings.append("Some recognized lines have low confidence. Check these against the original; handwriting can be incomplete or incorrect.") }
    if observations.count > 1000 { warnings.append("This page contains more text regions than the safety limit; some regions were omitted.") }
    let truncated = observations.count > 1000 || text.count > attachmentTextMaxCharacters
    return AttachmentOcrPage(text: String(text.prefix(attachmentTextMaxCharacters)), confidence: confidence, warnings: warnings, truncated: truncated)
}

func attachmentRaster(_ data: Data) throws -> (CGImage, CGImagePropertyOrientation, Int) {
    guard let source = CGImageSourceCreateWithData(data as CFData, [kCGImageSourceShouldCache: false] as CFDictionary), CGImageSourceGetCount(source) > 0,
          let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
          let width = (props[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue,
          let height = (props[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue,
          width > 0, height > 0, width <= attachmentTextMaxPixels / height else {
        throw HelperError("attachment_image", "Image dimensions could not be validated or exceed 24 megapixels. Choose a supported, smaller image.")
    }
    guard let image = CGImageSourceCreateImageAtIndex(source, 0, [kCGImageSourceShouldCacheImmediately: false] as CFDictionary) else {
        throw HelperError("attachment_image", "This image could not be decoded. Choose a PNG, JPEG, TIFF, HEIC or another supported image.")
    }
    let rawOrientation = (props[kCGImagePropertyOrientation] as? NSNumber)?.uint32Value ?? 1
    return (image, CGImagePropertyOrientation(rawValue: rawOrientation) ?? .up, CGImageSourceGetCount(source))
}

func attachmentPdfPageImage(_ page: CGPDFPage) throws -> CGImage {
    let box = page.getBoxRect(.mediaBox)
    guard box.width.isFinite, box.height.isFinite, box.width > 0, box.height > 0 else { throw HelperError("attachment_pdf_page", "This PDF page has invalid dimensions.") }
    // Scale to at most 2,000 pixels on the long edge before allocating its raster.
    let scale = min(2.0, 2000 / max(box.width, box.height))
    let width = max(1, Int(ceil(box.width * scale)))
    let height = max(1, Int(ceil(box.height * scale)))
    guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
        throw HelperError("attachment_pdf_page", "This PDF page could not be rendered within the memory limit.")
    }
    context.setFillColor(CGColor(gray: 1, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    context.concatenate(page.getDrawingTransform(.mediaBox, rect: CGRect(x: 0, y: 0, width: width, height: height), rotate: 0, preserveAspectRatio: true))
    context.drawPDFPage(page)
    guard let image = context.makeImage() else { throw HelperError("attachment_pdf_page", "The PDF page could not be decoded.") }
    return image
}

func extractAttachmentText(_ data: Data, mime: String) throws -> [String: Any] {
    guard data.count > 0, data.count <= attachmentTextMaxBytes else { throw HelperError("attachment_size", "Text extraction requires a nonempty file no larger than 20 MiB.") }
    var pages: [AttachmentOcrPage] = []
    var pageCount = 1
    var warnings = [attachmentTextReviewWarning]
    if mime == "application/pdf" {
        guard data.prefix(5) == Data("%PDF-".utf8), let provider = CGDataProvider(data: data as CFData), let pdf = CGPDFDocument(provider), !pdf.isEncrypted else {
            throw HelperError("attachment_pdf", "This PDF is unreadable, invalid or password-protected. Unlock or export it first.")
        }
        pageCount = pdf.numberOfPages
        guard pageCount > 0 else { throw HelperError("attachment_pdf", "This PDF has no pages.") }
        for number in 1...min(pageCount, attachmentTextMaxPages) {
            let result: AttachmentOcrPage = autoreleasepool {
                do {
                    guard let page = pdf.page(at: number) else { throw HelperError("attachment_pdf_page", "The PDF page could not be opened.") }
                    return try recognizeAttachmentImage(attachmentPdfPageImage(page))
                } catch {
                    return AttachmentOcrPage(text: "", confidence: nil, warnings: ["Page \(number) could not be read; review or select that page separately."], truncated: true)
                }
            }
            pages.append(result)
        }
        if pageCount > attachmentTextMaxPages { warnings.append("Only the first \(attachmentTextMaxPages) of \(pageCount) PDF pages were processed. Select remaining pages separately.") }
    } else {
        let (image, orientation, imageCount) = try attachmentRaster(data)
        pageCount = imageCount
        pages = [try recognizeAttachmentImage(image, orientation: orientation)]
        if imageCount > 1 { warnings.append("Only the first of \(imageCount) image frames or pages was processed. Select additional frames or pages separately.") }
    }
    let text = pages.enumerated().map { index, page in
        guard pageCount > 1 else { return page.text }
        return "[Page \(index + 1)]\n" + (page.text.isEmpty ? "[No readable text recognized on this page]" : page.text)
    }.joined(separator: "\n\n")
    for page in pages { warnings.append(contentsOf: page.warnings) }
    let truncated = pageCount > pages.count || pages.contains(where: { $0.truncated }) || text.count > attachmentTextMaxCharacters
    if text.count > attachmentTextMaxCharacters { warnings.append("The extracted transcript exceeded the text safety limit and was shortened.") }
    let hasText = pages.contains { !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
    return ["ok": true, "text": hasText ? String(text.prefix(attachmentTextMaxCharacters)) : "", "confidence": pages.compactMap { $0.confidence }.min() as Any? ?? NSNull(), "warnings": Array(warnings.prefix(20)), "pageCount": pageCount, "processedPages": pages.count, "truncated": truncated]
}

func runAttachmentTextMode() -> Int32 {
    do {
        let mime = argumentValue("--mime") ?? "image/png"
        guard mime == "application/pdf" || ["image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp", "image/bmp", "image/x-ms-bmp", "image/heic", "image/heif", "image/tiff"].contains(mime) else {
            throw HelperError("attachment_type", "Text extraction supports images and PDF files only.")
        }
        writeLine(try extractAttachmentText(readAttachmentInput(), mime: mime))
        return 0
    } catch let error as HelperError {
        writeLine(["ok": false, "error": error.message, "code": error.code])
        return 1
    } catch {
        writeLine(["ok": false, "error": "The selected bytes could not be read. Select the file again.", "code": "attachment_read"])
        return 1
    }
}
