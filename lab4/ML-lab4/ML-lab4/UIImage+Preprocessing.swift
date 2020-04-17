//
//  UIImage+Preprocessing.swift
//  ML-lab4
//
//  Created by Artiom Bastun on 16.04.2020.
//  Copyright © 2020 Artiom Bastun. All rights reserved.
//

import UIKit
import ImageIO
import Accelerate

extension UIImage {
    func resizedImage(for size: CGSize) -> UIImage? {
//        // Decode the source image
//        guard let imageSource = CGImageSourceCreateWithURL(url as NSURL, nil),
//            let image = CGImageSourceCreateImageAtIndex(imageSource, 0, nil),
//            let properties = CGImageSourceCopyPropertiesAtIndex(imageSource, 0, nil) as? [CFString: Any],
//            let imageWidth = properties[kCGImagePropertyPixelWidth] as? vImagePixelCount,
//            let imageHeight = properties[kCGImagePropertyPixelHeight] as? vImagePixelCount
//            else {
//                return nil
//        }

        guard
            let cgImage = self.cgImage,
            var format = vImage_CGImageFormat(cgImage: cgImage) else {
                return nil
        }
//        // Define the image format
//        var format = vImage_CGImageFormat(bitsPerComponent: 8,
//                                          bitsPerPixel: 32,
//                                          colorSpace: nil,
//                                          bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.first.rawValue),
//                                          version: 0,
//                                          decode: nil,
//                                          renderingIntent: .defaultIntent)

        var error: vImage_Error

        // Create and initialize the source buffer
        var sourceBuffer = vImage_Buffer()
        defer { sourceBuffer.data.deallocate() }
        error = vImageBuffer_InitWithCGImage(&sourceBuffer,
                                             &format,
                                             nil,
                                             cgImage,
                                             vImage_Flags(kvImageNoFlags))
        guard error == kvImageNoError else { return nil }

        // Create and initialize the destination buffer
        var destinationBuffer = vImage_Buffer()
        error = vImageBuffer_Init(&destinationBuffer,
                                  vImagePixelCount(size.height),
                                  vImagePixelCount(size.width),
                                  format.bitsPerPixel,
                                  vImage_Flags(kvImageNoFlags))
        guard error == kvImageNoError else { return nil }

        // Scale the image
        error = vImageScale_ARGB8888(&sourceBuffer,
                                     &destinationBuffer,
                                     nil,
                                     vImage_Flags(kvImageEdgeExtend))
        guard error == kvImageNoError else { return nil }

//        var channelBuffer = vImage_Buffer()
//        error = vImageBuffer_Init(&channelBuffer,
//                                  vImagePixelCount(size.height),
//                                  vImagePixelCount(size.width),
//                                  format.bitsPerPixel,
//                                  vImage_Flags(kvImageNoFlags))
//        guard error == kvImageNoError else { return nil }
//
//        error = vImageExtractChannel_ARGB8888(&sourceBuffer, &channelBuffer, 1, vImage_Flags(kvImageNoFlags))
//        guard error == kvImageNoError else { return nil }
//
//        let capacity = Int(size.height) * Int(size.width) * 4
//        let channelDataPtr = destinationBuffer.data.bindMemory(to: UInt8.self, capacity: capacity)
//        let channelRawData = UnsafeBufferPointer(start: channelDataPtr, count: capacity)
//        let channelData = Array(channelRawData)
//        print(channelData[0..<10])

        // Create a CGImage from the destination buffer
        guard let resizedImage =
            vImageCreateCGImageFromBuffer(&destinationBuffer,
                                          &format,
                                          nil,
                                          nil,
                                          vImage_Flags(kvImageNoAllocate),
                                          &error)?.takeRetainedValue(),
            error == kvImageNoError
            else {
                return nil
        }

        return UIImage(cgImage: resizedImage)
    }

    func buffer() -> CVPixelBuffer? {
        let image = self
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue, kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        var pixelBuffer : CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(image.size.width), Int(image.size.height), kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
        guard (status == kCVReturnSuccess) else {
            return nil
        }

        CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)

        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: pixelData, width: Int(image.size.width), height: Int(image.size.height), bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!), space: rgbColorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)

        context?.translateBy(x: 0, y: image.size.height)
        context?.scaleBy(x: 1.0, y: -1.0)

        UIGraphicsPushContext(context!)
        image.draw(in: CGRect(x: 0, y: 0, width: image.size.width, height: image.size.height))
        UIGraphicsPopContext()
        CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))

        return pixelBuffer
    }
}

extension UIImage {
    func cropping(to size: CGSize) -> UIImage? {
        let x = (self.size.width - size.width) / 2
        let y = (self.size.height - size.height) / 2

        let cropRect = CGRect(x: x, y: y, width: size.width, height: size.height)
        guard let cgImage = self.cgImage,
            let croppedCGImage = cgImage.cropping(to: cropRect) else { return nil }

        return UIImage(cgImage: croppedCGImage, scale: 0, orientation: imageOrientation)
    }

    func resizing(to size: CGSize) -> UIImage? {
//        let format = UIGraphicsImageRendererFormat()
//        format.scale = 1
//        format.preferredRange = .standard
//        let renderer = UIGraphicsImageRenderer(size: size, format: format)
//        let image = renderer.image { context in
//            self.draw(in: CGRect(x: 0, y: 0, width: size.width, height: size.height))
//        }
//        return image
        guard let image = self.cgImage, let colorSpace = image.colorSpace else { return nil }
        guard let context = CGContext(
            data: nil,
            width: Int(size.width),
            height: Int(size.height),
            bitsPerComponent:
            image.bitsPerComponent,
            bytesPerRow: image.bytesPerRow,
            space: colorSpace,
            bitmapInfo: image.alphaInfo.rawValue) else { return nil }

        // draw image to context (resizing it)
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: Int(size.width), height: Int(size.height)))

        // extract resulting image from context
        return context.makeImage().map { UIImage(cgImage: $0) }
    }

    func pixelData(size: CGSize = CGSize(width: 54, height: 54)) -> [UInt8]? {
        guard let cgImage = self.cgImage else { return nil }

        let size = self.size
        let dataSize = Int(size.width * size.height) * 4
        var pixelData = [UInt8](repeating: 0, count: dataSize)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &pixelData,
            width: Int(size.width),
            height: Int(size.height),
            bitsPerComponent: cgImage.bitsPerComponent,
            bytesPerRow: cgImage.bytesPerRow,
            space: colorSpace,
            bitmapInfo: cgImage.alphaInfo.rawValue) else { return nil }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: size.width, height: size.height))

        return pixelData
    }
}
