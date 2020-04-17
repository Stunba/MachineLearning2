//
//  ModelExtensions.swift
//  ML-lab4
//
//  Created by Artiom Bastun on 16.04.2020.
//  Copyright © 2020 Artiom Bastun. All rights reserved.
//

import UIKit
import CoreML

struct RGBA {
    let red: UInt8
    let green: UInt8
    let blue: UInt8
    let alpha: UInt8
}
// RGBA, RGBA, ... -> [R], [G], [B]
extension Array where Element == UInt8 {
    func channelBasedOrder() -> [UInt8] {
        let input = self
        assert(input.count % 4 == 0)
        let pixels: [RGBA] = (0..<input.count / 4).map {
            let offset = $0 * 4
            return RGBA(red: input[offset], green: input[offset + 1], blue: input[offset + 2], alpha: input[offset + 3])
        }
        return pixels.map { $0.red } + pixels.map { $0.green } + pixels.map { $0.blue }
    }

    func normalized() -> [Float32] {
        map { (Float32($0) / 255 - 0.5) / 0.5 }
    }
}

extension SVHNModel {
    struct Prediction: Codable {
        let length: Int
        let digits: [Int]
    }

    func predict(image: UIImage) -> Prediction? {
        guard let input = image.resizedImage(for: CGSize(width: 54, height: 54))?.pixelData()?.channelBasedOrder().normalized() else {
            return nil
        }
        return predict(input: input)
    }

    func predict(input: [Float32]) -> Prediction? {
        do {
            let input = SVHNModelInput(input_1: try MLMultiArray(input, shape: [1, 3, 54, 54]))
            let result = try prediction(input: input)
            if let length = result._62.array(of: Float32.self).maxIndex(),
                let digit1 = result._63.array(of: Float32.self).maxIndex(),
                let digit2 = result._64.array(of: Float32.self).maxIndex(),
                let digit3 = result._65.array(of: Float32.self).maxIndex(),
                let digit4 = result._66.array(of: Float32.self).maxIndex(),
                let digit5 = result._67.array(of: Float32.self).maxIndex() {

                let digits = [digit1, digit2, digit3, digit4, digit5]
                print("Prediction: Length = \(length)")
                print("Prediction: Result = \(digits[0..<length])")
                return Prediction(length: length, digits: digits)
            }
        } catch {
            print(error)
        }
        return nil
    }
}
