//
//  ResultItem.swift
//  ML-lab4
//
//  Created by Artiom Bastun on 16.04.2020.
//  Copyright © 2020 Artiom Bastun. All rights reserved.
//

import UIKit

struct StoredResultItem: Codable {
    let imageURL: URL
    let prediction: SVHNModel.Prediction
}

struct StoredPredicitonResults: Codable {
    let results: [StoredResultItem]
}

extension SVHNModel.Prediction {
    var value: Int {
        var multiplier = 1
        return digits[0..<length].reversed().reduce(0, { prev, item in
            let partialResult = prev + item * multiplier
            multiplier *= 10
            return partialResult
        })
    }
}

struct PresentableResultItem: Hashable {
    let image: UIImage
    let result: Int
}

extension PresentableResultItem {
    init?(_ stored: StoredResultItem) {
        guard let image = UIImage(contentsOfFile: stored.imageURL.path) else { return nil }
        self.init(image: image, result: stored.prediction.value)
    }
}
