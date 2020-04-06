//
//  NotMNISTDataset.swift
//  ML-lab
//
//  Created by Artiom Bastun on 03.04.2020.
//  Copyright © 2020 Artiom Bastun. All rights reserved.
//

import Foundation
import TensorFlow

struct NotMnistBatch: KeyPathIterable, Collatable {
    var features: Tensor<Float>
    var labels: Tensor<Int32>
}

struct NotMNISTDataset {
    typealias Dataset = [NotMnistBatch]

    let numberOfFeatures: Int
    var training: Batcher<Dataset>
    var test: Batcher<Dataset>
}

extension NotMNISTDataset {
    init() {
        var dataset = Self.read(filePath: "/Users/stunba/Projects/MachineLearning2/lab1/notmnist_test.csv")
        print(dataset.last!)
        dataset.shuffle()
        let testSize = Int(Float(dataset.count) * 0.1)
        let train = dataset.dropLast(testSize)
        let test = dataset[dataset.endIndex - testSize..<dataset.endIndex]
        self.init(train: Dataset(train), test: Dataset(test), numberOfFeatures: train.first?.features.scalarCount ?? 0)
    }

    init(filename: String, hasHeader: Bool = true, limitLines: Int? = nil) {
//        let dataset = Self.read(filePath: "\(FileManager.default.currentDirectoryPath)/\(filename).csv")
//        let dataset = Self.read(filePath: "/Users/stunba/Downloads/notmnist_train.csv", limitLines: limitLines)
        let train = Self.read(filePath: "/Users/stunba/Downloads/notmnist_train.csv", limitLines: limitLines)
        let test = Self.read(filePath: "/Users/stunba/Projects/MachineLearning2/lab1/notmnist_test.csv", limitLines: limitLines)
        self.init(train: train, test: test, numberOfFeatures: train.first?.features.scalarCount ?? 0)
    }

    init(train: Dataset, test: Dataset, numberOfFeatures: Int, numberOfLabels: Int = 10) {
        self.numberOfFeatures = numberOfFeatures
        self.training = Batcher(on: train, batchSize: 1024, shuffle: true)
        self.test = Batcher(on: test, batchSize: 1024)
    }

    private static func read(filePath: String, hasHeader: Bool = true, limitLines: Int? = nil, numberOfLabels: Int = 10) -> Dataset {
        let fileURL = URL(fileURLWithPath: filePath)
        guard let reader = StreamReader(url: fileURL) else { return [] }

        if hasHeader {
            _ = reader.nextLine()
        }

        var linesRead = 0
        var data: Dataset = []
        while let line = reader.nextLine() {
            let entry = line.split(separator: ",").dropFirst().map { String($0) }
            linesRead += 1

            let features = entry.compactMap { Float($0) }
            if let label = entry.last?.first?.asciiValue.map({ Int32($0 - Character("A").asciiValue!) }) {

                data.append(NotMnistBatch(features: (Tensor<Float>(features) / 255.0) * 2 - 1, labels: Tensor<Int32>(label)))
            }

            if let limit = limitLines, linesRead > limit {
                break
            }
        }
        return data
    }

    private static func oneHotEncoded(index: Int, size: Int) -> Tensor<Int32> {
        var result = [Int32](repeating: 0, count: size)
        result[index] = 1
        return Tensor<Int32>(result)
    }
}
