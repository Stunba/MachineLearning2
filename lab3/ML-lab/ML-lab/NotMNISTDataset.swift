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
    static let shapeWithChannels: TensorShape = [28, 28, 1]

    static func small(
        hasHeader: Bool = true,
        normalize: Bool = true,
        featureShape: TensorShape? = shapeWithChannels) -> NotMNISTDataset {

        var dataset = Self.read(
            filePath: "/Users/stunba/Projects/MachineLearning2/lab1/notmnist_test.csv",
            hasHeader: hasHeader,
            normalize: normalize,
            featureShape: featureShape
        )
        print(dataset.last!)
        dataset.shuffle()
        let testSize = Int(Float(dataset.count) * 0.1)
        let train = dataset.dropLast(testSize)
        let test = dataset[dataset.endIndex - testSize..<dataset.endIndex]
        return NotMNISTDataset(train: Dataset(train), test: Dataset(test), numberOfFeatures: train.first?.features.scalarCount ?? 0)
    }

    static func large(
        hasHeader: Bool = true,
        normalize: Bool = true,
        featureShape: TensorShape? = shapeWithChannels) -> NotMNISTDataset {

        NotMNISTDataset(
            trainFilePath: "/Users/stunba/Downloads/notmnist_train.csv",
            testFilePath: "/Users/stunba/Projects/MachineLearning2/lab1/notmnist_test.csv",
            hasHeader: hasHeader,
            normalize: normalize,
            featureShape: featureShape
        )
    }

    init(trainFilePath: String,
         testFilePath: String,
         hasHeader: Bool = true,
         normalize: Bool = true,
         featureShape: TensorShape? = nil) {

        let train = Self.read(filePath: trainFilePath, hasHeader: hasHeader, normalize: normalize, featureShape: featureShape)
        let test = Self.read(filePath: testFilePath, hasHeader: hasHeader, normalize: normalize, featureShape: featureShape)
        self.init(train: train, test: test, numberOfFeatures: train.first?.features.scalarCount ?? 0)
    }

    init(train: Dataset, test: Dataset, numberOfFeatures: Int, numberOfLabels: Int = 10) {
        self.numberOfFeatures = numberOfFeatures
        self.training = Batcher(on: train, batchSize: 1024, shuffle: true)
        self.test = Batcher(on: test, batchSize: 1024)
    }

    private static func read(
        filePath: String,
        hasHeader: Bool = true,
        numberOfLabels: Int = 10,
        normalize: Bool = true,
        featureShape: TensorShape? = nil) -> Dataset {

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
            var featureTensor = Tensor<Float>(features) / 255.0
            if normalize {
                featureTensor = featureTensor * 2 - 1
            }
            if let featureShape = featureShape {
                featureTensor = featureTensor.reshaped(to: featureShape)
            }

            if let label = entry.last?.first?.asciiValue.map({ Int32($0 - Character("A").asciiValue!) }) {
                data.append(NotMnistBatch(features: featureTensor, labels: Tensor<Int32>(label)))
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
