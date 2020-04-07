//
//  LeNet.swift
//  ML-lab
//
//  Created by Artiom Bastun on 07.04.2020.
//  Copyright © 2020 Artiom Bastun. All rights reserved.
//

import Foundation
import TensorFlow

// Original Paper:
// "Gradient-Based Learning Applied to Document Recognition"
// Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner
// http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
//
// Note: this implementation connects all the feature maps in the second convolutional layer.
// Additionally, ReLU is used instead of sigmoid activations.
struct LeNet: Layer {
    var conv1 = Conv2D<Float>(filterShape: (5, 5, 1, 6), padding: .same, activation: relu)
    var pool1 = AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    var conv2 = Conv2D<Float>(filterShape: (5, 5, 6, 16), activation: relu)
    var pool2 = AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    var flatten = Flatten<Float>()
    var fc1 = Dense<Float>(inputSize: 400, outputSize: 120, activation: relu)
    var fc2 = Dense<Float>(inputSize: 120, outputSize: 84, activation: relu)
    var fc3 = Dense<Float>(inputSize: 84, outputSize: 10)

    public init() {}

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let convolved = input.sequenced(through: conv1, pool1, conv2, pool2)
        return convolved.sequenced(through: flatten, fc1, fc2, fc3)
    }
}

extension NNClassifier where Optimizer == SGD<LeNet> {
    static func leNet() -> NNClassifier<LeNet, SGD<LeNet>> {
        NNClassifier<LeNet, SGD<LeNet>>(
            modelCreator: {
                LeNet()
        },
            optimazerCreator: {
                SGD<LeNet>(for: $0, learningRate: 0.1)
        },
            epochCount: 12
        )
    }
}
