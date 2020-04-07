//
//  CNN.swift
//  ML-lab
//
//  Created by Artiom Bastun on 07.04.2020.
//  Copyright © 2020 Artiom Bastun. All rights reserved.
//

import Foundation
import TensorFlow

struct SimpleConvolutionalModel: Layer {
    var conv1 = Conv2D<Float>(filterShape: (5, 5, 1, 2), padding: .same, activation: relu)
    var conv2 = Conv2D<Float>(filterShape: (5, 5, 2, 4), activation: relu)
    var flatten = Flatten<Float>()
    var fc = Dense<Float>(inputSize: 2304, outputSize: 10)

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let convolved = input.sequenced(through: conv1, conv2)
        return convolved.sequenced(through: flatten, fc)
    }
}

struct ConvolutionalWithPooling: Layer {
    var conv = Conv2D<Float>(filterShape: (5, 5, 1, 4), padding: .same, activation: relu)
    var pool = AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    var flatten = Flatten<Float>()
    var fc = Dense<Float>(inputSize: 784, outputSize: 10)

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let convolved = input.sequenced(through: conv, pool)
        return convolved.sequenced(through: flatten, fc)
    }
}

extension NNClassifier where Optimizer == SGD<SimpleConvolutionalModel> {
    static func convolutional() -> NNClassifier<SimpleConvolutionalModel, SGD<SimpleConvolutionalModel>> {
        NNClassifier<SimpleConvolutionalModel, SGD<SimpleConvolutionalModel>>(
            modelCreator: {
                SimpleConvolutionalModel()
        },
            optimazerCreator: {
                SGD<SimpleConvolutionalModel>(for: $0, learningRate: 0.1)
        },
            epochCount: 20
        )
    }
}

extension NNClassifier where Optimizer == SGD<ConvolutionalWithPooling> {
    static func convolutionalWithPooling() -> NNClassifier<ConvolutionalWithPooling, SGD<ConvolutionalWithPooling>> {
        NNClassifier<ConvolutionalWithPooling, SGD<ConvolutionalWithPooling>>(
            modelCreator: {
                ConvolutionalWithPooling()
        },
            optimazerCreator: {
                SGD<ConvolutionalWithPooling>(for: $0, learningRate: 0.1)
        },
            epochCount: 20
        )
    }
}
