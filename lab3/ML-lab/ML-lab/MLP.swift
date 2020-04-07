//
//  MLP.swift
//  ML-lab
//
//  Created by Artiom Bastun on 01.04.2020.
//  Copyright © 2020 Artiom Bastun. All rights reserved.
//

import Foundation
import TensorFlow

struct DropoutModel: Layer {
    var dense1: Dense<Float>
    var dense2: Dense<Float>
    var dropout1 = Dropout<Float>(probability: 0.5)
    var dense3: Dense<Float>
    var dropout2 = Dropout<Float>(probability: 0.5)
    var dense4: Dense<Float>

    init(numberOfFeatures: Int,
         numberOfLabels: Int,
         numberOfUnits: Int) {

        dense1 = Dense(inputSize: numberOfFeatures, outputSize: numberOfUnits, activation: relu)
        dense2 = Dense(inputSize: numberOfUnits, outputSize: numberOfUnits, activation: relu)
        dense3 = Dense(inputSize: numberOfUnits, outputSize: numberOfUnits, activation: relu)
        dense4 = Dense(inputSize: numberOfUnits, outputSize: numberOfLabels, activation: identity)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        input.sequenced(through: dense1, dense2, dropout1, dense3, dropout2, dense4)
    }
}

/// MLP is a multi-layer perceptron and is used as a component of the DLRM model
public struct MLP: Layer {
    public var blocks: [Dense<Float>] = []

    /// Randomly initializes a new multilayer perceptron from the given hyperparameters.
    ///
    /// - Parameter dims: Dims represents the size of the input, hidden layers, and output of the
    ///   multi-layer perceptron.
    /// - Parameter sigmoidLastLayer: if `true`, use a `sigmoid` activation function for the last layer,
    ///   `relu` otherwise.
    init(dims: [Int], identityLastLayer: Bool = false) {
        for i in 0..<(dims.count-1) {
            if identityLastLayer && i == dims.count - 2 {
                blocks.append(Dense(inputSize: dims[i], outputSize: dims[i+1], activation: identity))
            } else {
                blocks.append(Dense(inputSize: dims[i], outputSize: dims[i+1], activation: relu))
            }
        }
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let blocksReduced = blocks.differentiableReduce(input) { last, layer in
            layer(last)
        }
        return blocksReduced
    }
}

extension MLP {
    init(numberOfFeatures: Int,
         numberOfLabels: Int,
         numberOfLayers: Int,
         numberOfUnits: Int,
         identityLastLayer: Bool = true) {

        let insides = [Int](repeating: numberOfUnits, count: numberOfLayers)
        self.init(dims: [numberOfFeatures] + insides + [numberOfLabels] , identityLastLayer: identityLastLayer)
    }
}
