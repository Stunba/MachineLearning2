//
//  NNModel.swift
//  ML-lab
//
//  Created by Artiom Bastun on 01.04.2020.
//  Copyright © 2020 Artiom Bastun. All rights reserved.
//

import Foundation
import TensorFlow

struct NNClassifier<Model: Layer, Optimizer: TensorFlow.Optimizer> where Optimizer.Model == Model, Model.Input == Tensor<Float>, Model.Output == Tensor<Float> {
    struct Statistics {
        var correctGuessCount: Int = 0
        var totalGuessCount: Int = 0
        var totalLoss: Float = 0
        var batches: Int = 0
    }

    let epochCount: Int
    var model: Model
    var optimizer: Optimizer

    var logger: Logger

    init(modelCreator: () -> Model,
         optimazerCreator: (Model) -> Optimizer,
         epochCount: Int = 50,
         logger: Logger = PrintLogger()) {

        self.epochCount = epochCount
        self.model = modelCreator()
        self.optimizer = optimazerCreator(self.model)
        self.logger = logger
    }

    mutating func fit(dataset: NotMNISTDataset) {
        for epoch in 1...epochCount {
            var trainStats = Statistics()
            var testStats = Statistics()

            Context.local.learningPhase = .training
            for batch in dataset.training.sequenced() {
                // Compute the gradient with respect to the model.
                let 𝛁model = gradient(at: model) { classifier -> Tensor<Float> in
                    let ŷ = classifier(batch.features)
//                    print("Logits: \(ŷ[0])\n")
                    let correctPredictions = ŷ.argmax(squeezingAxis: 1) .== batch.labels
                    trainStats.correctGuessCount += Int(
                        Tensor<Int32>(correctPredictions).sum().scalarized()
                    )
                    trainStats.totalGuessCount += batch.features.shape[0]
                    let loss = softmaxCrossEntropy(logits: ŷ, labels: batch.labels)
                    trainStats.totalLoss += loss.scalarized()
                    trainStats.batches += 1
                    return loss
                }
                // Update the model's differentiable variables along the gradient vector.
                optimizer.update(&model, along: 𝛁model)
            }

            Context.local.learningPhase = .inference
            for batch in dataset.test.sequenced() {
                // Compute loss on test set
                let ŷ = model(batch.features)
                let correctPredictions = ŷ.argmax(squeezingAxis: 1) .== batch.labels
                testStats.correctGuessCount += Int(Tensor<Int32>(correctPredictions).sum().scalarized())
                testStats.totalGuessCount += batch.features.shape[0]
                let loss = softmaxCrossEntropy(logits: ŷ, labels: batch.labels)
                testStats.totalLoss += loss.scalarized()
                testStats.batches += 1
            }

            let trainAccuracy = Float(trainStats.correctGuessCount) / Float(trainStats.totalGuessCount)
            let testAccuracy = Float(testStats.correctGuessCount) / Float(testStats.totalGuessCount)
            logger.log(
                """
                [Epoch \(epoch)] \
                Training Loss: \(trainStats.totalLoss / Float(trainStats.batches)), \
                Training Accuracy: \(trainStats.correctGuessCount)/\(trainStats.totalGuessCount) \
                (\(trainAccuracy)), \
                Test Loss: \(testStats.totalLoss / Float(testStats.batches)), \
                Test Accuracy: \(testStats.correctGuessCount)/\(testStats.totalGuessCount) \
                (\(testAccuracy))
                """)
        }
    }

    func predict(features: Tensor<Float>) -> Tensor<Int32> {
        model(features).argmax(squeezingAxis: 1)
    }

    func accuracy(features: Tensor<Float>, truths: Tensor<Int32>) -> Float {
        Tensor<Float>(predict(features: features) .== truths).mean().scalarized()
    }
}

extension NNClassifier {
    static func defaultClassifier(
        numberOfFeatures: Int,
        numberOfLabels: Int,
        numberOfLayers: Int = 4,
        numberOfUnits: Int = 128) -> NNClassifier<MLP, SGD<MLP>> {

        NNClassifier<MLP, SGD<MLP>>(modelCreator: { () -> MLP in
            MLP(numberOfFeatures: numberOfFeatures,
                numberOfLabels: numberOfLabels,
                numberOfLayers: numberOfLayers,
                numberOfUnits: numberOfUnits,
                identityLastLayer: true
            )
        }, optimazerCreator:  { (model: MLP) -> SGD<MLP> in
            SGD<MLP>(for: model, learningRate: 0.01)
        })
    }
}

extension NNClassifier where Optimizer == SGD<DropoutModel> {
    static func defaultDropout(numberOfFeatures: Int,
                               numberOfLabels: Int,
                               numberOfUnits: Int = 128) -> NNClassifier<DropoutModel, SGD<DropoutModel>> {

        NNClassifier<DropoutModel, SGD<DropoutModel>>(
            modelCreator: {
                DropoutModel(numberOfFeatures: numberOfFeatures, numberOfLabels: numberOfLabels, numberOfUnits: numberOfUnits)
        }, optimazerCreator: {
            SGD<DropoutModel>(for: $0, learningRate: 0.01)
        })
    }
}

extension NNClassifier where Optimizer == AdaDelta<DropoutModel> {
    static func dropoutWithAda(numberOfFeatures: Int,
                               numberOfLabels: Int,
                               numberOfUnits: Int = 128) -> NNClassifier<DropoutModel, AdaDelta<DropoutModel>> {

        NNClassifier<DropoutModel, AdaDelta<DropoutModel>>(
            modelCreator: {
                DropoutModel(numberOfFeatures: numberOfFeatures, numberOfLabels: numberOfLabels, numberOfUnits: numberOfUnits)
        }, optimazerCreator: {
            AdaDelta<DropoutModel>(for: $0)
        }, epochCount: 20)
    }
}
