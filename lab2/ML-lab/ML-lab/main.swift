//
//  main.swift
//  ML-lab
//
//  Created by Artiom Bastun on 01.04.2020.
//  Copyright © 2020 Artiom Bastun. All rights reserved.
//

import Foundation
import TensorFlow

#if canImport(PythonKit)
    import PythonKit
#else
    import Python
#endif

let locale = Python.import("locale")
locale.setlocale(locale.LC_ALL, "")

//let plt = Python.import("matplotlib.pyplot")
//let np = Python.import("numpy")

//do {
//    let np = try Python.attemptImport("numpy")
//} catch {
//    print(error)
//    print(Python.version)
//    print(Python.versionInfo)
//}

var dataset = NotMNISTDataset(filename: "notmnist_test")
//var dataset = NotMNISTDataset()

let queue = DispatchQueue(label: "workQueue", attributes: .concurrent)
let dispatchGroup = DispatchGroup()

queue.async(group: dispatchGroup) {
    var classifier = NNClassifier<MLP, SGD<MLP>>.defaultClassifier(numberOfFeatures: dataset.numberOfFeatures, numberOfLabels: 10)
    classifier.fit(dataset: dataset)
    print("Dense model using SGD")
    print(classifier.logger.log.joined(separator: "\n"))
}

queue.async(group: dispatchGroup) {
    var dropoutClf = NNClassifier.defaultDropout(numberOfFeatures: dataset.numberOfFeatures, numberOfLabels: 10)
    dropoutClf.fit(dataset: dataset)
    print("Dropout model with SGD")
    print(dropoutClf.logger.log.joined(separator: "\n"))
}

queue.async(group: dispatchGroup) {
    var adaptiveClf = NNClassifier.dropoutWithAda(numberOfFeatures: dataset.numberOfFeatures, numberOfLabels: 10)
    adaptiveClf.fit(dataset: dataset)
    print("Dropout Model with adaptive learning rate")
    print(adaptiveClf.logger.log.joined(separator: "\n"))
}

dispatchGroup.wait()
