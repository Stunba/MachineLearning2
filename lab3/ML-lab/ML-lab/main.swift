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

var dataset = NotMNISTDataset.large()
//var dataset = NotMNISTDataset()

var classifier = NNClassifier.leNet()
classifier.fit(dataset: dataset)

//let queue = DispatchQueue(label: "workQueue", attributes: .concurrent)
//let dispatchGroup = DispatchGroup()
//
//queue.async(group: dispatchGroup) {
//    var classifier = NNClassifier.convolutional()
//    classifier.fit(dataset: dataset)
//    print("Simple conv model using SGD")
//    print(classifier.logger.log.joined(separator: "\n"))
//}
//
//queue.async(group: dispatchGroup) {
//    var classifier = NNClassifier.convolutionalWithPooling()
//    classifier.fit(dataset: dataset)
//    print("Conv with pooling model with SGD")
//    print(classifier.logger.log.joined(separator: "\n"))
//}
//
//dispatchGroup.wait()
