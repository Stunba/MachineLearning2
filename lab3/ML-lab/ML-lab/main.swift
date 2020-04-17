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

var dataset = NotMNISTDataset.large()

var classifier = NNClassifier.leNet()
classifier.fit(dataset: dataset)
