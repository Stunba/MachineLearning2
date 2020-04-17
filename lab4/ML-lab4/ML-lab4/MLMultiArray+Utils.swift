//
//  MLMultiArray+Utils.swift
//  ML-lab4
//
//  Created by Artiom Bastun on 16.04.2020.
//  Copyright © 2020 Artiom Bastun. All rights reserved.
//

import Foundation
import CoreML

extension MLMultiArray {
    func array<T>(of type: T.Type) -> [T] {
        let arrayPtr =  self.dataPointer.bindMemory(to: T.self, capacity: self.count)
        let arrayBuffer = UnsafeBufferPointer(start: arrayPtr, count: self.count)
        return Array(arrayBuffer)
    }

    convenience init<C>(_ data: C, shape: [Int]) throws where C : Collection, C.Element == Float {
        try self.init(shape: shape as [NSNumber], dataType: .float32)
        data.enumerated().forEach { offset, element in
            self[offset] = NSNumber(value: element)
        }
    }
}

extension Array where Self.Element: Comparable {
    func maxIndex() -> Int? {
        let elem = self.enumerated().max(by: { $0.element < $1.element })
        return elem?.offset
    }
}
