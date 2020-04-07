//
//  Logger.swift
//  ML-lab
//
//  Created by Artiom Bastun on 07.04.2020.
//  Copyright © 2020 Artiom Bastun. All rights reserved.
//

import Foundation

protocol Logger {
    mutating func log(_ message: @autoclosure () -> String)
}

struct SimpleLogger: Logger {
    var log: [String] = []

    mutating func log(_ message: @autoclosure () -> String) {
        log.append(message())
    }
}

struct PrintLogger: Logger {
    func log(_ message: @autoclosure () -> String) {
        print(message())
    }
}
