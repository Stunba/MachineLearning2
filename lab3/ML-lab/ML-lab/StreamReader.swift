//
//  StreamReader.swift
//  ML-lab
//
//  Created by Artiom Bastun on 03.04.2020.
//  Copyright © 2020 Artiom Bastun. All rights reserved.
//

import Foundation

final class StreamReader {
    let encoding: String.Encoding
    let chunkSize: Int
    let fileHandle: FileHandle
    var buffer: Data
    let delimPattern : Data
    var isAtEOF: Bool = false

    init?(url: URL, delimeter: String = "\n", encoding: String.Encoding = .utf8, chunkSize: Int = 4096)
    {
        let handle: FileHandle?
        do {
            handle = try FileHandle(forReadingFrom: url)
        } catch {
            print(error)
            handle = nil
        }

        guard let fileHandle = handle else { return nil }
        self.fileHandle = fileHandle
        self.chunkSize = chunkSize
        self.encoding = encoding
        buffer = Data(capacity: chunkSize)
        delimPattern = delimeter.data(using: .utf8)!
    }

    deinit {
        fileHandle.closeFile()
    }

    func rewind() {
        fileHandle.seek(toFileOffset: 0)
        buffer.removeAll(keepingCapacity: true)
        isAtEOF = false
    }

    func nextLine() -> String? {
        if isAtEOF { return nil }

        repeat {
            if let range = buffer.range(of: delimPattern, options: [], in: buffer.startIndex..<buffer.endIndex) {
                let subData = buffer.subdata(in: buffer.startIndex..<range.lowerBound)
                let line = String(data: subData, encoding: encoding)
                buffer.replaceSubrange(buffer.startIndex..<range.upperBound, with: [])
                return line
            } else {
                let tempData = fileHandle.readData(ofLength: chunkSize)
                if tempData.count == 0 {
                    isAtEOF = true
                    return (buffer.count > 0) ? String(data: buffer, encoding: encoding) : nil
                }
                buffer.append(tempData)
            }
        } while true
    }
}
