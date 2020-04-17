//
//  AppDelegate.swift
//  ML-lab4
//
//  Created by Artiom Bastun on 14.04.2020.
//  Copyright © 2020 Artiom Bastun. All rights reserved.
//

import UIKit
import CoreML

@UIApplicationMain
class AppDelegate: UIResponder, UIApplicationDelegate {
    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        // Override point for customization after application launch.
//        do {
//            let model = SmallModel()
//            let input = SmallModelInput(input_1: try MLMultiArray([Float32](repeating: 1000, count: 768)))
//            let result = try model.prediction(input: input)
//            print(result)
//            print(result.featureNames)
//            print(result._12)
//        } catch {
//            print(error)
//        }
//        if let url = Bundle.main.url(forResource: "flat_image", withExtension: "txt") {
//            do {
//                let data = try Data(contentsOf: url)
//                guard let str = String(data: data, encoding: .utf8) else { return true }
//                let input = str.split(separator: "\n").compactMap { Float32($0) }
//                predict(input: input)
//            } catch {
//                print(error)
//            }
//        }
        //            let frmt = NumberFormatter()
        //            frmt.maximumFractionDigits = 6
        //            frmt.minimumFractionDigits = 6

//        if let url = Bundle.main.url(forResource: "test-75", withExtension: "png"),
//            let image = resizedImage(at: url, for: CGSize(width: 54, height: 54)),
//        let image = UIImage(named: "test-75")!.resizedImage(for: CGSize(width: 54, height: 54))!
//        let buffer = image.buffer()!
//        let format = vImageCVImageFormat.make(buffer: buffer)!
//        print(format.channelDescription(bufferType: .rgbRed))

        return true
    }

    // MARK: UISceneSession Lifecycle

    func application(_ application: UIApplication, configurationForConnecting connectingSceneSession: UISceneSession, options: UIScene.ConnectionOptions) -> UISceneConfiguration {
        // Called when a new scene session is being created.
        // Use this method to select a configuration to create the new scene with.
        return UISceneConfiguration(name: "Default Configuration", sessionRole: connectingSceneSession.role)
    }

    func application(_ application: UIApplication, didDiscardSceneSessions sceneSessions: Set<UISceneSession>) {
        // Called when the user discards a scene session.
        // If any sessions were discarded while the application was not running, this will be called shortly after application:didFinishLaunchingWithOptions.
        // Use this method to release any resources that were specific to the discarded scenes, as they will not return.
    }


}

