//
//  ResultItemCell.swift
//  ML-lab4
//
//  Created by Artiom Bastun on 16.04.2020.
//  Copyright © 2020 Artiom Bastun. All rights reserved.
//

import UIKit

final class ResultItemCell: UICollectionViewCell {
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var predictionLabel: UILabel!

    override func awakeFromNib() {
        super.awakeFromNib()
        // Initialization code
    }

    var item: PresentableResultItem? {
        didSet {
            guard let item = item else { return }
            imageView.image = item.image
            predictionLabel.text = "\(item.result)"
        }
    }
}
