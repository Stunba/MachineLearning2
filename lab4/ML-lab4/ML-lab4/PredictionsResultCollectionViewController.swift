//
//  PredictionsResultCollectionViewController.swift
//  ML-lab4
//
//  Created by Artiom Bastun on 16.04.2020.
//  Copyright © 2020 Artiom Bastun. All rights reserved.
//

import UIKit

private let reuseIdentifier = "ResultItemCell"

public extension URL {
    static var documentsDirectory: URL {
        let path = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true).first!
        return URL(fileURLWithPath: path)
    }
}

extension StoredPredicitonResults {
    var presentableItems: [PresentableResultItem] {
        results.compactMap { item in
            do {
                print(try FileManager.default.contentsOfDirectory(atPath: URL.documentsDirectory.path))
                let data = try Data(contentsOf: item.imageURL)
                guard let image = UIImage(data: data) else {
                    return nil
                }
                return PresentableResultItem(image: image, result: item.prediction.value)
            } catch {
                print(error)
                return nil
            }
        }
    }
}

final class PredictionResultItemsProvider {
    private let queue = DispatchQueue(label: "com.ml.lab4.data")
    private var items: StoredPredicitonResults?

    private var storeFileURL: URL {
        let url = URL.documentsDirectory.appendingPathComponent("data.json")
        if !FileManager.default.fileExists(atPath: url.path) {
            FileManager.default.createFile(atPath: url.path, contents: nil, attributes: [FileAttributeKey.protectionKey: FileProtectionType.none])
        }
        return url
    }

    func store(results: StoredPredicitonResults) {
        let jsonEncoder = JSONEncoder()
        do {
            let data = try jsonEncoder.encode(results)
            try data.write(to: storeFileURL)
        } catch {
            print(error)
        }
    }

    func load() -> StoredPredicitonResults {
        do {
            let data = try Data(contentsOf: storeFileURL)

            let jsonDecoder = JSONDecoder()
            let results = try jsonDecoder.decode(StoredPredicitonResults.self, from: data)
            items = results
            return results
        } catch {
            print(error)
            return StoredPredicitonResults(results: [])
        }
    }

    func loadItems(then completion: @escaping ([PresentableResultItem]) -> Void) {
        queue.async { [weak self] in
            guard let slf = self else { return completion([]) }

            let stored: StoredPredicitonResults
            if let items = slf.items {
                stored = items
            } else {
                stored = slf.load()
            }

            return completion(stored.presentableItems)
        }
    }

    func save(image: UIImage, prediction: SVHNModel.Prediction, then completion: @escaping ([PresentableResultItem]) -> Void) {
        queue.async { [weak self] in
            guard let imageData = image.pngData() else { return }

            let id = UUID().uuidString
            let url = URL.documentsDirectory.appendingPathComponent("\(id).png")
            do {
                try imageData.write(to: url, options: [.atomic, .noFileProtection])
                let item = StoredResultItem(id: id, prediction: prediction)
                let new = StoredPredicitonResults(results: (self?.items?.results ?? []) + [item])
                self?.store(results: new)
                self?.items = new
                completion(new.presentableItems)
            } catch {
                print(error)
                completion(self?.items?.presentableItems ?? [])
            }
        }
    }

    func delete(at index: Int, then completion: @escaping ([PresentableResultItem]) -> Void) {
        queue.async { [weak self] in
            guard let slf = self, var items = slf.items?.results else { return completion([]) }

            do {
                let item = items.remove(at: index)
                try FileManager.default.removeItem(at: item.imageURL)
                let new = StoredPredicitonResults(results: items)
                slf.store(results: new)
                slf.items = new
                completion(new.presentableItems)
            } catch {
                print(error)
                completion(slf.items?.presentableItems ?? [])
            }
        }
    }

    func loadTestItems(model: SVHNModel, then completion: ([PresentableResultItem]) -> Void) {
        let images = [UIImage(named: "test-75")?.resizedImage(for: CGSize(width: 54, height: 54)),
                      UIImage(named: "test-190")?.cropping(to: CGSize(width: 54, height: 54))].compactMap { $0 }

        let results = images.compactMap { $0.pixelData()?.channelBasedOrder().normalized() }
            .compactMap { model.predict(input: $0) }

        let items = zip(images, results).map { PresentableResultItem(image: $0.0, result: $0.1.value) }
        completion(items)
    }
}

final class PredictionsResultCollectionViewController: UICollectionViewController {

    let itemsProvider = PredictionResultItemsProvider()
    private let queue = DispatchQueue(label: "com.ml.lab4")

    private var model: SVHNModel?

    var items: [PresentableResultItem] = [] {
        didSet {
            var snapshot = NSDiffableDataSourceSnapshot<Int, PresentableResultItem>()
            snapshot.appendSections([0])
            snapshot.appendItems(items)
            dataSource?.apply(snapshot, animatingDifferences: true)
        }
    }
    var dataSource: UICollectionViewDiffableDataSource<Int, PresentableResultItem>?

    init() {
        super.init(collectionViewLayout: Self.createLayout())
        queue.async { [weak self] in
            self?.model = SVHNModel()
        }
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func viewDidLoad() {
        super.viewDidLoad()

        self.clearsSelectionOnViewWillAppear = false

        self.navigationItem.title = "Prediction Results"
        self.navigationItem.rightBarButtonItem = UIBarButtonItem(barButtonSystemItem: .add, target: self, action: #selector(addItem(_:)))

        collectionView.backgroundColor = .systemBackground
        self.collectionView!.register(UINib(nibName: reuseIdentifier, bundle: nil), forCellWithReuseIdentifier: reuseIdentifier)
        configureDataSource()
        loadData()
    }

    private func loadData() {
        itemsProvider.loadItems() { [weak self] items in
            DispatchQueue.main.async {
                self?.items = items
            }
        }
    }

    private static func createLayout() -> UICollectionViewLayout {
        let itemSize = NSCollectionLayoutSize(widthDimension: .fractionalWidth(0.48),
                                              heightDimension: .estimated(100))
        let item = NSCollectionLayoutItem(layoutSize: itemSize)

        let groupSize = NSCollectionLayoutSize(widthDimension: .fractionalWidth(1.0),
                                               heightDimension: .estimated(100))
        let group = NSCollectionLayoutGroup.horizontal(layoutSize: groupSize,
                                                       subitems: [item, item])
        group.interItemSpacing = .flexible(8)

        let section = NSCollectionLayoutSection(group: group)
        section.interGroupSpacing = 8
        section.contentInsets = NSDirectionalEdgeInsets(top: 20, leading: 20, bottom: 20, trailing: 20)

        let layout = UICollectionViewCompositionalLayout(section: section)
        return layout
    }

    private func configureDataSource() {
        dataSource = UICollectionViewDiffableDataSource<Int, PresentableResultItem>(collectionView: collectionView) {
            (collectionView: UICollectionView, indexPath: IndexPath, item: PresentableResultItem) -> UICollectionViewCell? in

            // Get a cell of the desired kind.
            guard let cell = collectionView.dequeueReusableCell(
                withReuseIdentifier: reuseIdentifier,
                for: indexPath) as? ResultItemCell else { fatalError("Cannot create new cell") }

            // Populate the cell with our item description.
            cell.item = item
            cell.deleteClosure = { [weak self] in
                self?.itemsProvider.delete(at: indexPath.item, then: { items in
                    DispatchQueue.main.async {
                        self?.items = items
                    }
                })
            }

            // Return the cell.
            return cell
        }
        //        collectionView.dataSource = dataSource
    }

    @objc
    private func addItem(_ sender: Any) {
        let selectImageViewController = UIImagePickerController()
        selectImageViewController.sourceType = .camera
        selectImageViewController.cameraCaptureMode = .photo
        selectImageViewController.allowsEditing = true
        selectImageViewController.delegate = self
        present(selectImageViewController, animated: true)
    }

    // MARK: UICollectionViewDelegate

    /*
     // Uncomment this method to specify if the specified item should be highlighted during tracking
     override func collectionView(_ collectionView: UICollectionView, shouldHighlightItemAt indexPath: IndexPath) -> Bool {
     return true
     }
     */

    /*
     // Uncomment this method to specify if the specified item should be selected
     override func collectionView(_ collectionView: UICollectionView, shouldSelectItemAt indexPath: IndexPath) -> Bool {
     return true
     }
     */


     // Uncomment these methods to specify if an action menu should be displayed for the specified item, and react to actions performed on the item
     override func collectionView(_ collectionView: UICollectionView, shouldShowMenuForItemAt indexPath: IndexPath) -> Bool {
        return true
     }

     override func collectionView(_ collectionView: UICollectionView, canPerformAction action: Selector, forItemAt indexPath: IndexPath, withSender sender: Any?) -> Bool {
        return action == #selector(UIResponderStandardEditActions.delete(_:))
        return true
     }

     override func collectionView(_ collectionView: UICollectionView, performAction action: Selector, forItemAt indexPath: IndexPath, withSender sender: Any?) {
        guard action == #selector(UIResponderStandardEditActions.delete(_:)) else { return }
     }
}

extension PredictionsResultCollectionViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        queue.async { [weak self] in
            guard let image = info[.editedImage] as? UIImage else { return }

            let result = self?.model?.predict(image: image)
            guard let res = result else { return }

            DispatchQueue.main.async {
                self?.items.append(PresentableResultItem(image: image, result: res.value))
            }

            self?.itemsProvider.save(image: image, prediction: res, then: {_ in })
        }
        dismiss(animated: true)
    }

    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        dismiss(animated: true)
    }
}
