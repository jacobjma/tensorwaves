import numpy as np


def data_generator(images, labels, batch_size=32, augmentations=None):
    if augmentations is None:
        augmentations = []

    num_iter = len(images) // batch_size
    assert images.shape[1] == images.shape[2]

    while True:
        for i in range(num_iter):
            if i == 0:
                indices = np.arange(len(images))
                np.random.shuffle(indices)

            batch_indices = indices[i * batch_size:(i + 1) * batch_size]
            batch_images = images[batch_indices].copy()
            batch_labels = [labels[i].copy() for i in batch_indices]

            yield batch_images, batch_labels

            # for j in range(batch_size):
            #     for augmentation in augmentations:
            #         if augmentation.image_only:
            #             batch_images[j] = augmentation.apply(batch_images[j])
            #         else:
            #             batch_images[j], batch_atoms[j] = augmentation.apply(batch_images[j], batch_atoms[j])
            #
            # batch_images = np.stack(batch_images)
            #
            # size = batch_images.shape[1:-1]
            #
            # batch_labels = label_func(batch_atoms, size)
            #
            # yield batch_images, batch_labels
