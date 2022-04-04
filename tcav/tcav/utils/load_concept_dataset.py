import tensorflow as tf


def load_dataset(
        path_input_dir,
        image_size=(224, 224),
        rescale='mobilenet'):
    """
    """
    if rescale == 'mobilenet':
        rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)
    else:
        raise NotImplementedError(f"Rescale {rescale} not implemented.")

    dataset = tf.keras.utils.image_dataset_from_directory(
        path_input_dir,
        image_size=image_size,
        shuffle=False,
        batch_size=None).map(lambda x, y: (rescale(x), y))

    return dataset
