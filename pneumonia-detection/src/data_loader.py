from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(base_dir):

    train_dir = f"{base_dir}/train"
    val_dir = f"{base_dir}/val"
    test_dir = f"{base_dir}/test"

    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224,224),
        batch_size=32,
        class_mode='binary'
    )

    val_data = test_datagen.flow_from_directory(
        val_dir,
        target_size=(224,224),
        batch_size=32,
        class_mode='binary'
    )

    test_data = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224,224),
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )

    return train_data, val_data, test_data