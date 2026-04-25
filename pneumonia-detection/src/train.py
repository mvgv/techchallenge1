from data_loader import get_data_generators
from model import build_model

def main():

    base_dir = "data/chest_xray"

    train_data, val_data, test_data = get_data_generators(base_dir)

    model = build_model()

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=5
    )

    model.save("models/model.h5")

if __name__ == "__main__":
    main()