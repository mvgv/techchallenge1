import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from data_loader import get_data_generators

def main():

    model = load_model("models/model.h5")

    _, _, test_data = get_data_generators("data/chest_xray")

    preds = model.predict(test_data)
    preds = (preds > 0.5).astype(int)

    print(classification_report(test_data.classes, preds))

    cm = confusion_matrix(test_data.classes, preds)

    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.savefig("outputs/confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    main()