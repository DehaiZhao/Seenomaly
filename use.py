import os
import argparse
import constants
import extract_features

import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

def printExit(out):
    print(out)
    exit()

def evaluate(modelDir):
    #from model
    x_train = None
    y_train = None

    #from input image
    x_test = None
    y_test = None

    images, pca_features, labels = pickle.load(open(os.path.join(modelDir, 'features.p'), 'rb'))
    for image, feature, label in list(zip(images, pca_features, labels)):
        print("Image: {}, Feature: {}, Label: {}".format(image, feature, label))
        break
    K = 10
    neigh = KNeighborsClassifier(n_neighbors=K)
    neigh.fit(x_train, y_train)
    score = neigh.score(x_test, y_test)
    y_pred = neigh.predict(x_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred) 

    save_dir = os.path.join(constants.DATA_PATH, "features", "real", args.netName)

def main(args, modelDir):
    #evaluate(modelDir)

    logitsName = "gan/generator/encoder/fc6"
    ckPath = os.path.join(constants.ROOT_PATH, "Seenomaly", "models", args.netName, f"model.ckpt-{args.checkpoint}")
    imagePaths = [os.path.join(constants.DATA_PATH,"custom","label.txt")]
    saveDir = os.path.join(constants.DATA_PATH,"custom")
    extract_features.extract_features(args.netName, ckPath, logitsName, imagePaths, saveDir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Process a file through the model")
    #parser.add_argument("file", nargs='?', help="file to be processed", default=None)
    parser.add_argument("-o", "--out", help="overwrites the default output name (format is automatically added)", default=os.path.join(constants.DATA_PATH, "results", "result"))
    parser.add_argument("-n", "--netName", help="chooses the network type to be used", choices= ("gan", "vae", "vaegan", "aernn"), default="gan")
    parser.add_argument("-c", "--checkpoint", help="sets the checkpint number", type=int, default=29471)

    args = parser.parse_args()

    # Verification
    #if args.file == None: printExit("File must be present, use --help for more information.")

    modelDir = os.path.join(constants.DATA_PATH, 'features', 'real', args.netName)
    main(args, modelDir)