import os

ROOT_PATH = ".." #Relative path should work fine, if issues arrise set to the absoulute path of the Seenomaly containing directory.
UNQUALIFIED_DATA_PATH = "data"
DATA_PATH = os.path.join(ROOT_PATH, "Seenomaly", UNQUALIFIED_DATA_PATH) #Path to data, which includes training, test, tagged data and the resulting features files.
