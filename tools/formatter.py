import numpy as np
from sklearn.model_selection import train_test_split

def preprocess(data_set = "../datasets/fertility.txt", test_size=0.2, random_state=42):
    ### the words (features) and authors (labels), already largely preprocessed
    ### this preprocessing will be repeated in the text learning mini-project
    datasets_file_handler = open(data_set, "r")
    data_dict = {
        'season': [],
        'age': [],
        'child_diseases': [],
        'accident': [],
        'surgical_intervention': [],
        'high_fevers': [],
        'alcohol_consumption': [],
        'smoking': []
    }

    labels = []
    features = []

    for line in datasets_file_handler:
        stripped_line = line.rstrip();
        lines_data = stripped_line.split(',')
        for idx, key in enumerate(data_dict.keys()):
            data_dict[key].append(lines_data[idx])
        features.append(np.array(lines_data[0:-1]))
        labels.append(lines_data[-1])

    datasets_file_handler.close()

    X_train, X_test, y_train, y_test = train_test_split(np.array(features), np.array(labels), test_size=test_size, random_state=random_state)
    print(X_train, X_test, y_train, y_test)
    # features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)


preprocess()
