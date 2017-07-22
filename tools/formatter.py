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
        'smoking': [],
        'number_of_hours_sitting': [],
    }

    labels = []
    features = []

    to_float = np.vectorize(float)
    for line in datasets_file_handler:
        stripped_line = line.rstrip();
        lines_data = stripped_line.split(',')
        for idx, key in enumerate(data_dict.keys()):
            data_dict[key].append(lines_data[idx])

        feature = np.array(lines_data[0:-1])
        features.append(to_float(feature))
        is_diagnosed = 1 if (lines_data[-1] == 'O') else 0
        labels.append(is_diagnosed)

    datasets_file_handler.close()

    features_train, features_test, labels_train, labels_test = train_test_split(np.array(features), np.array(labels), test_size=test_size, random_state=random_state)

    return features_train, features_test, labels_train, labels_test, features, labels, data_dict
