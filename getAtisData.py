import os
import torch
import torch.utils.data
from pathlib import Path
from transformers import DistilBertTokenizerFast
from sklearn.model_selection import train_test_split
import pandas as pd

class AtisDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def read_atis_split(split_dir):
    split_dir = Path(split_dir)
    print(split_dir)
    texts = []
    labels = []

    text_file = "seq.in"
    label_file = 'label'

    with open(os.path.join(split_dir, text_file)) as f:
        for x in f:
            texts.append(x.rstrip("\n"))
    f.close()


    with open(os.path.join(split_dir, label_file)) as f:
        for x in f:
            labels.append(x.rstrip("\n"))
    f.close()


    return texts, labels


def getAtisData():
    train_texts, train_labels = read_atis_split('AtisData/train') 
    test_texts, test_labels = read_atis_split('AtisData/test')
    validation_texts, validation_labels = read_atis_split('AtisData/valid')
    
    test_texts_ = test_texts + validation_texts
    test_labels_ =test_labels + validation_labels

    columns = ["text", "intent"]    
    df_atis_train = pd.DataFrame(list(zip(train_texts, train_labels)), columns =columns)
    df_atis_test = pd.DataFrame(list(zip(test_texts_, test_labels_)), columns =columns)
    
    df_atis_train["intent"] = 1
    df_atis_test["intent"] = 1
    
    return df_atis_train, df_atis_test