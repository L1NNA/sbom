import pickle
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
import torch
import random
import os
from collections import defaultdict
from sentence_transformers import SentenceTransformer, InputExample, losses, util, evaluation
from torch.utils.data import DataLoader
from sentence_transformers import evaluation
import sklearn.metrics as metrics
import numpy as np
import pandas as pd
import csv
import logging
from itertools import combinations


def process_data(al):
    all_examples = defaultdict(list)
    all_libraries = set()

    for product, package, files, invalid_strs in tqdm(al):
        all_libraries.add(product.lower())

    all_libraries = sorted(list(all_libraries))
    print(len(all_libraries))

    all_invalids = set()
    num_negative = 1

    for product, package, files, invalid_strs in tqdm(al):
        product = product.lower()
        package_name = os.path.basename(package).lower()
        
        for file in files:
            strs = files[file]
            for version_string in set(strs):
                version_string = version_string.lower()
                # using specific product as key, relate product to all version strs
                all_examples[product].append(
                    InputExample(texts=[version_string, product], label=1.0)
                )
                
                for _ in range(num_negative):
                    # maybe also train on invalid strs?
                    another_lib = random.choice(all_libraries)
                    label = 0 if another_lib != product else 1
                    # using specific product as key, decorrelate other products to 
                    # the version strs for product
                    all_examples[product].append(
                        InputExample(texts=[version_string, another_lib], label=label)
                    )
        
        for ivs in invalid_strs:
            all_invalids.add(ivs.lower())
    
    all_invalids = list(all_invalids)

    print(len(all_libraries), all_examples['openssl'][1].texts, all_examples['openssl'][1].label)
    return all_libraries, all_examples


def show_classes(all_libraries, all_examples):
    total = 0 
    classes = []; dataset = []
    for product in all_examples:
        n = len(all_examples[product])
        classes.append([product, n])
        total += n
        dataset.append([product, all_examples[product]])
    sorted_dataset = sorted(dataset, key=lambda x: len(x[1]), reverse=True)
    sorted_classes = sorted(classes, key=lambda x: x[1], reverse=True)
    sorted_classes.append([["product", "n samples"]])
    sorted_classes.append(["TOTAL", total])
    csv_file = "classes.csv"
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(sorted_classes)
    return total, sorted_dataset


def split_out_of_sample(dataset, total):
    random.seed(10)
    train_set = []; test_set = []; 
    train_sz = int(total*0.8)
    test_sz = total - train_sz
    total_test = 0; total_train = 0
    data = [["product", "n samples", "set"]]
    i = 0
    while total_train < train_sz:
        try:
            samples = random.sample(dataset[i][1], 4000)
        except:
            samples = dataset[i][1]
        train_set.extend(samples)
        total_train += len(samples)
        data.append([dataset[i][0], len(samples), "train"])
        i += 1

    all_pairs = []
    for product, examples in dataset[i:]:
        for example in examples:
            all_pairs.append((product, example))
    sampled_idx = random.sample(range(len(all_pairs)), test_sz)
    pairs = [(all_pairs[i][0], all_pairs[i][1]) for i in sampled_idx]

    frq_dict = defaultdict(int)
    products = []
    for _, (product, example) in enumerate(pairs):
        frq_dict[product] += 1
        test_set.append(example)
        products.append(product)
        total_test += 1

    for product in frq_dict.keys():
        data.append([product, frq_dict[product], "test"])

    random.shuffle(train_set); random.shuffle(test_set)
    data.append(["TOTAL test", total_test, "test"])
    data.append(["TOTAL train", total_train, "train"])
    csv_file = 'split.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print(f"Train set len = {total_train}")
    print(f"Test set len = {total_test}")
    return train_set, test_set, products


def split_train_test(all_libraries, all_examples, total):
    random.seed(10)
    dataset = []
    for product in all_examples:
        dataset.extend(all_examples[product])
    dataset = random.sample(dataset, total)
    random.shuffle(dataset)
    training_samples = dataset[0:int(len(dataset)*0.8)]
    testing_samples = dataset[int(len(dataset)*0.8):]
    return training_samples, testing_samples


def train(training_samples):
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    # let's try small subset for debugging purpose
    train_dataloader = DataLoader(training_samples, shuffle=True, batch_size=512)
    train_loss = losses.CosineSimilarityLoss(model)
    # Tune the model
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=0) #evaluator=evalu, evaluation_steps=250)
    model.save("1ep-oos_model")
    print("Training done - model saved")
    return model


def test(model, test_samples):
    print("<--- Beginning testing --->")
    y_pred = []; y_true = []
    for i, example in enumerate(test_samples):
        if i % 5_000 == 0:
            print(f"Processing sample {i} out of {len(test_samples)}")
        emb1 = model.encode(example.texts[0])
        emb2 = model.encode(example.texts[1])
        #y = util.cos_sim(emb1, emb2)
        y = util.dot_score(emb1, emb2)
        y_true.append(example.label)
        y_pred.append(y.item())
    return y_pred, y_true


def comp_metrics(y_pred, y_true, logfile):
    fpr, tpr, t = metrics.roc_curve(y_true, y_pred)
    test_auc = metrics.auc(fpr, tpr)
    argmax = np.argmax(tpr - fpr)
    optimal_t = t[argmax]
    correct = 0
    classified = y_pred > optimal_t
    for y_p, y_t in zip(classified, y_true):
        if y_p == y_t:
            correct += 1
    test_acc = correct / len(y_true)
    cmat = metrics.confusion_matrix(y_true, classified)
    pr = metrics.precision_score(y_true, classified)
    rc = metrics.recall_score(y_true, classified)
    f1 = metrics.f1_score(y_true, classified)
    msg = f"Test Accuracy: {test_acc}\t Test AUC: {test_auc}\t Thresh: {optimal_t}\t Precision: {pr}\t Recall: {rc}\t F1: {f1}\t"
    if os.path.exists(logfile):  # Check if the file exists
        os.remove(logfile)
    logging.basicConfig(filename=logfile, level=logging.INFO,
                        format="%(asctime)s - %(message)s")
    logger = logging.getLogger("my_logger")
    logger.info(msg)
    print(msg)
    return classified, logger


def observe_true_pos(logger, classified, y_true, products):
    classes = defaultdict(lambda: [0, 0])
    total_tp = 0; total_fn = 0
    for y_p, y_t, prod in zip(classified, y_true, products):
        if y_t == 1 and y_p == y_t:
            classes[prod][0] += 1
            total_tp += 1
        elif y_t == 1:
            classes[prod][1] += 1
            total_fn += 1

    csv_path= "results_by_class.csv"
    with open(csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Product', 'True Positives', 'False Negatives', 'Total', 'Recall', 'FN Rate'])
        for product, (tp, fn) in classes.items():
            if tp + fn > 0:
                csv_writer.writerow([product, tp, fn, tp+fn, tp/(tp+fn), fn/(tp+fn)])
    msg = f"True Positives: {total_tp}\t False Negatives: {total_fn}"
    logger.info(msg)


def main():
    l_debian = pickle.load(open('extracted_strings4.pickle', 'rb'))
    l_cannon = pickle.load(open('extracted_strings_cannon.pickle', 'rb'))
    al = [*l_debian, *l_cannon]
    all_libraries, all_examples = process_data(al)
    _, dataset = show_classes(all_libraries, all_examples)
    #training_samples, test_samples = split_train_test(all_libraries, all_examples, 100_000) # experiment 1
    training_samples, test_samples, products = split_out_of_sample(dataset, 100_000) # experiment 2
    model = train(training_samples)
    y_pred, y_true = test(model, test_samples)
    classified, logger = comp_metrics(y_pred, y_true, "1ep-oos-dot.log")
    observe_true_pos(logger, classified, y_true, products)

main()