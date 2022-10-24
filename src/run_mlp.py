from argparse import ArgumentParser
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm 
import torch
import numpy as np
import pickle as pkl
from dataset import SocialNormDataset
from datasets import DatasetDict, Dataset, Features, Value, Array2D
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import get_scheduler
from utils.train_utils import *
from utils.utils import *
from models import MLP
import logging
import os
from constants import *

TIMESTAMP = get_current_timestamp()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"../logs/{TIMESTAMP}.log"),
        logging.StreamHandler()
    ]
)

def evaluate(dataloader, model, return_predictions=False):
    accuracy_metric = load_metric("accuracy")
    f1_metric = load_metric("f1")
    
    model.eval()
    all_situations = ['situations']
    all_ids = ['verdicts']
    all_pred = ['predictions']
    all_labels = ['gold labels']
    
    for batch in dataloader:
        verdicts = batch.pop("verdict")
        situations = batch.pop("situation")
        labels = batch.pop("label").to(DEVICE)

        verdict_embedding = torch.stack(batch["verdict_embeddings"][0]).transpose(1, 0).to(DEVICE)
        situation_embedding = torch.stack(batch["situation_embeddings"][0]).transpose(1, 0).to(DEVICE)

        embedding = torch.cat([situation_embedding, verdict_embedding], dim=1)
        with torch.no_grad():
            logits = model(embedding.float())

        predictions = torch.argmax(logits, dim=-1)
        accuracy_metric.add_batch(predictions=predictions, references=labels)
        f1_metric.add_batch(predictions=predictions, references=labels)
        all_pred.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_ids.extend(verdicts)
        all_situations.extend(situations)
    
    if return_predictions:
        return {'accuracy': accuracy_metric.compute()['accuracy'], 'f1_score': f1_metric.compute()['f1'], 
                'results': list(zip(all_situations, all_ids, all_pred, all_labels))}

    return {'accuracy': accuracy_metric.compute()['accuracy'], 'f1_score': f1_metric.compute()['f1']}

parser = ArgumentParser()
parser.add_argument("--path_to_data", dest="path_to_data", required=True, type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    print_args(args, logging)
    path_to_data = args.path_to_data
    
    social_chemistry = pd.read_pickle(path_to_data +'social_chemistry_clean_with_fulltexts.gzip', compression='gzip')

    with open(path_to_data+'social_norms_clean.csv') as file:
        social_comments = pd.read_csv(file)
        

    dataset = SocialNormDataset(social_comments, social_chemistry)
    
    stratify = 'fulltext'
    fulltext_louvain_labels = np.load('../louvain_labels_and_embeddings/fulltext_louvain_labels.npy', allow_pickle=True).item()
    #situation_louvain_labels = np.load('../louvain_labels_and_embeddings/situation_louvain_labels.npy', allow_pickle=True).item()
    # rot_louvain_labels = np.load('../louvain_labels_and_embeddings/rot_louvain_labels.npy', allow_pickle=True).item()
    verdict_ids = list(dataset.verdictToLabel.keys())
    labels = list(dataset.verdictToLabel.values())
    verdictToLouvainLabel = {}
    louvain_labels = list()
    ratio = 0.3
    #  titles=0.4, fulltext=0.3, rots=0.5 

    for v in verdict_ids:
        parent = dataset.verdictToParent[v]
        idx = dataset.postIdToId[parent]
        lv_label = fulltext_louvain_labels[ratio][idx]
        verdictToLouvainLabel[v] = lv_label
        louvain_labels.append(lv_label)

    SEED = 1234
    verdict_ids = list(dataset.verdictToLabel.keys())
    labels = list(dataset.verdictToLabel.values())

    train_verdicts, test_verdicts, train_labels, test_labels = train_test_split(verdict_ids, labels, test_size=0.2, 
                                                                            random_state=SEED, stratify=louvain_labels)

    train_verdicts, val_verdicts, train_labels, val_labels = train_test_split(train_verdicts, train_labels, test_size=0.15, 
                                                                        random_state=SEED)

    # verdict_embeddings = pkl.load(open('../data/embeddings/titleVerdicts_situationBert.pkl', 'rb'))
    # situation_embeddings = pkl.load(open('../data/embeddings/posts_situationsSBert.pkl', 'rb'))
    verdict_embeddings = pkl.load(open('../data/embeddings/titleVerdicts_realFullTextSBert.pkl', 'rb'))
    situation_embeddings = pkl.load(open('../data/embeddings/posts_fullTextSBert.pkl', 'rb'))
    
    
    raw_dataset = {'train': {'verdict': [], 'situation': [], 'label': [], 'verdict_embeddings': [], 'situation_embeddings': []}, 
            'val': {'verdict': [], 'situation': [], 'label': [], 'verdict_embeddings': [], 'situation_embeddings': [] }, 
            'test': {'verdict': [], 'situation': [], 'label': [], 'verdict_embeddings': [], 'situation_embeddings': [] }}


    for i, verdict in enumerate(train_verdicts):
        situation_text = dataset.postIdToTitle[dataset.verdictToParent[verdict]]
        post = dataset.verdictToParent[verdict]
        if situation_text != '' and situation_text is not None and post in situation_embeddings: 
            raw_dataset['train']['verdict'].append(verdict)
            raw_dataset['train']['situation'].append(post)
            raw_dataset['train']['label'].append(dataset.verdictToLabel[verdict])
            raw_dataset['train']['verdict_embeddings'].append(verdict_embeddings[verdict].unsqueeze(0))
            raw_dataset['train']['situation_embeddings'].append(situation_embeddings[post].unsqueeze(0))


    for i, verdict in enumerate(val_verdicts):
        situation_text = dataset.postIdToTitle[dataset.verdictToParent[verdict]]
        post = dataset.verdictToParent[verdict]
        if situation_text != '' and situation_text is not None and post in situation_embeddings: 
            raw_dataset['val']['verdict'].append(verdict)
            raw_dataset['val']['situation'].append(post)
            raw_dataset['val']['label'].append(dataset.verdictToLabel[verdict])
            raw_dataset['val']['verdict_embeddings'].append(verdict_embeddings[verdict].unsqueeze(0))
            raw_dataset['val']['situation_embeddings'].append(situation_embeddings[post].unsqueeze(0))

        
    for i, verdict in enumerate(test_verdicts):
        situation_text = dataset.postIdToTitle[dataset.verdictToParent[verdict]]
        post = dataset.verdictToParent[verdict]
        if situation_text != '' and situation_text is not None and post in situation_embeddings: 
            raw_dataset['test']['verdict'].append(verdict)
            raw_dataset['test']['situation'].append(post)
            raw_dataset['test']['label'].append(dataset.verdictToLabel[verdict])
            raw_dataset['test']['verdict_embeddings'].append(verdict_embeddings[verdict].unsqueeze(0))
            raw_dataset['test']['situation_embeddings'].append(situation_embeddings[post].unsqueeze(0))


    ds = DatasetDict()

    for split, data in raw_dataset.items():
        ds[split] = Dataset.from_dict(mapping=data, features=Features({'label': Value(dtype='int64'), 
                                                                        'situation': Value(dtype='string'), 'verdict': Value(dtype='string'),
                                                                        'verdict_embeddings': Array2D(shape=(1, 384), dtype='float32'),
                                                                        'situation_embeddings': Array2D(shape=(1, 384), dtype='float32')
                                                                        }))
        
    batch_size = 64

    train_dataloader = DataLoader(
        ds["train"], batch_size=batch_size, shuffle = True
    )
    eval_dataloader = DataLoader(
        ds["val"], batch_size=batch_size
    )

    test_dataloader = DataLoader(
        ds["test"], batch_size=batch_size
    )
    
    model = MLP()
    model.to(DEVICE)
    
    optimizer = AdamW(model.parameters(), lr=1e-3)

    samples_per_class_train = get_samples_per_class(torch.tensor(raw_dataset["train"]['label']))
    num_epochs = 5
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    print(num_training_steps)
    
    progress_bar = tqdm(range(num_training_steps))
    loss_type = 'focal'
    best_accuracy = 0
    best_f1 = 0
    train_metrics = []
    val_metrics = []
    train_loss = []

    checkpoint_dir = '../results/best_models/debugging.pt'

    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            verdicts = batch.pop("verdict")
            situations = batch.pop("situation")
            labels = batch.pop("label").to(DEVICE)

            verdict_embedding = torch.stack(batch["verdict_embeddings"][0]).transpose(1, 0).to(DEVICE)
            situation_embedding = torch.stack(batch["situation_embeddings"][0]).transpose(1, 0).to(DEVICE)

            embedding = torch.cat([situation_embedding, verdict_embedding], dim=1)        
            output = model(embedding.float())
            
            loss = loss_fn(output, labels, samples_per_class_train, loss_type=loss_type)
            loss.backward()
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        
        val_metric = evaluate(eval_dataloader, model)
        val_metrics.append(val_metric)
        
        print("Epoch {} ---- Metrics validation: {}".format(epoch, val_metric))
        if val_metric['accuracy'] > best_accuracy:
            print("Saving best model")
            best_accuracy = val_metric['accuracy'] 
            torch.save(model.state_dict(), checkpoint_dir)
            
    logging.info("Evaluating")
    model.load_state_dict(torch.load(checkpoint_dir))
    model.to(DEVICE)
    test_metrics = evaluate(test_dataloader, model, True)
    results = test_metrics.pop('results')
    logging.info(test_metrics)
    
    result_logs = {'id': TIMESTAMP}
    result_logs['model_name'] = 'MLP'
    result_logs['stratify'] = stratify
    result_logs['epochs'] = num_epochs
    result_logs['optimizer'] = optimizer.defaults
    result_logs["loss_type"] = loss_type
    result_logs['test_metrics'] = test_metrics
    result_logs['checkpoint_dir'] = checkpoint_dir
    result_logs['val_metrics'] = val_metrics
    result_logs['results'] = results
    
    results_dir = '../results'
    res_file = os.path.join(results_dir, TIMESTAMP + ".json")
    with open(res_file, mode='w') as f:
        json.dump(result_logs, f, cls=NpEncoder, indent=2)