from argparse import ArgumentParser
import json
import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from constants import *
from tqdm import tqdm 
import string
import torch
import numpy as np
from dataset import SocialNormDataset
from datasets import DatasetDict, Dataset, Features, Value, load_metric
from transformers import AdamW, AutoTokenizer, BertTokenizerFast, DataCollatorWithPadding, get_scheduler
from torch.utils.data import DataLoader

from models import JudgeBert, SentBertClassifier
from utils.train_utils import loss_fn
from utils.utils import NpEncoder, get_current_timestamp, get_samples_per_class, print_args

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
    all_ids = ['verdict_ids']
    all_pred = ['predictions']
    all_labels = ['gold labels']
    
    for batch in dataloader:
        verdicts_index = batch.pop("index")
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        labels = batch.pop("labels")
        with torch.no_grad():
            logits = model(batch)

        predictions = torch.argmax(logits, dim=-1)
        accuracy_metric.add_batch(predictions=predictions, references=labels)
        f1_metric.add_batch(predictions=predictions, references=labels)
        all_pred.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_ids.extend(verdicts_index.numpy())
    
    if return_predictions:
        return {'accuracy': accuracy_metric.compute()['accuracy'], 'f1_score': f1_metric.compute()['f1'], 
                'results': list(zip(all_ids, all_pred, all_labels))}

    return {'accuracy': accuracy_metric.compute()['accuracy'], 'f1_score': f1_metric.compute()['f1']}

# python ft_social_norms.py --model_name='judge_bert' --stratify='fulltext' --ratio=0.3
parser = ArgumentParser()
parser.add_argument("--path_to_data", dest="path_to_data", required=True, type=str)
parser.add_argument("--sbert_model", dest="sbert_model", default='sentence-transformers/all-MiniLM-L6-v2', type=str)     #sbert_model = 'sentence-transformers/all-distilroberta-v1'
parser.add_argument("--sbert_dim", dest="sbert_dim", default=384, type=int)
parser.add_argument("--num_epochs", dest="num_epochs", default=10, type=int)
parser.add_argument("--learning_rate", dest="learning_rate", default=1e-4, type=float)
parser.add_argument("--batch_size", dest="batch_size", default=32, type=int)
parser.add_argument("--loss_type", dest="loss_type", default='focal', type=str)
parser.add_argument("--bert_tokenizer", dest="bert_tokenizer", default='bert-base-uncased', type=str)
parser.add_argument("--model_name", dest="model_name", type=str, required=True) # ['judge_bert', 'sbert'] otherwise exception
parser.add_argument("--stratify", dest="stratify", type=str, required=True)
parser.add_argument("--ratio", dest="ratio", type=float, required=True)
parser.add_argument("--results_dir", dest="results_dir", type=str, default='../results')

if __name__ == '__main__':
    args = parser.parse_args()
    print_args(args, logging)
    
    path_to_data = args.path_to_data
    model_name = args.model_name
    results_dir = args.results_dir
    sbert_model = args.sbert_model
    sbert_dim = args.sbert_dim
    stratify = args.stratify
    
    checkpoint_dir = os.path.join(results_dir, f'best_models/{TIMESTAMP}_best_model_sampled.pt')

    social_chemistry = pd.read_pickle(path_to_data +'social_chemistry_clean_with_fulltexts.gzip', compression='gzip')

    with open(path_to_data+'social_norms_clean.csv') as file:
        social_comments = pd.read_csv(file)
        
    logging.info("Using DEVICE {}".format(DEVICE))
    
    if stratify == 'fulltext':
        sit_louvain_labels = np.load('../louvain_labels_and_embeddings/fulltext_louvain_labels.npy', allow_pickle=True).item()
    elif stratify == 'situation':
        sit_louvain_labels = np.load('../louvain_labels_and_embeddings/situation_louvain_labels.npy', allow_pickle=True).item()
    else:
        raise Exception("The stratify argument was {}. It should be either fulltext or situation.".format(stratify))
    #rot_louvain_labels = np.load('../louvain_labels_and_embeddings/rot_louvain_labels.npy', allow_pickle=True).item()
    
    dataset = SocialNormDataset(social_comments, social_chemistry)
    verdict_ids = list(dataset.verdictToLabel.keys())
    labels = list(dataset.verdictToLabel.values())
    verdictToLouvainLabel = {}
    louvain_labels = list()
    ratio = args.ratio # So, titles=0.4, fulltext=0.3, rots=0.5 maybe
    logging.info("Stratifying by ** {} **  with ratio ** {} **".format(stratify, ratio))
    
    for v in verdict_ids:
        parent = dataset.verdictToParent[v]
        idx = dataset.postIdToId[parent]
        lv_label = sit_louvain_labels[ratio][idx]
        verdictToLouvainLabel[v] = lv_label
        louvain_labels.append(lv_label)

    train_verdicts, test_verdicts, train_labels, test_labels = train_test_split(verdict_ids, labels, test_size=0.2, 
                                                                            random_state=SEED, stratify=louvain_labels)

    train_verdicts, val_verdicts, train_labels, val_labels = train_test_split(train_verdicts, train_labels, test_size=0.15, 
                                                                        random_state=SEED)


    
    raw_dataset = {'train': {'index': [], 'text': [], 'label': []}, 
            'val': {'index': [], 'text': [], 'label': [] }, 
            'test': {'index': [], 'text': [], 'label': [] }}


    for i, verdict in enumerate(train_verdicts):
        situation_text = dataset.postIdToTitle[dataset.verdictToParent[verdict]]
        if situation_text != '' and situation_text is not None: 
            raw_dataset['train']['index'].append(dataset.verdictToId[verdict])
            raw_dataset['train']['text'].append(situation_text + ' [SEP] ' + dataset.verdictToCleanedText[verdict])
            raw_dataset['train']['label'].append(train_labels[i])
            assert train_labels[i] == dataset.verdictToLabel[verdict] 
        
    for i, verdict in enumerate(val_verdicts):
        situation_text = dataset.postIdToTitle[dataset.verdictToParent[verdict]]
        if situation_text != '' and situation_text is not None: 
            raw_dataset['val']['index'].append(dataset.verdictToId[verdict])
            raw_dataset['val']['text'].append(situation_text + ' [SEP] ' + dataset.verdictToCleanedText[verdict])
            raw_dataset['val']['label'].append(val_labels[i])   
            assert val_labels[i] == dataset.verdictToLabel[verdict] 

    for i, verdict in enumerate(test_verdicts):
        situation_text = dataset.postIdToTitle[dataset.verdictToParent[verdict]]
        if situation_text != '' and situation_text is not None: 
            raw_dataset['test']['index'].append(dataset.verdictToId[verdict])
            raw_dataset['test']['text'].append(situation_text + ' [SEP] ' + dataset.verdictToCleanedText[verdict])
            raw_dataset['test']['label'].append(test_labels[i])
            assert test_labels[i] == dataset.verdictToLabel[verdict] 
            
    if model_name == 'sbert':
        logging.info("Training with SBERT, model name is {}".format(model_name))
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = SentBertClassifier(users_layer=False, sbert_model=sbert_model, sbert_dim=sbert_dim)
    elif model_name == 'judge_bert':
        logging.info("Training with Judge Bert, model name is {}".format(model_name))
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        model = JudgeBert()
    else:
        raise Exception('Wrong model name')

    model.to(DEVICE)
    ds = DatasetDict()

    for split, data in raw_dataset.items():
        ds[split] = Dataset.from_dict(mapping=data, features=Features({'label': Value(dtype='int64'), 
                                                                        'text': Value(dtype='string'), 'index': Value(dtype='int64')}))

    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True)

    tokenized_dataset = ds.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")
    
    batch_size = args.batch_size
    
    train_dataloader = DataLoader(
        tokenized_dataset["train"], batch_size=batch_size, collate_fn=data_collator, shuffle = True
    )
    eval_dataloader = DataLoader(
        tokenized_dataset["val"], batch_size=batch_size, collate_fn=data_collator
    )
    
    test_dataloader = DataLoader(
        tokenized_dataset["test"], batch_size=batch_size, collate_fn=data_collator
    )
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    
    num_epochs = args.num_epochs
    num_training_steps = num_epochs * len(train_dataloader)
    samples_per_class_train = get_samples_per_class(tokenized_dataset["train"]['labels'])

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    logging.info("Number of training steps {}".format(num_training_steps))
    loss_type='focal'
    

    progress_bar = tqdm(range(num_training_steps))
    best_accuracy = 0
    best_f1 = 0
    val_metrics = []
    train_loss = []
    logging.info("Initial evaluation")
    val_metric = evaluate(eval_dataloader, model)
    logging.info(val_metric)
    c = 0
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            c += 1
            verdicts_index = batch.pop("index")
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = batch.pop("labels")
            output = model(batch)
            
            loss = loss_fn(output, labels, samples_per_class_train, loss_type=loss_type)
            train_loss.append(loss.item())
            loss.backward()
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            
            if c % 1000 == 0:
                val_metric = evaluate(eval_dataloader, model)
                val_metrics.append(val_metric)
                
                logging.info("Epoch {} ---- Metrics validation: {}".format(epoch, val_metric))
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
    
    for i, entry in enumerate(results[1:]):
        entry = list(entry)
        entry[0] = dataset.idToVerdict[entry[0]]
        results[i+1] = entry
    
    result_logs = {'id': TIMESTAMP}
    result_logs['model_name'] = model_name
    
    if model_name == 'judge_bert':
        result_logs['bert_model'] = 'BERT'
        result_logs['bert_dim'] = 768   
    else:
        result_logs['sbert_model'] = sbert_model
        result_logs['sbert_dim'] = sbert_dim
        
    result_logs['stratify'] = stratify
    result_logs['epochs'] = num_epochs
    result_logs['optimizer'] = optimizer.defaults
    result_logs["loss_type"] = loss_type
    result_logs['test_metrics'] = test_metrics
    result_logs['checkpoint_dir'] = checkpoint_dir
    result_logs['val_metrics'] = val_metrics
    result_logs['results'] = results
    
    
    res_file = os.path.join(results_dir, TIMESTAMP + ".json")
    with open(res_file, mode='w') as f:
        json.dump(result_logs, f, cls=NpEncoder, indent=2)