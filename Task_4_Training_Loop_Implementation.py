#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
# numpy version=1.26, python version 3.10.12
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, logging
from peft import get_peft_model, LoraConfig, PeftConfig
import bitsandbytes as bnb
import numpy as np

# %%
from Task_1_Sentence_Transformer_Implementation import get_double_quantized_model, Sentence_Transformer
from Task_2_Multi_Task_Learning_Expansion import MultiTaskClassifier, get_dataset

#%%

model, tokenizer = get_double_quantized_model()
Sentence_Transformer = Sentence_Transformer(model, tokenizer)
MultiTaskClassifier = MultiTaskClassifier(Sentence_Transformer, 4096, 32, 1, 1)
dataset1, dataset2 = get_dataset()

#%%
# construct the dataset, with each sample contains what tasks it belongs to
class Preprocessor:
    def __init__(
            self,  
            tokenizer, 
            max_length=512,
            task_name='cola',
        ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_name = task_name
    
    def preprocess(self, data):
        batch_size = len(data['sentence'])
        inputs = [f"Task: {self.task_name}. {'sentence'} : {x} This sentence is : " for x in data['sentence']]
        model_inputs = self.tokenizer(inputs)
        # pdb.set_trace()
        for i in range(batch_size):
            model_inputs['attention_mask'][i] = [1] * len(model_inputs['input_ids'][i])
        # padding to the same length for transformer model
        for i in range(batch_size):
            sample_input_ids = model_inputs['input_ids'][i]
            model_inputs['input_ids'][i] = [self.tokenizer.pad_token_id] * (self.max_length - len(sample_input_ids)) + sample_input_ids
            model_inputs['attention_mask'][i] = [0] * (self.max_length - len(sample_input_ids)) + model_inputs['attention_mask'][i]
        model_inputs['labels'] = data['label']
        model_inputs['task'] = [self.task_name] * batch_size
        return model_inputs
        
    def __call__(self, dataset):
        dataset = dataset.map(self.preprocess, batched=True)
        return dataset
    
dataset1 = Preprocessor(tokenizer, 512, 'cola')(dataset1)
dataset2 = Preprocessor(tokenizer, 512, 'sst2')(dataset2)
# %%
from torch.utils.data import WeightedRandomSampler, DataLoader
# get the dataloader with equal number of samples for each class
def get_equal_datalaoder(dataset, batch_size=2):
    labels = [sample['labels'] for sample in dataset['train']]
    class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
    weights = 1. / torch.tensor(class_sample_count, dtype=torch.float)
    sample_weights = weights[labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples = len(sample_weights), replacement=True)
    dataloader = DataLoader(dataset['train'], sampler=sampler, batch_size=2)
    return dataloader

dataloader1 = get_equal_datalaoder(dataset1)
dataloader2 = get_equal_datalaoder(dataset2)

# %%
# build training loop
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.nn import BCEWithLogitsLoss
import tqdm

loss_fn = BCEWithLogitsLoss()
MultiTaskClassifier.to('cuda')
optimizer = AdamW(MultiTaskClassifier.get_trainable_parameters(), lr=1e-4)
num_epochs = 10
# for epoch in range(num_epochs):
    # since classifier 1 did not share the same weights with classifier 2, we need to train them separately

    # # train the classifier head 1
    # for i, data in enumerate(dataloader1):
    #     batch = data
    #     input_batch = {k: torch.stack(v).T.to('cuda') for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
    #     with torch.autocast(device_type='cuda'):
    #         outputs = MultiTaskClassifier(**input_batch)[0][:, 0]
    #         labels = batch['labels'].float().to('cuda')
    #         loss = loss_fn(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()
    #     print(f"Epoch {epoch} batch {i} Task 1 loss: {loss.item()}")
    
    # # train the classifier head 2
    # for i, data in enumerate(dataloader2):
    #     batch = data
    #     input_batch = {k: torch.stack(v).T.to('cuda') for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
    #     with torch.autocast(device_type='cuda'):
    #         outputs = MultiTaskClassifier(**input_batch)[0][:, 0]
    #         labels = batch['labels'].float().to('cuda')
    #         loss = loss_fn(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()
    #     print(f"Epoch {epoch} batch {i} Task 2 loss: {loss.item()}")

    # # validation loss 1
    # for i, data in enumerate(dataset1['validation']):
    #     batch = data
    #     input_batch = {k: torch.tensor(v).to('cuda').unsqueeze(0) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
    #     with torch.autocast(device_type='cuda'):
    #         outputs = MultiTaskClassifier(**input_batch)[0][:, 0]
    #         labels = torch.tensor([batch['labels']]).float().to('cuda')
    #         loss = loss_fn(outputs, labels)
    #     print(f"Epoch {epoch} validation Task 1 loss: {loss.item()}")

    # # validation loss 2
    # for i, data in enumerate(dataset2['validation']):
    #     batch = data
    #     input_batch = {k: torch.tensor(v).to('cuda').unsqueeze(0) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
    #     with torch.autocast(device_type='cuda'):
    #         outputs = MultiTaskClassifier(**input_batch)[0][:, 0]
    #         labels = torch.tensor([batch['labels']]).float().to('cuda')
    #         loss = loss_fn(outputs, labels)
    #     print(f"Epoch {epoch} validation Task 2 loss: {loss.item()}")


# test on both tasks
tp1, tn1, fp1, fn1 = 0, 0, 0, 0
for i, data in tqdm.tqdm(enumerate(dataset1['test']), desc='Task 1 testing'):
    batch = data
    input_batch = {k: torch.tensor(v).to('cuda').unsqueeze(0) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
    with torch.autocast(device_type='cuda'):
        outputs = MultiTaskClassifier(**input_batch)[0][:, 0].cpu().detach().numpy()
        labels = batch['labels']
        outputs = outputs > 0
        tp1 += np.sum(outputs & labels)
        tn1 += np.sum(~outputs & ~labels)
        fp1 += np.sum(outputs & ~labels)
        fn1 += np.sum(~outputs & labels)
precision1 = tp1 / (tp1 + fp1)
recall1 = tp1 / (tp1 + fn1)
f1_score1 = 2 * precision1 * recall1 / (precision1 + recall1)
print(f"Task 1 precision: {precision1}, recall: {recall1}, f1_score: {f1_score1}")

tp2, tn2, fp2, fn2 = 0, 0, 0, 0
for i, data in tqdm.tqdm(enumerate(dataset2['test']), desc='Task 1 testing'):
    batch = data
    input_batch = {k: torch.tensor(v).to('cuda').unsqueeze(0) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
    with torch.autocast(device_type='cuda'):
        outputs = MultiTaskClassifier(**input_batch)[0][:, 0].cpu().detach().numpy()
        labels = batch['labels']
        outputs = outputs > 0
        tp2 += np.sum(outputs & labels)
        tn2 += np.sum(~outputs & ~labels)
        fp2 += np.sum(outputs & ~labels)
        fn2 += np.sum(~outputs & labels)
precision2 = tp2 / (tp2 + fp2)
recall2 = tp2 / (tp2 + fn2)
f1_score2 = 2 * precision2 * recall2 / (precision2 + recall2)
print(f"Task 2 precision: {precision2}, recall: {recall2}, f1_score: {f1_score2}")
        
        
'''
hypothetical data is not handled due to the time limit. However, in the training set, for example
for task 1 (cola), we can construct more negative samples from positive samples by randomly changing
or swapping words in the sentence. For task 2 (sst2), we can construct more negative samples by randomly
changing the sentiment words of the sentence. 

For metrics I use precision, recall and f1 score. Precision is the ratio of true positive to the sum of true positive and false positive. 
'''
# %%
