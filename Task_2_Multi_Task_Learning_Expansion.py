import torch.nn as nn
from datasets import load_dataset

# %%
# Task 2: Multi-Task Learning Expansion. 
'''
For simplicity, I use the GLUE dataset's CoLA and SST-2 tasks. The CoLA task is a single-sentence classification task, where the model
is required to classify whether a sentence is grammatically correct or not. The SST-2 task is a single-sentence classification task, where
the model is required to classify the sentiment of a sentence.

The change made to the model is to add two classifiers to the model, one for each task. The model will output two logits, one for each.
The classifiers are simple feed-forward neural networks with 1 hidden layer and ReLU activation function. The input to the classifiers
is the fixed-size embedding of the input sentence. The output of the classifiers is the logits, which are used to compute the loss of the
model.

The reason to choose such simple model based on the assumption that the transformer model is already trained on a large dataset, and the
hidden states of the last layer of the model should contain enough information to represent the input sentence. The hidden states are then
used to classify the input sentence into the two tasks.
'''

# Define the Multi-Task Classifier model
class MultiTaskClassifier(nn.Module):
    def __init__(self, 
            base_model, 
            embedding_size,
            hidden_size, 
            num_classes_task_a, 
            num_classes_task_b
        ):
        super().__init__()
        self.base_model = base_model
        self.classifier_a = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes_task_a)
        )
        self.classifier_b = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes_task_b)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        logits_a = self.classifier_a(outputs)
        logits_b = self.classifier_b(outputs)
        return logits_a, logits_b

    def get_trainable_parameters(self):
        return list(self.classifier_a.parameters()) + list(self.classifier_b.parameters())

#%%
def get_dataset():
    # get the GLUE dataset with CoLA and SST-2 tasks
    # task A. Sentence Classification: CoLA
    dataset1 = load_dataset('glue', 'cola')
    # task B. Sentiment Classification: SST-2
    dataset2 = load_dataset('glue', 'sst2')
    print(dataset1)
    print(dataset2)
    return dataset1, dataset2