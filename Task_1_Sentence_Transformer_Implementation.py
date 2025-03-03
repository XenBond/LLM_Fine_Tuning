import torch
import torch.nn as nn
import torch.nn.functional as F
# numpy version=1.26, python version 3.10.12
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, logging
from peft import get_peft_model, LoraConfig, PeftConfig
import bitsandbytes as bnb

# %%
# Task 1: Sentence Transformer Implementation
'''
This task requires me to implement a sentence transformer model that takes a sentence as input and output a fixed-size embedding.
A basic implementation has a tokenizer, and a transformer model. 

The tokenizer is used to convert the input sentence into tokens, the tokens are trained by Byte pair encoding (BPE) algorithm, try to 
extract most frequent subwords from the training data. The tokenizer should contains a vocabulary, which is a dictionary that maps
subwords to indices. The tokenizer should also contains a method that converts a sentence into tokens, and a method that converts tokens
into sentence.

The transformer model is a deep learning model that is based on the transformer architecture. The transformer model's input is a sequence
of token indices, then look up the embeddings of the tokens from an trained embedding matrix. The embeddings are then passed through a
series of transformer layers, which contains multi-head self-attention mechanism and feed-forward neural network. The output of the
transformer model is a sequence of hidden states, which can be used to represent the input sentence.

Implementing the tokenizer and train the transformer model has too much work and might be out of scope of this task. So for simplicity, 
I use the transformer model that is already trained on a large dataset, and use it to extract the hidden states of the last layer of the 
model. Then I will take the mean of the hidden states to get a fixed-size embedding of the input sentence.

The model that I will use is the DeepSeek-R1-Distill-Llama-8B model, which is a transformer model that is trained on a large dataset.
In some cases, I can also add a Lora adaptor to the model to improve the performance, if necessary. 

The model is quantized using the BitsAndBytes library, which double quantized the llama model to 4-bit weights. 
'''

def get_double_quantized_model(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 
        quantization_config=None, 
        device_map='cuda:0', 
        torch_type=torch.float16,
        max_seq_len=512,
    ):
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch_type,
    )
    # load the llama 3 8B model distilled by DeepSeek-R1
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,    
        device_map=device_map,
        torch_dtype=torch_type
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", max_seq_len=max_seq_len)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

class Sentence_Transformer(nn.Module):
    def __init__(self, model=None, tokenizer=None):
        super().__init__()
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        if self.model is None:
            self.model, self.tokenizer = get_double_quantized_model()
        
    def forward(self, input_ids, attention_mask, **kwargs):
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        mean_hidden_states = torch.mean(outputs.hidden_states[-1], dim=1) # shape: (batch_size, hidden_size)
        return mean_hidden_states
