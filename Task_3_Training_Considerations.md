# Task 3: Training Consideration

Discuss the implications and advantages of each scenario and explain your rationale as to how
the model should be trained given the following:
1. If the entire network should be frozen.
    
    ### Answer: 
    In this case, the **prompt tuning** style should be considered. 
    
    #### Implementation
    We should abandom the classifcation
    head since it is not trainable. More over, we can build a trainble embedding as a prefix/suffix (often prefix) of the input with an initialized prompt, and (if necessary) concatenated with fixed human-designed prompt.
    
    For different tasks, we can use different prompt embeddings. For some similar tasks, sometimes part of the prompt embedding can be shared. The model can be trained by using the prompt tuning method by minimizing the negative log-likelihood of the desired output given the prompt. The prompt is a sequence of tokens that
    describes the desired output, and is used to guide the model to generate the desired output. For example: 

    ```python
    from peft import PromptTuningInit, PromptTuningConfig, TaskType, get_peft_model, get_peft_config
    from torch.utils.data import DataLoader
    import pdb
    # define the prompt tuning configuration for CoLA task
    prompt_cola_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=4,
        prompt_tuning_init_text='Classify whether the sentence is grammatically correct or not.',
        tokenizer_name_or_path='deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
    )
    # define the prompt tuning configuration for SST-2 task
    prompt_sst2_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=4,
        prompt_tuning_init_text='Classify the sentiment of the sentence.',
        tokenizer_name_or_path='deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
    )

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
                
                model_inputs['input_ids'][i] = torch.tensor(model_inputs['input_ids'][i][:self.max_length])
                model_inputs['attention_mask'][i] = torch.tensor(model_inputs['attention_mask'][i][:self.max_length])
            return model_inputs
            
        def __call__(self, dataset):
            dataset = dataset.map(self.preprocess, batched=True)
            return dataset
        
    model, tokenizer = get_double_quantized_model()
    model.eval()
    preprocessor1 = Preprocessor(tokenizer=tokenizer, task_name='Check whether the sentence is grammatically correct or not.')
    dataset1_prompt_tuning = preprocessor1(dataset1)
    preprocessor2 = Preprocessor(tokenizer=tokenizer, task_name='Classify the sentiment of the sentence.')
    dataset2_prompt_tuning = preprocessor2(dataset2)

    cola_model = get_peft_model(model, prompt_cola_config)
    sst2_model = get_peft_model(model, prompt_sst2_config)


    # %%
    nu_epoch = 1 # "1 epoch" rule for tuning LLM to avoid overfitting. Can be larger, like 4, of course.

    # here we only train the cola task. since the sst2 task is similar, we can use the same code to train the sst2 task.
    train_1_dataloader = DataLoader(dataset1_prompt_tuning['train'], batch_size=1, shuffle=True)
    val_1_dataloader = DataLoader(dataset1_prompt_tuning['validation'], batch_size=1, shuffle=False)
    test_1_dataloader = DataLoader(dataset1_prompt_tuning['test'], batch_size=1, shuffle=False)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(cola_model.parameters(), lr=1e-5)
    cola_model = cola_model.to('cuda')


    for epoch in range(nu_epoch):
        cola_model.train()
        for step, batch in enumerate(train_1_dataloader):
            print(batch)
            input_batch = {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            input_batch = {k: torch.stack(v).T.to('cuda') for k, v in input_batch.items()}
            labels = batch['label']
            label_tokens = tokenizer("correct", return_tensors="pt")['input_ids'][0][0]
            
            with torch.autocast(device_type='cuda'):
                outputs = cola_model(**input_batch)
                # use next token prediction loss
                logits = outputs.logits[:, -1, :].contiguous()
                label_batch = label_tokens.repeat(logits.shape[0]).to('cuda')
                
                # get logits in "correct" tokenized for each batch
                predicted_logits = logits[:, label_tokens]
                loss = loss_fn(predicted_logits, labels.float().to('cuda'))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            print(f"epoch {epoch}: step {step}: Loss: {loss.item()}")

            # validate the model
            cola_model.eval()
            eval_loss = 0
            tp, tn, fp, fn = 0, 0, 0, 0
            with torch.no_grad():
                eval_loss = 0
                for step, batch in enumerate(val_1_dataloader):
                    input_batch = {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
                    input_batch = {k: torch.stack(v).T.to('cuda') for k, v in input_batch.items()}
                    labels = batch['label']
                    label_tokens = tokenizer("correct", return_tensors="pt")['input_ids'][0][0]
                    with torch.autocast(device_type='cuda'):
                        outputs = cola_model(**input_batch)
                        # use next token prediction loss
                        logits = outputs.logits[:, -1, :].contiguous()
                        label_batch = label_tokens.repeat(logits.shape[0]).to('cuda')
                        
                        # get logits in "correct" tokenized for each batch
                        predicted_logits = logits[:, label_tokens]
                        loss = loss_fn(predicted_logits, labels.float().to('cuda'))
                        eval_loss += loss.item()
                eval_loss /= len(val_1_dataloader)
                print(f"epoch {epoch}: validation loss={eval_loss}")
    ```
    
    #### Scenario and reason
        
    When training data is limited, or the backbone model itself is power enough, we don't need to tune the parameters
    of the LLM model. Meanwhile, it also reduce the computation cost since it doesn't need to store the gradients and optimization parameters, which reduce the memory usage and increase training speed.
        
    <br>


2. If only the transformer backbone should be frozen.

    ### Answer: 
    In this case, **only the two classification head will be trained**, while the transformer backbone will be frozen.
        

    #### Implication:
    This can be accelerated by first running the model on the training data to get the hidden states of the last layer of the model.
    Then the hidden states are used to train the classification heads. The model is trained by minimizing the negative log-likelihood
    of the desired output given the hidden states. The hidden states are used to classify the input sentence into the two tasks.
    To further improve the performance of the model, **we can also use adaptor, like Lora**, to add trainable adaptors to the multi-head attention layers (often q and v, according to the lora paper) of the transformer backbone. A common Lora configuration example:
    ```python
    from peft import LoraConfig
    peft_config = LoraConfig(
        lora_alpha=512,
        lora_dropout=0,
        r=4,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=["q_proj", "v_proj"], # in lora article, they say r=4, and apply to q_proj, v_proj have best performance
    )
    ```

    #### Scenario: 
    If not considering adaptor like Lora, The training is fast since we only need to feed the training data to the transformer model once to get the hidden states. Another advantage is that the model is less likely to overfit since the transformer backbone is frozen, and the model is only trained on the classification heads. when we have more data than can train the classification heads, and we want to prevent the model from overfitting. This choice can be used.

    If considering adaptor like Lora, we can also make the model adapt to the specific task better without changing too much on the model parameters, which might cause significant overfitting and lose significant features in the pretrained model.

    <br>

3. If only one of the task-specific heads (either for Task A or Task B) should be frozen.
    ### Answer: 
    #### Implication:
    We can let two classification heads share the same hidden states of the transformer backbone, and only freeze
    one of the classification heads. This case can be used when we want to utilize the shared information between the two tasks
    to improve the performance of the model. 

    #### Scenario:
    When the two tasks have shared information, and we want to utilize the shared information to improve the performance of the model. The model can learn the shared information between the two tasks, and use it to improve the performance of the model.

    <br>
    <br>

Consider a scenario where transfer learning can be beneficial. Explain how you would approach
the transfer learning process, including:
1. The choice of a pre-trained model.
    ### Answer: 
    Pre-trained model should be considered based on the balance of accuracy and computational cost. The pre-trained model should be
    trained on a large dataset, and should be fine-tuned on the target dataset to improve the performance of the model.

    If only performacne is considered, LLM model like DeepSeek-R1-Distill-Llama-8B model is a good choice, since it is trained on a large dataset

2. The layers you would freeze/unfreeze.
    ### Answer:
    The layers that should be frozen are the LLM backbone, and the layers that should be unfrozen are the classification heads. For Lora, we can add the Lora adaptor to the model, and train both the Lora adaptor and the classification heads.

3. The rationale behind these choices.
    ### Answer: 
    Has already justified in the above answers
'''