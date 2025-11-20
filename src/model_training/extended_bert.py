"""Utility functions for model training (incl. hyperparameter optimisation)
"""
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, List, Tuple
from datasets import Dataset
from tqdm import tqdm
from transformers import (AutoTokenizer, Trainer,
                          TrainingArguments, BertForSequenceClassification,
                          BertConfig)
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from src.model_training.bert import compute_metrics
# suggest model: https://huggingface.co/Twitter/twhin-bert-base


class ExtendedNumBertForSequenceClassification(BertForSequenceClassification):
    """Custom BERT model for sequence classification with additional numerical features.
    Sources:
    - https://stackoverflow.com/questions/73567055/extend-bert-or-any-transformer-model-using-manual-features
    - https://www.kaggle.com/code/gamazic/transformer-with-nontext-features
    """

    def __init__(self, config: BertConfig, num_numerical_features: int, class_weights: torch.Tensor | None = None,
                 classification_head: str = 'simple') -> None:
        """
        Initializes the ExtendedBertForSequenceClassification model.

        Args:
            config (BertConfig): Configuration for the BERT model.
            num_numerical_features (int): The number of numerical features to be included in the model.
            class_weights (torch.Tensor | None, optional): Weights for each class in the classification task,
                ordered by their integer label. Defaults to None.
            classification_head (str, optional): Type of classification head. Defaults to 'simple'.
        """
        super().__init__(config)

        # deactivated embeddings(!)
        # Define an embedding layer for numerical features - this is essentially just a linear layer
        # self.num_feature_embeddings = nn.Embedding(num_numerical_features, embedding_dim)

        # Update the classification head to include the additional input features
        self.new_hidden_size = config.hidden_size + num_numerical_features
        self.classification_head = classification_head

        if self.classification_head == 'complex':
            # Process non-text features separately
            self.non_text_processor = torch.nn.Sequential(
                torch.nn.Linear(num_numerical_features, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 32),
                torch.nn.ReLU()
            )
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size + 32, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(256, config.num_labels)
            )
        else:
            self.classifier = torch.nn.Linear(self.new_hidden_size, config.num_labels)

        # Define learnable scalar weights for each feature type
        if num_numerical_features > 0:
            self.numeric_feature_weights = torch.nn.Parameter(torch.ones(num_numerical_features))

        # Store class weights for the CrossEntropy loss
        if class_weights is not None:
            self.class_weights = class_weights
        else:
            self.class_weights = torch.ones(config.num_labels)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        numerical_features: Optional[torch.Tensor] = None,
    ):
        """Forward pass for the model.

        Args:
            input_ids (Optional[torch.Tensor], optional): Input tensor containing token IDs. Defaults to None.
            attention_mask (Optional[torch.Tensor], optional): Mask to avoid performing attention on padding token indices.
                Defaults to None.
            token_type_ids (Optional[torch.Tensor], optional): Segment token indices to indicate different portions of the inputs.
                Defaults to None.
            position_ids (Optional[torch.Tensor], optional): Indices of positions of each input sequence tokens in the
                position embeddings. Defaults to None.
            head_mask (Optional[torch.Tensor], optional): Mask to nullify selected heads of the self-attention modules.
                Defaults to None.
            inputs_embeds (Optional[torch.Tensor], optional): Optionally, instead of passing input_ids you can choose to
                directly pass an embedded representation. Defaults to None.
            labels (Optional[torch.Tensor], optional): Labels for computing the loss. Defaults to None.
            output_attentions (Optional[bool], optional): Whether to return the attentions tensors of all attention layers.
                Defaults to None.
            output_hidden_states (Optional[bool], optional): Whether to return the hidden states of all layers. Defaults to None.
            return_dict (Optional[bool], optional): Whether to return a dict instead of a plain tuple. Defaults to None.
            numerical_features (Optional[torch.Tensor], optional): Additional numerical features to be concatenated with the
                pooled output. Defaults to None.

        Returns:
            SequenceClassifierOutput: A dataclass with the following attributes:
                - loss (torch.Tensor, optional): Classification (or regression) loss.
                - logits (torch.Tensor): Classification (or regression) scores.
                - hidden_states (tuple(torch.FloatTensor), optional): Hidden-states of the model at the output of each
                    layer plus the initial embedding outputs.
                - attentions (tuple(torch.FloatTensor), optional): Attentions weights after the attention softmax, used to
                    compute the weighted average in the self-attention heads.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass input through BERT
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output: torch.Tensor = outputs[1]
        # // pooled_output = self.dropout(pooled_output)

        # Handle numerical features
        if numerical_features is not None:
            if self.classification_head == 'complex':
                weighted_features: torch.Tensor = numerical_features * self.numeric_feature_weights
                processed_non_text: torch.Tensor = self.non_text_processor(weighted_features)  # shape: (batch_size, 32)
                combined_output: torch.Tensor = torch.cat([pooled_output, processed_non_text], dim=1)
            else:
                combined_output: torch.Tensor = torch.cat([pooled_output, numerical_features * self.numeric_feature_weights],
                                                          dim=1)
        else:
            combined_output = pooled_output
        # combined_output = self.dropout(combined_output)

        # Pass the combined output through the classification head
        logits: torch.Tensor = self.classifier(combined_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss(weight=self.class_weights.to(logits.device))
                # loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class WeightedTrainer(Trainer):
    """Custom trainer with weighted loss function.
    """
    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def train_classifier_w_numerical_features_with_hp(
    texts_train: list[str],
    texts_val: list[str],
    y_train: list,
    y_val: list,
    numerical_features_train: dict[str, list[float]],
    numerical_features_val: dict[str, list[float]],
    model_name: str = 'Twitter/twhin-bert-base',
    output_path: str = './training/results',
    logging_path: str = './training/logs',
    model_path: str = './training/model',
    device: str = 'cuda',
    class_weights: np.ndarray | None = None,
    metric_for_best_model: str = 'f1',
    id2label: dict = None,
    label2id: dict = None,
    classification_head: str = 'simple',
    n_trials: int = 20
) -> tuple:
    """
    Train a classifier with text and optional numerical features using hyperparameter optimization.

    The function splits the data, tokenizes text (and numerical features), and performs hyperparameter search
    over key training parameters. The best configuration is then used to retrain the model.

    Args:
        texts_train (list[str]): List of input texts for training.
        texts_val (list[str]): List of input texts for validation.
        y_train (list): Labels for training data.
        y_val (list): Labels for validation data.
        numerical_features_train (dict[str, list[float]], optional): Dictionary of numerical
            features for trraining. Defaults to None.
        numerical_features_val (dict[str, list[float]], optional): Dictionary of numerical features for
            testing. Defaults to None.
        model_name (str, optional): Pre-trained model identifier. Defaults to 'Twitter/twhin-bert-base'.
        output_path (str, optional): Directory to save training results. Defaults to './training/results'.
        logging_path (str, optional): Directory to save training logs. Defaults to './training/logs'.
        model_path (str, optional): Directory to save the trained model. Defaults to './training/model'.
        device (str, optional): Device to use for training. Defaults to 'cuda'.
        class_weights (np.ndarray | None): Weights for each class (if applicable). Defaults to None.
        metric_for_best_model (str, optional): Metric to identify the best model. Defaults to 'f1'.
        id2label (dict, optional): id2label mapping for the model config. Defaults to None.
        label2id (dict, optional): label2id mapping for the model config. Defaults to None.
        classification_head (str, optional): Type of classification head. Defaults to 'simple'.
        n_trials (int, optional): Number of Optuna trials to run. Defaults to 20.

    Returns:
        tuple: (best_model, tokenizer, best_hyperparameters, eval_results)
    """
    # Prepare DataFrame with texts, labels, and (if provided) numerical features
    train_data = pd.DataFrame({'text': texts_train, 'label': y_train})
    val_data = pd.DataFrame({'text': texts_val, 'label': y_val})
    
    # Add numerical features
    if numerical_features_train is not None and numerical_features_val is not None:
            for feature_name, feature_values in numerical_features_train.items():
                train_data[feature_name] = feature_values
            for feature_name, feature_values in numerical_features_val.items():
                val_data[feature_name] = feature_values
    
                
    # Convert DataFrames to HuggingFace Datasets
    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(val_data)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define a tokenization function that also adds numerical features if available.
    def tokenize(examples: dict, numerical_features: dict) -> dict:
        tokenized = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
        if numerical_features is not None:
            # For each feature, get the list of values from the examples dictionary.
            num_feats = [examples[feat] for feat in numerical_features.keys()]
            # Transpose to get shape (batch_size, num_numerical_features)
            tokenized['numerical_features'] = torch.tensor(np.array(num_feats).T, dtype=torch.float)
        return tokenized

    train_dataset = train_dataset.map(lambda examples: tokenize(examples, numerical_features_train), batched=True)
    val_dataset = val_dataset.map(lambda examples: tokenize(examples, numerical_features_val), batched=True)

    # Set dataset format for PyTorch
    if numerical_features_train is not None:
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label', 'numerical_features'])
        val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label', 'numerical_features'])
    else:
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Determine number of labels and number of numerical features
    num_labels = len(train_data['label'].unique())
    num_numerical_features = len(numerical_features_train) if numerical_features_train is not None else 0

    # Define a model initialization function (ensuring a fresh model for each trial)
    def model_init():
        return ExtendedNumBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            num_numerical_features=num_numerical_features,
            classification_head=classification_head
        )

    # Set up initial TrainingArguments with placeholder hyperparameters (they will be tuned)
    training_args = TrainingArguments(
        output_dir=output_path,
        evaluation_strategy='epoch',
        learning_rate=2e-5,                         # placeholder
        per_device_train_batch_size=32,             # placeholder
        num_train_epochs=5,                         # placeholder
        weight_decay=0.0,                           # placeholder, to be tuned
        save_strategy='epoch',
        logging_dir=logging_path,
        load_best_model_at_end=True,
        save_total_limit=1,
        report_to='none',
        metric_for_best_model=f'eval_{metric_for_best_model}',
    )

    # Instantiate a Trainer for hyperparameter search
    trainer_hp = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Define the hyperparameter search space using Optuna's API
    def hp_space(trial):
        return {
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 5e-5),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32]),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 10),
            "weight_decay": trial.suggest_loguniform("weight_decay", 1e-4, 1e-1)
        }

    # Run hyperparameter search
    best_run = trainer_hp.hyperparameter_search(
        hp_space=hp_space,
        direction="maximize",  # adjust if a lower metric is better
        n_trials=n_trials,
        backend="optuna"
    )
    print("Best hyperparameters:", best_run.hyperparameters)

    # Update training arguments with the best hyperparameters
    training_args.learning_rate = best_run.hyperparameters["learning_rate"]
    training_args.per_device_train_batch_size = best_run.hyperparameters["per_device_train_batch_size"]
    training_args.num_train_epochs = best_run.hyperparameters["num_train_epochs"]
    training_args.weight_decay = best_run.hyperparameters["weight_decay"]

    # Create folders for saving the model and logs
    Path(output_path).mkdir(parents=True, exist_ok=True)
    Path(logging_path).mkdir(parents=True, exist_ok=True)

    # Reinitialize the trainer with the best hyperparameters
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train the model with the best hyperparameters
    trainer.train()

    # Optionally update the model configuration with id2label and label2id if provided
    best_model = trainer.model
    if id2label:
        best_model.config.id2label = id2label
    if label2id:
        best_model.config.label2id = label2id

    # Save the trained model and tokenizer
    if model_path:
        Path(model_path).mkdir(parents=True, exist_ok=True)
        best_model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

    # Evaluate the model on the validation set
    eval_results = trainer.evaluate()
    print("Evaluation Results:", eval_results)

    return best_model, tokenizer, best_run.hyperparameters, eval_results


def load_and_infer_batch(model_path: str, texts: List[str], numerical_features: List[List[float]], device: str = 'cuda',
                         batch_size: int = 16) -> Tuple[List[int], List[int]]:
    """
    Loads a fine-tuned ExtendedNumBertForSequenceClassification model and tokenizer,
    then performs batch inference on a list of text samples with associated numerical features.

    Args:
        model_path (str): Path to the saved model directory.
        texts (List[str]): List of input text samples.
        numerical_features (List[List[float]]): Nested list of numerical features, one sublist per sample.
        device (str, optional): Device to run inference on. Defaults to 'cuda'.
        batch_size (int, optional): Batch size for inference. Defaults to 16.

    Returns:
        Tuple[List[int], np.ndarray]: List of predicted labels and the softmax probabilities.
    """
    # Load tokenizer and model (ensure ExtendedNumBertForSequenceClassification is imported or defined)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = ExtendedNumBertForSequenceClassification.from_pretrained(
        model_path, num_numerical_features=len(numerical_features[0])
    )
    model.to(device)

    # Total number of samples
    total_samples = len(texts)
    num_batches = np.ceil(total_samples / batch_size)

    all_predictions = []
    all_probs = []

    # Process in batches with a progress bar
    for i in tqdm(range(0, total_samples, batch_size), desc="Inference", total=num_batches):
        batch_texts = texts[i: i + batch_size]
        batch_numerical = numerical_features[i: i + batch_size]

        # Tokenize the batch
        inputs = tokenizer(batch_texts, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Convert numerical features to a tensor and add to inputs
        numerical_tensor = torch.tensor(batch_numerical, dtype=torch.float).to(device)
        inputs['numerical_features'] = numerical_tensor

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        softmax = [F.softmax(logit, dim=-1) for logit in logits]
        batch_pred_int = torch.argmax(logits, dim=-1).tolist()

        # Map integer predictions to labels using the model's id2label dictionary
        id2label = (model.config.id2label if hasattr(model.config, 'id2label')
                    else {i: str(i) for i in range(model.config.num_labels)})
        batch_pred_labels = [id2label[label] for label in batch_pred_int]  # noqa

        all_predictions.extend(batch_pred_int)  # todo: fix this some point later
        all_probs.extend(softmax)

    return all_predictions, torch.stack(all_probs).cpu().numpy()
