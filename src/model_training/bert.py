"""Utility functions for training a naive BERT model
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import Dataset
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_fscore_support, root_mean_squared_error, mean_absolute_error,
    roc_auc_score
)
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer,
                          TrainingArguments, EvalPrediction, pipeline)


def compute_metrics(prediction: EvalPrediction) -> dict:
    """Computes evaluation metrics for model predictions.

    Args:
        prediction (EvalPrediction): An object containing the model's predictions and the true labels.

    Returns:
        dict: A dictionary containing the accuracy, precision, recall, and F1-score of the predictions.
    """
    labels: np.ndarray = prediction.label_ids
    predictions: np.ndarray = prediction.predictions.argmax(-1)

    # Calculate accuracy
    accuracy: float = accuracy_score(labels, predictions)

    # Calculate precision, recall, and F1-score
    precision: float = precision_score(labels, predictions, average='macro')
    recall: float = recall_score(labels, predictions, average='macro')
    f1: float = f1_score(labels, predictions, average='macro')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train_classifier(texts_train: list[str], texts_val: list[str],
                     y_train: list, y_val: list,
                     model_name: str = 'Twitter/twhin-bert-base',
                     output_path: str = './training/results', logging_path: str = './training/logs',
                     model_path: str = './training/model', metric_for_best_model='f1',
                     learning_rate: float = 2e-5, batch_size: int = 32, epochs: int = 5,
                     weight_decay: float = 0.0, weighted_loss: bool = False,
                     id2label: dict = None, label2id: dict = None) -> tuple:
    """Trains a text classifier with fixed parameters.

    Args:
        texts_train (list[str]): List of input texts for training.
        texts_val (list[str]): List of input texts for validation.
        y_train (list): Labels for training data.
        y_val (list): Labels for validation data.
        model_name (str, optional): Pre-trained model to be used. Defaults to 'Twitter/twhin-bert-base'.
        output_path (str, optional): Directory to save training results. Defaults to './training/results'.
        logging_path (str, optional): Directory to save training logs. Defaults to './training/logs'.
        model_path (str, optional): Directory to save the trained model. Defaults to './training/model'.
        metric_for_best_model (str, optional): The metric used to identify the best model. Defaults to 'f1'.
        learning_rate (float, optional): Learning rate for training. Defaults to 2e-5.
        batch_size (int, optional): Batch size for training. Defaults to 32.
        epochs (int, optional): Number of training epochs. Defaults to 5.
        weight_decay (int, optional): Weight decay hyperparameter. Defaults to 0.
        weighted_loss (bool, optional): Whether to use a class-weighted loss function. Defaults to False.
        id2label (bool, optional): The id2label dict for the config.json. Defaults to None.
        label2id (bool, optional): The label2id dict for the config.json. Defaults to None.

    Returns:
        tuple: A tuple containing the trained model, tokenizer and evaluation results.
    """
    train_data: pd.DataFrame = pd.DataFrame({'text': texts_train, 'label': y_train})
    val_data: pd.DataFrame = pd.DataFrame({'text': texts_val, 'label': y_val})

    # Compute class weights from training data using the formula:
    # weight = total_samples / (num_classes * count for class)
    if weighted_loss:
        label_counts = train_data['label'].value_counts(sort=False)
        num_classes = len(label_counts)
        total_samples = len(train_data)
        print(label_counts)
        weights = {label: total_samples / (num_classes * count) for label, count in label_counts.items()}
        sorted_labels = sorted(label_counts.index)
        class_weights = torch.tensor([weights[label] for label in sorted_labels], dtype=torch.float)
        print(sorted_labels)
        print("Computed class weights:", class_weights)
    else:
        print("Training model without weighted loss function.")

    # Define a custom Trainer that uses weighted loss if needed
    if weighted_loss:
        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
                loss = loss_fct(logits, labels)
                return (loss, outputs) if return_outputs else loss
        TrainerClass = WeightedTrainer
    else:
        TrainerClass = Trainer

    # Convert pandas DataFrames to HuggingFace Datasets
    train_dataset: Dataset = Dataset.from_pandas(train_data)
    val_dataset: Dataset = Dataset.from_pandas(val_data)

    # Load tokenizer
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize datasets
    train_dataset = train_dataset.map(lambda examples: tokenizer(examples['text'], padding='max_length',
                                                                 truncation=True, max_length=512),
                                      batched=True)
    val_dataset = val_dataset.map(lambda examples: tokenizer(examples['text'], padding='max_length',
                                                             truncation=True, max_length=512),
                                  batched=True)

    # Set the format of datasets for PyTorch
    # input_ids = tokenised text, attention mask = 0/1 whether it is a real token or padding
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Load model with specified number of labels
    num_labels: int = len(train_data['label'].unique())
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # Create folders for results and logging
    Path(output_path).mkdir(parents=True, exist_ok=True)
    Path(logging_path).mkdir(parents=True, exist_ok=True)

    # Train with defined hyperparameters
    training_args = TrainingArguments(
        output_dir=output_path,
        evaluation_strategy='epoch',
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        save_strategy='epoch',
        logging_dir=logging_path,
        load_best_model_at_end=True,
        save_total_limit=1,
        report_to='none',
        metric_for_best_model=f'eval_{metric_for_best_model}',
    )

    trainer = TrainerClass(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
    )
    trainer.train()

    if id2label:
        model.config.id2label = id2label
    if label2id:
        model.config.label2id = label2id

    # Save the model and tokenizer
    if model_path:
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

    # Evaluate the model
    eval_results = trainer.evaluate()
    print("Evaluation Results:", eval_results)

    return model, tokenizer, eval_results


def train_classifier_with_hp(texts_train: list[str], texts_val: list[str],
                             y_train: list, y_val: list,
                             model_name: str = 'Twitter/twhin-bert-base',
                             output_path: str = './training/results', logging_path: str = './training/logs',
                             model_path: str = './training/model', metric_for_best_model: str = 'f1',
                             weighted_loss: bool = False,
                             id2label: dict = None, label2id: dict = None,
                             n_trials: int = 20) -> tuple:
    """Trains a text classifier with hyperparameter optimisation using Optuna.

    This function performs hyperparameter search over key training parameters and then retrains
    the model with the best configuration.

    Args:
        texts_train (list[str]): List of training texts.
        texts_val (list[str]): List of validation texts.
        y_train (list): List of training labels.
        y_val (list): List of validation labels.
        model_name (str, optional): Pre-trained model identifier. Defaults to 'Twitter/twhin-bert-base'.
        output_path (str, optional): Directory to save training results. Defaults to './training/results'.
        logging_path (str, optional): Directory to save training logs. Defaults to './training/logs'.
        model_path (str, optional): Directory to save the trained model. Defaults to './training/model'.
        metric_for_best_model (str, optional): The metric to use for selecting the best model. Defaults to 'f1'.
        weighted_loss (bool, optional): Whether to use a class-weighted loss function. Defaults to False.
        id2label (dict, optional): id2label mapping for the model config. Defaults to None.
        label2id (dict, optional): label2id mapping for the model config. Defaults to None.
        n_trials (int, optional): Number of Optuna trials to run. Defaults to 20.

    Returns:
        tuple: (best_model, tokenizer, best_hyperparameters, eval_results)
    """
    # Prepare dataframes
    train_data = pd.DataFrame({'text': texts_train, 'label': y_train})
    val_data = pd.DataFrame({'text': texts_val, 'label': y_val})

    # If weighted loss is desired, compute class weights
    if weighted_loss:
        label_counts = train_data['label'].value_counts(sort=False)
        num_classes = len(label_counts)
        total_samples = len(train_data)
        weights = {label: total_samples / (num_classes * count) for label, count in label_counts.items()}
        sorted_labels = sorted(label_counts.index)
        class_weights = torch.tensor([weights[label] for label in sorted_labels], dtype=torch.float)
        print("Computed class weights:", class_weights)
    else:
        print("Training model without weighted loss function.")

    # Define a custom Trainer that uses weighted loss if needed
    if weighted_loss:
        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
                loss = loss_fct(logits, labels)
                return (loss, outputs) if return_outputs else loss
        TrainerClass = WeightedTrainer
    else:
        TrainerClass = Trainer

    # Convert DataFrames to HuggingFace Datasets
    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(val_data)

    # Load tokenizer and tokenize datasets
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenize_fn = lambda examples: tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)  # noqa
    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    val_dataset = val_dataset.map(tokenize_fn, batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Determine the number of labels from the training data
    num_labels = len(train_data['label'].unique())

    # Create a model initialization function for Trainer (ensuring a fresh model for each trial)
    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # Set up initial TrainingArguments (placeholders; hyperparameters will be overridden)
    training_args = TrainingArguments(
        output_dir=output_path,
        evaluation_strategy='epoch',
        learning_rate=2e-5,                   # placeholder; to be tuned
        per_device_train_batch_size=32,       # placeholder; to be tuned
        num_train_epochs=5,                   # placeholder; to be tuned
        weight_decay=0.0,                     # placeholder; to be tuned
        save_strategy='epoch',
        logging_dir=logging_path,
        load_best_model_at_end=True,
        save_total_limit=1,
        report_to='none',
        metric_for_best_model=f'eval_{metric_for_best_model}',
    )

    # Instantiate a Trainer for hyperparameter search
    trainer_hp = TrainerClass(
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

    # Update training arguments with the best hyperparameters found
    training_args.learning_rate = best_run.hyperparameters["learning_rate"]
    training_args.per_device_train_batch_size = best_run.hyperparameters["per_device_train_batch_size"]
    training_args.num_train_epochs = best_run.hyperparameters["num_train_epochs"]
    training_args.weight_decay = best_run.hyperparameters["weight_decay"]

    # Create folders for saving the model and logs
    Path(output_path).mkdir(parents=True, exist_ok=True)
    Path(logging_path).mkdir(parents=True, exist_ok=True)

    # Reinitialize the trainer with the best hyperparameters
    trainer = TrainerClass(
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


def evaluate_text_classification_pipeline(
        test_df: pd.DataFrame, model_path: str, label_mapping: dict, device: int = -1,
        text_column: str = 'text', label_column: str = 'int_label',
        prediction_path: str | None = None) -> tuple:
    """
    Evaluate a text classification pipeline using a fine-tuned model.

    Args:
        test_df (pd.DataFrame): DataFrame containing test data with texts and true labels.
        model_path (str): Path to the folder containing the fine-tuned model.
        label_mapping (dict): Mapping from string label (used in the pipeline) to integer label.
        device (int, optional): Device index to run the model (e.g., 0 for GPU, -1 for CPU). Defaults to -1.
        text_column (str, optional): Column name for input texts. Defaults to 'text'.
        label_column (str, optional): Column name for true labels. Defaults to 'int_label'.

    Returns:
        dict: A tuple with two elements:
         (1) dictionary containing evaluation metrics (macro precision, recall, F1, accuracy, ROC-AUC, RMSE, MAE).
         (2) the dataframe with prediction
    """
    # Create the text classification pipeline with all scores returned
    classifier = pipeline("text-classification",
                          model=model_path,
                          device=device,
                          return_all_scores=True)

    pred_labels = []
    pred_probs = []  # List to hold probability vectors for each text

    # Process each text and obtain predictions and scores
    for text in tqdm(test_df[text_column], desc="Processing texts"):
        # classifier returns a list (for each input) containing a list of dicts (one per class)
        scores = classifier(text)[0]
        # Sort the scores by label to ensure a consistent order using label_mapping
        scores = sorted(scores, key=lambda x: label_mapping[x['label']])
        probs = [s['score'] for s in scores]
        pred_label = int(label_mapping[scores[np.argmax(probs)]['label']])
        pred_labels.append(pred_label)
        pred_probs.append(probs)

    # Optionally, store predictions in the DataFrame
    test_df['prediction'] = pred_labels

    # Compute evaluation metrics
    prec, rec, f1, _ = precision_recall_fscore_support(test_df[label_column], pred_labels, average='macro')
    acc = accuracy_score(test_df[label_column], pred_labels)
    roc_auc = roc_auc_score(test_df[label_column], np.array(pred_probs), multi_class='ovr', average='macro')
    rmse = root_mean_squared_error(test_df[label_column], pred_labels)
    mae = mean_absolute_error(test_df[label_column], pred_labels)

    # Print evaluation metrics
    print("Evaluation Results:")
    print(f"- Macro Precision: {prec}")
    print(f"- Macro Recall: {rec}")
    print(f"- Macro F1: {f1}")
    print(f"- Accuracy: {acc}")
    print(f"- Macro ROC-AUC: {roc_auc}")
    print(f"- RMSE: {rmse}")
    print(f"- MAE: {mae}")

    # Return metrics in a dictionary
    metrics = {
        'macro_precision': prec,
        'macro_recall': rec,
        'macro_f1': f1,
        'accuracy': acc,
        'macro_roc_auc': roc_auc,
        'rmse': rmse,
        'mae': mae
    }

    return metrics, test_df


# Define a function to extract probabilities for each class
def extract_probabilities(text: str, classifier: pipeline) -> dict:
    """Extract probabilities for each class given an instanciated classifier.

    Args:
        text (str): The text to classifier.
        classifier (pipeline): The instanciated classified.

    Returns:
        pd.Series: a dictionary containing the probabilities for each class.
    """
    # The classifier returns a list of lists of dictionaries. For a single input, use result[0]
    result = classifier(text)[0]
    # Build a dictionary mapping each label to its score
    probs = {item['label']: item['score'] for item in result}
    # Return a Series with probabilities for each class in the desired order.
    return {
        'p_not_related': probs.get('Not related', 0.0),
        'p_related_but_not_relevant': probs.get('Related but not relevant', 0.0),
        'p_related_and_relevant': probs.get('Related and relevant', 0.0),
    }