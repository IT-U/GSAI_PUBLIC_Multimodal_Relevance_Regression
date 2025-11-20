import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

def get_cls_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

def generate_embeddings(df, text_column, tokenizer, model, batch_size=16):
    embeddings = []
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size][text_column].tolist()
        batch_embeddings = []
        for text in batch:
            emb = get_cls_embedding(text, tokenizer, model)
            batch_embeddings.append(emb)
        embeddings.extend(batch_embeddings)
    return np.stack(embeddings)

def save_embeddings(embeddings, path):
    np.save(path, embeddings)
