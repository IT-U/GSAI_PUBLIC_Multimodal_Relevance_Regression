# Multimodal Relevance Regression
This repository contains the reproduction code and data for the paper: 

- *"Towards continuous-valued relevance assessment of social media posts in disaster response: A decay-based regression approach"* by David Hanny, Andreas Kramer, Ehsaneddin Jajilian, Sebastian Schmidt, and Bernd Resch.

The work was submitted to the **AGILE Conference 2026**.

## 💾 Data
Our evaluation dataset was collected using the X (formerly Twitter) v1.1 filtered stream and recent search Application Programming Programming Interfaces (APIs). In accordance with X's [developer policies](https://developer.x.com/en/developer-terms/policy), it can only be shared as a list of post IDs.

We therefore provide post IDs, the corresponding relevance labels, and all derived features used for inference. Furthermore, we provide the softmax outputs of the fine-tuned TwHIN-BERT model used for text classification.

All data is available in the folder `./data`:
- `train_data_public.parquet`: Anonymised training data (3,659 posts)
- `oof_preds_text_train.npy`: Out-of-fold softmax predictions on the training data from a fine-tuned TwHIN-BERT text classifier for meta learning
- `test_data_public.parquet`: Anonymised test data (915 posts)
- `preds_text_test.npy`: Softmax predictions of a TwHIN-BERT text classifier, fine-tuned on the full training data

## 🐳 Reproduction
Our full experimental pipeline is available as a containerised Docker image.

To run the experiments, use:
```bash
docker compose run run-experiments   # run experiments as python script
docker compose run edit-experiments  # edit mode as a marimo notebook
```

All results will be stored to `./data/output`, and all figures will automatically go to `./data/figures`.

To run the visualisations, use:
```bash
docker compose run run-visualisations   # view mode as a marimo notebook
docker compose run edit-visualisations  # edit mode as a marimo notebook
```

## 📖 Citation
If you use this code or dataset in your research, please cite our work accordingly.

```
TBD
```

