local bert_model = "bert-base-uncased";

{
    "dataset_reader": {
        "lazy": false,
        "type": "bert_snli",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": bert_model,
            "do_lowercase": true
        },
        "token_indexers": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": bert_model,
            }
        }
    },
    "train_data_path": "drive/My Drive/text-augmentation/snli/snli_1.0_train.jsonl",
    "validation_data_path": "drive/My Drive/text-augmentation/snli/snli_1.0_dev.jsonl",
    "model": {
        "type": "bert_for_classification",
        "bert_model": bert_model,
        "dropout": 0.1,
        "num_labels": 3,
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["tokens", "num_tokens"]],
        "batch_size": 32
    },
    "trainer": {
        "optimizer": {
            "type": "bert_adam",
            "lr": 2e-5
        },
        "validation_metric": "+accuracy",
        "num_serialized_models_to_keep": 1,
        "num_epochs": 5,
        "grad_norm": 1.0,
        "cuda_device": 0
    }
}
