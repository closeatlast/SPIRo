VIDEO_EMBEDDING_DIM = 512
MFCC_DIM = 39
VGGISH_DIM = 128
EGEMAPS_DIM = 23
BERT_DIM = 768
VIDEO_TEMPORAL_DIM = 128
MFCC_TEMPORAL_DIM = 32
VGGISH_TEMPORAL_DIM = 32
EGEMAPS_TEMPORAL_DIM = 32
BERT_TEMPORAL_DIM = 512

config = {
    # Frequency of each feature (frames/sec). If not relevant for your classification, set to None.
    "frequency": {
        "video": None,
        "mfcc": 100,
        "egemaps": 100,
        "vggish": None,
        # Remove or set to None any continuous or text features you are not using:
        "continuous_label": None,
        "bert": None
    },

    # How many times each feature is “expanded” or “multiplied” in data loading
    # For classification, typically keep them at 1.
    "multiplier": {
        "video": 1,
        "cnn": 1,
        "mfcc": 1,
        "egemaps": 1,
        "vggish": 1,
        "logmel": 1,
        # Remove or ignore any continuous or unused features:
        "continuous_label": 1,
        "AU_continuous_label": 1,
        "EXPR_continuous_label": 1,
        "VA_continuous_label": 1,
        "bert": 1,
    },

    # Dimensionality (shape) of each feature.  
    # Remove or ignore continuous_label references for binary classification.
    "feature_dimension": {
        "video": (48, 48, 3),
        "cnn": (512,),
        "mfcc": (39,),
        "egemaps": (88,),
        "vggish": (128,),
        "logmel": (96, 64),
        # Remove the ones you don't use:
        "continuous_label": (1,),
        "AU_continuous_label": (12,),
        "EXPR_continuous_label": (1,),
        "VA_continuous_label": (1,),
        "bert": (768,)
    },

    "tcn": {
        # Typically the embedding_dim for video features
        "embedding_dim": VIDEO_EMBEDDING_DIM,
        "channels": {
            'video': [VIDEO_EMBEDDING_DIM, VIDEO_EMBEDDING_DIM, VIDEO_EMBEDDING_DIM, VIDEO_EMBEDDING_DIM],
            'cnn_res50': [VIDEO_EMBEDDING_DIM, VIDEO_EMBEDDING_DIM, VIDEO_EMBEDDING_DIM, VIDEO_EMBEDDING_DIM],
            'mfcc': [VIDEO_TEMPORAL_DIM, VIDEO_TEMPORAL_DIM, VIDEO_TEMPORAL_DIM, VIDEO_TEMPORAL_DIM],
            'vggish': [VGGISH_DIM // 2, VGGISH_DIM // 2, VGGISH_DIM // 4, VGGISH_DIM // 4],
            'logmel': [VGGISH_DIM, VGGISH_DIM, VGGISH_DIM, VGGISH_DIM],
            'egemaps': [EGEMAPS_TEMPORAL_DIM, EGEMAPS_TEMPORAL_DIM, EGEMAPS_TEMPORAL_DIM, EGEMAPS_TEMPORAL_DIM],
            # Remove or ignore if you’re not using bert:
            'bert': [BERT_TEMPORAL_DIM, BERT_TEMPORAL_DIM, BERT_TEMPORAL_DIM, BERT_TEMPORAL_DIM]
        },
        "kernel_size": 5,
        "dropout": 0.1,
        "attention": 0,
    },

    "tcn_settings": {
        # TCN configs per modality
        "video": {
            "input_dim": 512,
            "channel": [256, 256, 128, 128, 128],
            "kernel_size": 5
        },
        "cnn": {
            "input_dim": 512,
            "channel": [256, 256, 128, 128],
            "kernel_size": 5
        },
        "cnn_res50": {
            "input_dim": 512,
            "channel": [256, 256, 128, 128],
            "kernel_size": 5
        },
        "vggish": {
            "input_dim": 128,
            "channel": [128, 128, 64, 64],
            "kernel_size": 5
        },
        "logmel": {
            "input_dim": 128,
            "channel": [128, 128, 64, 64, 64],
            "kernel_size": 5
        },
        "egemaps": {
            "input_dim": 88,
            "channel": [64, 64, 32, 32],
            "kernel_size": 5
        },
        "mfcc": {
            "input_dim": 39,
            "channel": [32, 32, 32, 32],
            "kernel_size": 5
        },
        "landmark": {
            "input_dim": 136,
            "channel": [64, 64, 32, 32],
            "kernel_size": 5
        },
        "bert": {
            "input_dim": 768,
            "channel": [256, 256, 128, 128],
            "kernel_size": 5
        }
    },

    "vae_settings": {
        "input_dim": 128
    },

    "attn_settings": {
        "input_dim": 128,
        "embedding_dim": 64,
        "num_head": 4
    },

    "backbone_settings": {
        "visual_state_dict": "res50_ir_0.887",
        "audio_state_dict": "vggish"
    },

    # Delay alignment if you had continuous labels. For binary classification, likely 0.
    "time_delay": 0,

    # Replace continuous metrics with classification metrics. For example:
    "metrics": ["accuracy"],
    # If you only want 'accuracy', do: "metrics": ["accuracy"]

    "save_plot": 0,

    "backbone": {
        "state_dict": "res50_ir_0.887",
        "mode": "ir",
    },
}
