import tensorflow


# Whether GPT2 decoder is trainable
is_gpt2_trainable = True

# Specify sample strategy
sample_strategy = "train_greedy"
# sample_strategy = "beam_search"
beam_width = 4

# Overarching
gen_clas_test = False # Unit test of acc and perp
gen_test = False # Whether training or testing the generator perplexity
gen_perp_output = "/tmp/perplexities.txt" # Where to save generator test perplexities
clas_test = False # Whether training or testing the classifier test performance
clas_test_ckpt = "/tmp/ckpt-bestclas" # Which checkpoint to use for classifier testing
clas_pred_output = "/tmp/testpreds.txt" # Where to save classifier test predictions

# Saving/logging Config
restore_model= False # Whether to reinitialize or restore weights
clear_run_logs = False # Whether to delete prior run logs
log_dir= '/tmp/' # Where to store logs
checkpoint_dir= '/tmp' # Where to store ckpt files
load_checkpoint_file = None # Which checkpoint to load

# Logging frequency/verbosity
log_verbose_mle = True
log_verbose_rl = True
batches_per_summary = 100
batches_per_text_summary = 100

# Number of epochs to run
# g_unlab_every_n = 20 # Balance of exposure to labeled/unlabeled datasets
g_unlab_every_n = 5 # Balance of exposure to labeled/unlabeled datasets
# g_pretrain_epochs = 40 # 60
g_pretrain_epochs = 1 # 60
# d_pretrain_epochs = 5 # 60
d_pretrain_epochs = 1 # 60
# d_pretrain_critic_epochs = 10 #20
d_pretrain_critic_epochs = 0 #20
# c_pretrain_epochs = 20 # 20
c_pretrain_epochs = 1 # 20
# adversarial_epochs = 10 # How many total adversarial epochs, for all components
adversarial_epochs = 1 # How many total adversarial epochs, for all components
 
# During adversarial training, how many epochs to run for discriminator/classifier
disc_adv = 1
# clas_adv = 10 
clas_adv = 5

# gen_adv_epoch = 4 # Number of generator adversarial epochs
gen_adv_epoch = 4 # Number of generator adversarial epochs
g_unlab_every_n_adv = -1 # Frequency of generator ML epochs with unlabeled data
gen_mle_adv_epoch = 2 # Number of generator ML epochs with labeled data

adv_train_max_gen_examples = 1000 # Maximum size of training epoch for gen in adv
adv_disc_max_ex = 5000 # Maximum size of training epoch for disc in adv
adv_gen_train_with_unsup = False # Whether or not to use unlabeled examples in adv

# Early stopping parameters
gen_patience=20
gen_es_tolerance = 0.005
clas_es_tolerance = 0.005
clas_patience = 10

# Controls ML/generation max sentence length (in words)
max_decoding_length = 128
max_decoding_length_infer = 128
annealing_length = 128 # Can use shorter sentences for initial training
adversarial_length = 128 # Can use shorter sentences for adversarial training

sampling_temperature = 1.0 # Sampling temperature in generation


linear_decay_pg_weights = True # Place more importance on initial sentence rewards

# Context configs
prior_prob=0.5 # probability of class 1 in generated/unlabeled data.
# noise_size=10 # dim of noise vector
noise_size = 128 # dim of noise vector
class_size = 24 # dim of real class inside noise vector


# Training tweaks
disc_label_smoothing_epsilon = 0.05 # label smoothing for discriminator

# Set to experiement with clipping policy gradients at various points
adv_max_clip = 100
min_log_prob = 0.1
max_log_prob = 100
min_pg_loss = -200
max_pg_loss = 200


add_sentence_progress = True # Includes indicator of sentence length (decays 1 to 0)
# add_sentence_progress = False # avoids mode collapse to extremely short sentences

clas_loss_on_fake_lambda = 1.0 # Balancing param on real/generated clas
disc_crit_train_on_fake_only = True # Only train disc crit on generated sentences
clas_crit_train_on_fake_only = True # Only train disc crit on generated sentences

reward_blending = 'f1' # Additive vs f1 clas-disc reward blending

clas_min_ent_lambda = 1.0 # Controls strength of entropy minimization

clas_has_own_embedder = True # Pool or share embedders
disc_has_own_embedder = True
# clas_has_own_embedder = False # Pool or share embedders
# disc_has_own_embedder = False

# Different loss functions
mle_loss_in_adv = True # Whether or not to include ML optimization in adversarial 

# Relative weighting of discriminator and classifier in pg loss
discriminator_loss_lambda = 1.0 
classifier_loss_lambda = 1.0

norm_advantages = True # Normalize advantages
#let_discriminator_train_embedder = True # whether discriminator can update embedder
let_discriminator_train_embedder = True


train_data = {
    "num_epochs": 1,
#     "batch_size": 128,
    "batch_size": 26,
    "allow_smaller_final_batch": True,
    "shuffle": True,
    "shuffle_buffer_size": None,
    "shard_and_shuffle": False,

    "num_parallel_calls": 1,
    "prefetch_buffer_size": 0,
    "max_dataset_size": -1,
    "seed": None,
    "name": "train_data",
    'datasets' : [ 
        {
#             "files" : ['./minrun_train_reviews.txt'],
#             'vocab_file' : './minrun_opspam_vocab.txt',
            "files" : ['./data/yelp/unlabel50_label10/train_review.txt'],
            'vocab_file' : './data/yelp/vocab.txt',
            'max_seq_length' : 128,
            'length_filter_mode' : 'truncate',
            'bos_token' : '<BOS>',
            'delimiter' : ' ',
            'eos_token' : '<EOS>',
            'data_name' : 'x',
            'pad_to_max_seq_length' : True
        },
        {
#             'files' : ['./minrun_train_labels.txt'],
            'files' : ['./data/yelp/unlabel50_label10/train_label.txt'],
            'data_type' : 'int',
            'data_name' : 'label'
        }
    ]
}

clas_train_data = {
    "num_epochs": 1,
#     "batch_size": 128,
    "batch_size": 26,
    "allow_smaller_final_batch": True,
    "shuffle": True,
    "shuffle_buffer_size": None,
    "shard_and_shuffle": False,

    "num_parallel_calls": 1,
    "prefetch_buffer_size": 0,
    "max_dataset_size": -1,
    "seed": None,
    "name": "train_data",
    'datasets' : [ 
        {
#             "files" : ['./minrun_train_reviews.txt'],
#             'vocab_file' : './minrun_opspam_vocab.txt',
            "files" : ['./data/yelp/unlabel50_label10/train_review.txt'],
            'vocab_file' : './data/yelp/vocab.txt',
            'max_seq_length' : 128,
            'length_filter_mode' : 'truncate',
            'bos_token' : '<BOS>',
            'delimiter' : ' ',
            'eos_token' : '<EOS>',
            'data_name' : 'x',
            'pad_to_max_seq_length' : True
        },
        {
#             'files' : ['./minrun_train_labels.txt'],
            'files' : ['./data/yelp/unlabel50_label10/train_label.txt'],
            'data_type' : 'int',
            'data_name' : 'label'
        }
    ]
}



val_data = {
    "num_epochs": 1,
#     "batch_size": 50,
    "batch_size": 16,
    "allow_smaller_final_batch": True,
    "shuffle": True,
    "shuffle_buffer_size": None,
    "shard_and_shuffle": False,

    "num_parallel_calls": 1,
    "prefetch_buffer_size": 0,
    "max_dataset_size": -1,
    "seed": None,
    "name": "val_data",

    'datasets' : [ 
        {
#             "files" : ['./minrun_val_reviews.txt'],
#             'vocab_file' : './minrun_opspam_vocab.txt',
            "files" : ['./data/yelp/unlabel50_label10/val_review.txt'],
            'vocab_file' : './data/yelp/vocab.txt',
            'max_seq_length' : 128,
            'length_filter_mode' : 'truncate',
            'bos_token' : '<BOS>',
            'delimiter' : ' ',
            'eos_token' : '<EOS>',
            'data_name' : 'x',
            'pad_to_max_seq_length' : True
        },
        {
#             'files' : ['./minrun_val_labels.txt'],
            'files' : ['./data/yelp/unlabel50_label10/val_label.txt'],
            'data_type' : 'int',
            'data_name' : 'label'
        }
    ]
}

test_data = { 
    "num_epochs": 1,
#     "batch_size": 64,
    "batch_size": 16,
    "allow_smaller_final_batch": True,
    "shuffle": False,
    "shuffle_buffer_size": None,
    "shard_and_shuffle": False,
    "num_parallel_calls": 1,
    "prefetch_buffer_size": 0,
    "max_dataset_size": -1,
    "seed": None,
    "name": "test_data",
    'datasets' : [ 
        {
#             "files" : ['minrun_test_reviews.txt'],
#             'vocab_file' : 'minrun_opspam_vocab.txt',
            "files" : ['./data/yelp/test_review.txt'],
            'vocab_file' : './data/yelp/vocab.txt',
            'max_seq_length' : 128,
            'length_filter_mode' : 'truncate',
            'bos_token' : '<BOS>',
            'delimiter' : ' ',
            'eos_token' : '<EOS>',
            'data_name' : 'x',
            'pad_to_max_seq_length' : True
        },
        {
#             'files' : ['minrun_test_labels.txt'],
            'files' : ['./data/yelp/test_label.txt'],
            'data_type' : 'int',
            'data_name' : 'label'
        }
    ]
}



# EMBEDDER HPARAMS

bert_emb_hparams = {
    "pretrained_model_name": "bert-base-uncased",
    "embed": {
        "dim": 768,
        "name": "word_embeddings"
    },
    "vocab_size": 30522,
    "segment_embed": {
        "dim": 768,
        "name": "token_type_embeddings"
    },
    "type_vocab_size": 2,
    "position_embed": {
        "dim": 768,
        "name": "position_embeddings"
    },
    "position_size": 512,

    "encoder": {
        "dim": 768, 
        "embedding_dropout": 0.1,
        "multihead_attention": {
            "dropout_rate": 0.1,
            "name": "self",
            "num_heads": 12,
            "num_units": 768,
            "output_dim": 768,
            "use_bias": True
        },
        "name": "encoder",
        "num_blocks": 12,
        "poswise_feedforward": {
            "layers": [
                {   "kwargs": {
                        "activation": "gelu",
                        "name": "intermediate",
                        "units": 3072,
                        "use_bias": True
                    },
                    "type": "Dense"
                },
                {   "kwargs": {"activation": None,
                    "name": "output",
                    "units": 768,
                    "use_bias": True
                    },
                    "type": "Dense"
                }
            ]
        },
        "residual_dropout": 0.1,
        "use_bert_config": True
    },
    "hidden_size": 768,
    "initializer": None,
    "name": "bert_embedder"
}


xlnet_emb_hparams = {
    "name": "xlnet_encoder",
    "pretrained_model_name": "xlnet-base-cased",
    "untie_r": True,
    "num_layers": 12,
    "mem_len": 0,
    "reuse_len": 0,
    "initializer": None,
    "num_heads": 12,
    "hidden_dim": 768,
    "head_dim": 64,
    "dropout": 0.1,
    "attention_dropout": 0.1,
    "use_segments": True,
    "ffn_inner_dim": 3072,
    "activation": 'gelu',
    "vocab_size": 32000,
    "max_seq_len": 512,
}


gpt2_emb_hparams = {
    "pretrained_model_name": "gpt2-small",
    "vocab_size": 50257,
    "context_size": 1024,
    "embedding_size": 768,
    "embed": {
        "dim": 768,
        "name": "word_embeddings"
    },
    "position_size": 1024,
    "position_embed": {
        "dim": 768,
        "name": "position_embeddings"
    },
    "encoder": {
        "dim": 768,
        "num_blocks": 12,
#         "use_gpt_config": True, # Unknown hyperparameter
        "embedding_dropout": 0,
        "residual_dropout": 0,
        "multihead_attention": {
            "use_bias": True,
            "num_units": 768,
            "num_heads": 12,
            "output_dim": 768
        },
        "initializer": {
            "type": "variance_scaling_initializer",
            "kwargs": {
                "factor": 1.0,
                "mode": "FAN_AVG",
                "uniform": True
            }
        },
        "poswise_feedforward": {
            "layers": [
                {
                    "type": "Dense",
                    "kwargs": {
                        "activation": "gelu",
                        "name": "intermediate",
                        "units": 3072,
                        "use_bias": True
                    }
                },
                {
                    "type": "Dense",
                    "kwargs": {
                        "activation": None,
                        "name": "output",
                        "units": 3072,
                        "use_bias": True
                    }
                }
            ],
            "name": "ffn"
        }
    },
    "initializer": None,
    "name": "gpt2_embedder",
}


emb_hparams = {
#     "dim": 50,
    "dim": 768,
    "dropout_rate": 0.2,
    "dropout_strategy": 'element',
    "trainable": True,
    "initializer": {
        "type": "random_uniform_initializer",
        "kwargs": {
            "minval": -0.1,
            "maxval": 0.1,
            "seed": None
        }
    },
    "regularizer": {
        "type": "L1L2",
       "kwargs": {
            "l1": 0.,
            "l2": 0
        }
    },
    "name": "gen_embedder",
}


disc_emb_hparams = {
#     "dim": 50,
    "dim": 768,
    "dropout_rate": 0.4,
    "dropout_strategy": 'element',
    "trainable": True,
    "initializer": {
        "type": "random_uniform_initializer",
        "kwargs": {
            "minval": -0.1,
            "maxval": 0.1,
            "seed": None
        }
    },
    "regularizer": {
        "type": "L1L2",
        "kwargs": {
            "l1": 0.,
            "l2": 0
        }
    },
    "name": "disc_embedder",
}


clas_emb_hparams = {
#     "dim": 50,
    "dim": 768,
    "dropout_rate": 0.4,
    "dropout_strategy": 'element',
    "trainable": True,
    "initializer": {
        "type": "random_uniform_initializer",
        "kwargs": {
            "minval": -0.1,
            "maxval": 0.1,
            "seed": None
        }
    },
    "regularizer": {
        "type": "L1L2",
        "kwargs": {
            "l1": 0.,
            "l2": 0
        }
    },
    "name": "clas_embedder",
}


# GENERATOR HPARAMS
gen_hparams = {
    "rnn_decoder": {
        
        "rnn_cell": {
                "type": tensorflow.contrib.cudnn_rnn.CudnnCompatibleGRUCell,
                "kwargs": {
                    "num_units": 1024,
                    
                },
                "num_layers": 2,
                "dropout": {
                    "input_keep_prob": 1,
                    "output_keep_prob": 0.5,
                    "state_keep_prob": 1.0,
                    "variational_recurrent": True,
#                     "input_size": [emb_hparams['dim'] + noise_size + 1,
#                                    1024]
                    "input_size": [emb_hparams['dim'] + noise_size + class_size,
                                   1024]
                },
                "residual": False,
                "highway": False,
            },
    
        "max_decoding_length_train": None,
        "max_decoding_length_infer": None,
        "helper_train": {
            "type": "TrainingHelper",
            "kwargs": {}
        },
        "helper_infer": {
            "type": "SampleEmbeddingHelper",
            "kwargs": {}
        },
        "name": "generator"
    },
    
    
    "gpt2_decoder": {
        "pretrained_model_name": "gpt2-small",
        "vocab_size": 50257,
        "context_size": 1024,
        "embedding_size": 768,
        "embed": {
            "dim": 768,
            "name": "word_embeddings"
        },
        "position_size": 1024,
        "position_embed": {
            "dim": 768,
            "name": "position_embeddings"
        },

        # hparams for TransformerDecoder
        "decoder": {
            "dim": 768,
            "num_blocks": 12,
#             "use_gpt_config": True, # Unknown hyperparameter
            "embedding_dropout": 0,
            "residual_dropout": 0,
            "multihead_attention": {
                "use_bias": True,
                "num_units": 768,
                "num_heads": 12,
                "dropout_rate": 0.0,
                "output_dim": 768
            },
            "initializer": {
                "type": "variance_scaling_initializer",
                "kwargs": {
                    "factor": 1.0,
                    "mode": "FAN_AVG",
                    "uniform": True
                }
            },
            "poswise_feedforward": {
                "layers": [
                    {
                        "type": "Dense",
                        "kwargs": {
                            "activation": "gelu",
                            "name": "intermediate",
                            "units": 3072,
                            "use_bias": True
                        }
                    },
                    {
                        "type": "Dense",
                        "kwargs": {
                            "activation": None,
                            "name": "output",
                            "units": 3072,
                            "use_bias": True
                        }
                    }
                ],
                "name": "ffn"
            }
        },
        "name": "generator",
    }
}


# DISCRIMINATOR HPARAMS
disc_hparams = {
    'rnn_encoder': {
 
        "rnn_cell": {
            'type':tensorflow.contrib.cudnn_rnn.CudnnCompatibleGRUCell,
            'kwargs': {'num_units': 512},
            'num_layers': 2,
            'dropout': {'input_keep_prob': 1.0,
            'output_keep_prob': 0.5,
            'state_keep_prob': 1,
            'variational_recurrent': True,
            'input_size': [emb_hparams['dim'] + 1, 512],
            '@no_typecheck': ['input_keep_prob',
            'output_keep_prob',
            'state_keep_prob']},
            'residual': False,
            'highway': False,
            '@no_typecheck': ['type']
        },
 
        "output_layer": {
            "num_layers": 1,
            "layer_size": 1,
            "activation": "identity",
            "final_layer_activation": None,
            "other_dense_kwargs": None,
            "dropout_layer_ids": [],
            "dropout_rate": 0.5,
            "variational_dropout": True
        },
        
        'name' : 'discriminator',
         
    },
    
    
    "bi_rnn_encoder": {
        
        "rnn_cell_fw": {
            'type':tensorflow.contrib.cudnn_rnn.CudnnCompatibleGRUCell,
            'kwargs': {'num_units': 512},
            'num_layers': 2,
            'dropout': {'input_keep_prob': 1.0,
            'output_keep_prob': 0.5,
            'state_keep_prob': 1,
            'variational_recurrent': True,
            'input_size': [emb_hparams['dim'] + 1, 512],
            '@no_typecheck': ['input_keep_prob',
            'output_keep_prob',
            'state_keep_prob']},
            'residual': False,
            'highway': False,
            '@no_typecheck': ['type']
        },
        
        "rnn_cell_bw": {
            'type':tensorflow.contrib.cudnn_rnn.CudnnCompatibleGRUCell,
              'kwargs': {'num_units': 512},
              'num_layers': 2,
              'dropout': {'input_keep_prob': 1.0,
              'output_keep_prob': 0.5,
              'state_keep_prob': 1,
              'variational_recurrent': True,
              'input_size': [emb_hparams['dim'] + 1, 512],
              '@no_typecheck': ['input_keep_prob',
              'output_keep_prob',
              'state_keep_prob']},
              'residual': False,
              'highway': False,
              '@no_typecheck': ['type']
        },
        "rnn_cell_share_config": True,
        
        "output_layer_fw": {
            "num_layers": 1,
            "layer_size": 1,
            "activation": "identity",
            "final_layer_activation": None,
            "other_dense_kwargs": None,
            "dropout_layer_ids": [],
            "dropout_rate": 0.5,
            "variational_dropout": True
        },
        "output_layer_bw": {
            "num_layers": 1,
            "layer_size": 1,
            "activation": "identity",
            "final_layer_activation": None,
            "other_dense_kwargs": None,
            "dropout_layer_ids": [],
            "dropout_rate": 0.5,
            "variational_dropout": True
        },
        "output_layer_share_config": True,
        
        "name": "discriminator"
    },
    
    
    'bert_encoder' : {

        "pretrained_model_name": "bert-base-uncased",
        "embed": {
            "dim": 768,
            "name": "word_embeddings"
        },
        "vocab_size": 30522,
        "segment_embed": {
            "dim": 768,
            "name": "token_type_embeddings"
        },
        "type_vocab_size": 2,
        "position_embed": {
            "dim": 768,
            "name": "position_embeddings"
        },
        "position_size": 512,
    
        "encoder": {
            "dim": 768,
            "embedding_dropout": 0.1,
            "multihead_attention": {
                "dropout_rate": 0.1,
                "name": "self",
                "num_heads": 12,
                "num_units": 768,
                "output_dim": 768,
                "use_bias": True
            },
            "name": "encoder",
            "num_blocks": 12,
            "poswise_feedforward": {
                "layers": [
                    {   "kwargs": {
                            "activation": "gelu",
                            "name": "intermediate",
                            "units": 3072,
                            "use_bias": True
                        },
                        "type": "Dense"
                    },
                    {   "kwargs": {"activation": None,
#                         "kwargs": {"activation": "identity",
                        "name": "output",
                        "units": 768,
#                         "units": 1,
                        "use_bias": True
                        },
                        "type": "Dense"
                    }
                ]
            },
            "residual_dropout": 0.1,
            "use_bert_config": True
        },
        "hidden_size": 768,
        "initializer": None,
        "name": "discriminator"
    },
    
    
    "xlnet_encoder": {
        "name": "discriminator",
        "pretrained_model_name": "xlnet-base-cased",
        "untie_r": True,
        "num_layers": 12,
        "mem_len": 0,
        "reuse_len": 0,
        "initializer": None,
        "num_heads": 12,
        "hidden_dim": 768,
        "head_dim": 64,
        "dropout": 0.1,
        "attention_dropout": 0.1,
        "use_segments": True,
        "ffn_inner_dim": 3072,
        "activation": 'gelu',
        "vocab_size": 32000,
        "max_seq_len": 512,
    },
    

    "output_layer": { # Additional hyperparameters
        "type": "Dense",
        "kwargs": {
            "units": 1,
            "activation": "identity",
            "name": "discriminator_output_layers"
        }
    },
    
    
    "bi_logits_output_layer": { # Additional hyperparameters
        "type": "Dense",
        "kwargs": {
            "units": 1,
            "activation": "identity",
            "name": "discriminator_bi_logits_output_layers"
        }
    },
    
    "bi_cell_output_layer": { # Additional hyperparameters
        "type": "Dense",
        "kwargs": {
            "units": 512,
            "activation": "identity",
            "name": "discriminator_bi_cell_output_layers"
        }
    }
}


disc_crit_hparams = {
    'units' : 1,
    'activation' : 'linear'
}



# CLASSIFIER HPARAMS

clas_hparams = {
    'rnn_encoder' : {
 
        "rnn_cell": {
            'type':tensorflow.contrib.cudnn_rnn.CudnnCompatibleGRUCell,
            #               'kwargs': {'num_units': 128},
            'kwargs': {'num_units': 512},
            'num_layers': 2,
            'dropout': {'input_keep_prob': 1.0,
            'output_keep_prob': 0.5,
            'state_keep_prob': 1,
            'variational_recurrent': True,
            #               'input_size': [emb_hparams['dim'], 128],
            'input_size': [emb_hparams['dim'], 512],
            '@no_typecheck': ['input_keep_prob',
            'output_keep_prob',
            'state_keep_prob']},
            'residual': False,
            'highway': False,
            '@no_typecheck': ['type']
        },
 
        "output_layer": {
            "num_layers": 1,
            "layer_size": 1,
            "activation": "identity",
            "final_layer_activation": None,
            "other_dense_kwargs": None,
            "dropout_layer_ids": [],
            "dropout_rate": 0.5,
            "variational_dropout": True
        },
        
        'name' : 'classifier',
    },
    
    "bi_rnn_encoder": {
        
        "rnn_cell_fw": {
            'type':tensorflow.contrib.cudnn_rnn.CudnnCompatibleGRUCell,
            #               'kwargs': {'num_units': 128},
            'kwargs': {'num_units': 512},
            'num_layers': 2,
            'dropout': {'input_keep_prob': 1.0,
            'output_keep_prob': 0.5,
            'state_keep_prob': 1,
            'variational_recurrent': True,
            #               'input_size': [emb_hparams['dim'], 128],
            'input_size': [emb_hparams['dim'], 512],
            '@no_typecheck': ['input_keep_prob',
            'output_keep_prob',
            'state_keep_prob']},
            'residual': False,
            'highway': False,
            '@no_typecheck': ['type']
        },
        "rnn_cell_bw": {
            'type':tensorflow.contrib.cudnn_rnn.CudnnCompatibleGRUCell,
            #               'kwargs': {'num_units': 128},
            'kwargs': {'num_units': 512},
            'num_layers': 2,
            'dropout': {'input_keep_prob': 1.0,
            'output_keep_prob': 0.5,
            'state_keep_prob': 1,
            'variational_recurrent': True,
            #               'input_size': [emb_hparams['dim'], 128],
            'input_size': [emb_hparams['dim'], 512],
            '@no_typecheck': ['input_keep_prob',
            'output_keep_prob',
            'state_keep_prob']},
            'residual': False,
            'highway': False,
            '@no_typecheck': ['type']
        },
        "rnn_cell_share_config": True,
        
        "output_layer_fw": {
            "num_layers": 1,
            "layer_size": 1,
            "activation": "identity",
            "final_layer_activation": None,
            "other_dense_kwargs": None,
            "dropout_layer_ids": [],
            "dropout_rate": 0.5,
            "variational_dropout": True
        },
        "output_layer_bw": {
            "num_layers": 1,
            "layer_size": 1,
            "activation": "identity",
            "final_layer_activation": None,
            "other_dense_kwargs": None,
            "dropout_layer_ids": [],
            "dropout_rate": 0.5,
            "variational_dropout": True
        },
        "output_layer_share_config": True,
        
        "name": "classifier"
    },
    
        
    'bert_encoder' : {

        "pretrained_model_name": "bert-base-uncased",
        "embed": {
            "dim": 768,
            "name": "word_embeddings"
        },
        "vocab_size": 30522,
        "segment_embed": {
            "dim": 768,
            "name": "token_type_embeddings"
        },
        "type_vocab_size": 2,
        "position_embed": {
            "dim": 768,
            "name": "position_embeddings"
        },
        "position_size": 512,
    
        "encoder": {
            "dim": 768,
            "embedding_dropout": 0.1,
            "multihead_attention": {
                "dropout_rate": 0.1,
                "name": "self",
                "num_heads": 12,
                "num_units": 768,
                "output_dim": 768,
                "use_bias": True
            },
            "name": "encoder",
            "num_blocks": 12,
            "poswise_feedforward": {
                "layers": [
                    {   "kwargs": {
                            "activation": "gelu",
                            "name": "intermediate",
                            "units": 3072,
                            "use_bias": True
                        },
                        "type": "Dense"
                    },
                    {   "kwargs": {"activation": None,
#                         "kwargs": {"activation": "identity",
                        "name": "output",
                        "units": 768,
#                         "units": 1,
                        "use_bias": True
                        },
                        "type": "Dense"
                    }
                ]
            },
            "residual_dropout": 0.1,
            "use_bert_config": True
        },
        "hidden_size": 768,
        "initializer": None,
        "name": "classifier"
    },
    
    
    "xlnet_encoder": {
        "name": "xlnet_encoder",
        "pretrained_model_name": "xlnet-base-cased",
        "untie_r": True,
        "num_layers": 12,
        "mem_len": 0,
        "reuse_len": 0,
        "initializer": None,
        "num_heads": 12,
        "hidden_dim": 768,
        "head_dim": 64,
        "dropout": 0.1,
        "attention_dropout": 0.1,
        "use_segments": True,
        "ffn_inner_dim": 3072,
        "activation": 'gelu',
        "vocab_size": 32000,
        "max_seq_len": 512,
    },
    
    
    "output_layer": { # Additional hyperparameters
        "type": "Dense",
        "kwargs": {
            "units": 1,
            "activation": "identity",
            "name": "classifier_output_layers"
        }
    },
    
    "bi_logits_output_layer": { # Additional hyperparameters
        "type": "Dense",
        "kwargs": {
            "units": 1,
            "activation": "identity",
            "name": "classifier_bi_logits_output_layers"
        }
    },
    
    "bi_cell_output_layer": { # Additional hyperparameters
        "type": "Dense",
        "kwargs": {
            "units": 512,
            "activation": "identity",
            "name": "classifier_bi_cell_output_layers"
        }
    }
    
}

clas_crit_hparams = {
    'units':1,
    'activation':'linear'
}

# OPTIMIZER HPARAMS 

g_opt_mle_hparams = {
    "optimizer": {
        "type": tensorflow.contrib.opt.AdamWOptimizer,
        "kwargs": {
#             'weight_decay' : 5e-3,
#             "learning_rate": 0.001
            'weight_decay' : 1e-4,
            "learning_rate": 0.0005
        }
    },
    "learning_rate_decay": {
        "type": "",
        "kwargs": {},
        "min_learning_rate": 0.,
        "start_decay_step": 0,
        "end_decay_step": 1e10
    },
    "gradient_clip": {
        "type": tensorflow.clip_by_global_norm,
        "kwargs": {'clip_norm':5}
    },
    "gradient_noise_scale": None,
    "name": None
}

g_opt_pg_hparams = {
    "optimizer": {
        "type": tensorflow.contrib.opt.AdamWOptimizer,
        "kwargs": {
            'weight_decay' : 1e-7,
            "learning_rate": 0.00005
#             "learning_rate": 0.00003
        }
    },
    "learning_rate_decay": {
        "type": "",
        "kwargs": {},
        "min_learning_rate": 0.,
        "start_decay_step": 0,
        "end_decay_step": 1e10
    },
    "gradient_clip": {
        "type": tensorflow.clip_by_global_norm,
        "kwargs": {'clip_norm':5}
    },
    "gradient_noise_scale": None,
    "name": None
}

c_opt_hparams = {
    "optimizer": {
        "type": tensorflow.contrib.opt.AdamWOptimizer,
        "kwargs": {
            'weight_decay' : 1e-4,
            "learning_rate": 0.0001
#             'weight_decay' : 1e-5,
#             "learning_rate": 0.00005
        }
    },
    "learning_rate_decay": {
        "type": "",
        "kwargs": {},
        "min_learning_rate": 0.,
        "start_decay_step": 0,
        "end_decay_step": 1e10
    },
    "gradient_clip": {
        "type": tensorflow.clip_by_global_norm,
        "kwargs": {'clip_norm':1}
    },
    "gradient_noise_scale": None,
    "name": None
}

d_opt_hparams = {
    "optimizer": {
        "type": tensorflow.contrib.opt.AdamWOptimizer,
        "kwargs": {
            'weight_decay' : 1e-4,
            "learning_rate": 0.0001
#             'weight_decay' : 1e-5,
#             "learning_rate": 0.00005
        }
    },
    "learning_rate_decay": {
        "type": "",
        "kwargs": {},
        "min_learning_rate": 0.,
        "start_decay_step": 0,
        "end_decay_step": 1e10
    },
    "gradient_clip": {
        "type": tensorflow.clip_by_global_norm,
        "kwargs": {'clip_norm':1}
    },
    "gradient_noise_scale": None,
    "name": None
}

d_crit_opt_hparams = {
    "optimizer": {
        "type": tensorflow.contrib.opt.AdamWOptimizer,
        "kwargs": {
            'weight_decay' : 1e-3,
            "learning_rate": 0.001
        }
    },
    "learning_rate_decay": {
        "type": "",
        "kwargs": {},
        "min_learning_rate": 0.,
        "start_decay_step": 0,
        "end_decay_step": 1e10
    },
    "gradient_clip": {
        "type": tensorflow.clip_by_global_norm,
        "kwargs": {'clip_norm':1e6}
    },
    "gradient_noise_scale": None,
    "name": None
}
c_crit_opt_hparams = {
    "optimizer": {
        "type": tensorflow.contrib.opt.AdamWOptimizer,
        "kwargs": {
            'weight_decay' : 1e-3,
            "learning_rate": 0.001
        }
    },
    "learning_rate_decay": {
        "type": "",
        "kwargs": {},
        "min_learning_rate": 0.0,
        "start_decay_step": 0,
        "end_decay_step": 1e10
    },
    "gradient_clip": {
        "type": tensorflow.clip_by_global_norm,
        "kwargs": {'clip_norm':1e6}
    },
    "gradient_noise_scale": None,
    "name": None
}
