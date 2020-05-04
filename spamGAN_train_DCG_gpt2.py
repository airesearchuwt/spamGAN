import tensorflow as tf
import texar.tf as tx
import json
import csv
import numpy as np
import logging
import os
import sys
import time
from copy import deepcopy

from custom_texar import custom_helpers
from custom_texar.custom_multi_aligned_data import MultiAlignedData
from custom_texar.custom_gpt2_decoder import GPT2Decoder
from custom_texar.custom_gpt2_stack import GPT2Stack
from texar.tf.utils.mode import is_train_mode

# import wrapper


class Generator(tf.keras.Model):
    """Generator wrapper for checkpointing"""
    def __init__(self, gen_config):
        super(Generator, self).__init__()
        self.decoder = GPT2Stack(hparams=gen_config["gpt2_stack"])
        self.embedder = self.decoder.embeddings()


class Discriminator(tf.keras.Model):
    """Discriminator wrapper"""
    def __init__(self, disc_config):
        super(Discriminator, self).__init__()
        self.encoder = GPT2Stack(hparams=disc_config["gpt2_stack"])
        self.embedder = self.encoder.embeddings()

class Classifier(tf.keras.Model):
    def __init__(self, class_config):
        super(Classifier, self).__init__()
        self.encoder = GPT2Stack(hparams=class_config["gpt2_stack"])
        self.embedder = self.encoder.embeddings()

def get_logger(log_dir):
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s")
    fh = logging.FileHandler("{0}/log.txt".format(log_dir))
    fh.setLevel(logging.DEBUG)
    logger = logging.getLogger("Pretrained-spamGAN")
    logger.addHandler(fh)
    return logger


class fakelogger():
    def __init__(self, f):
        self.logfile = f
    def debug(self, m):
        with open(self.logfile, 'a') as f:
            f.write(m + '\n')


def get_size(sup_dataset, unsup_dataset=None):
    return sup_dataset.dataset_size()


def print_out_array(header_names, value_lists, logger, final_line=None):
    header_format_string = ''.join(['{:<13}'] * len(header_names))
    logger.debug(header_format_string.format(*header_names))
    nvalues = len(value_lists[0])
    for i in range(nvalues):
        vals = [v[i] for v in value_lists if type(v[i]) != list]
        format_string = []
        for v in vals:
            if type(v) == np.str_:
                format_string.append('{:<13} ')
            else:
                format_string.append('{:<12.3f} ')
        format_string = ''.join(format_string)
        logger.debug(format_string.format(*vals))
    if final_line is not None:
        logger.debug(final_line)



def main(config = None):
    
#     timer = wrapper.PyTimingCPU()
#     timer.StartCounter()
    
    g = tf.Graph()
    with g.as_default():
        logger = get_logger(config["log_dir"])
        global_step = tf.train.get_or_create_global_step()
        global_mode = tx.global_mode()
        
        # Get data
        logger.info("Constructing graph...")
        train_data = MultiAlignedData(config["train_data"])
        val_data = MultiAlignedData(config["val_data"])
        test_data = MultiAlignedData(config["test_data"])
        clas_train_data_hparams = deepcopy(config["train_data"])
        clas_train_data = MultiAlignedData(config["clas_train_data"])
        clas_val_data = MultiAlignedData(config["val_data"])
        clas_test_data = MultiAlignedData(config["test_data"])
        unsup_iterator = tx.data.TrainTestDataIterator(train=train_data,
                                                 val=val_data,
                                                 test=test_data)
        nounsup_iterator = tx.data.TrainTestDataIterator(train=clas_train_data,
                                                         val=clas_val_data,
                                                         test=clas_test_data)
        use_unsup = tf.placeholder(tf.bool)
        data_batch = tf.cond(use_unsup,
                             lambda: unsup_iterator.get_next(),
                             lambda: nounsup_iterator.get_next())
        vocab = train_data.vocab('x')
        vocab_size = vocab.size

        # Inputs
        inp = data_batch['x_text_ids']
        all_data_labels = data_batch['label']
        seq_lengths = data_batch['x_length']
        # 0.5 indicates unsupervised, remove for class
        labeled = tf.logical_not(tf.equal(all_data_labels, -1))
        any_labeled = tf.reduce_any(labeled)
        label_inp = tf.squeeze(tf.gather(inp, tf.where(labeled)), axis=1)
        #
        # Notice!!! 
        # When batch_size is large enough, it's okay to do leave it there.
        # Because label_inp has relatively low possibility to be None.
        # But when batch_size is small (like 16), label_inp could be None 
        # and leads to InvalidArgumentError.
        # This has confused me for months, OMG!
        #
        
        label_seq_lengths = tf.squeeze(tf.gather(seq_lengths, tf.where(labeled)), axis=1)
        data_labels = tf.squeeze(tf.gather(all_data_labels, tf.where(labeled)), axis=1)
        

        batch_size = tf.shape(inp)[0]
        label_batch_size = tf.shape(label_inp)[0]
        padded_lengths = tf.shape(inp)[1]
        
        logger.info("Building model components...")
        
        # Generator
        generator_dropout = tf.placeholder(tf.string)
        gen_model = Generator(config["gen_hparams"])
        generator = gen_model.decoder
        gen_embedder = gen_model.embedder

        # Discriminator
        discriminator_dropout = tf.placeholder(dtype=tf.string)
        disc_model = Discriminator(config["disc_hparams"])
        discriminator = disc_model.encoder
        disc_embedder = disc_model.embedder

        # Classifier
        classifier_dropout = tf.placeholder(dtype=tf.string)
        clas_model = Classifier(config["clas_hparams"])
        classifier = clas_model.encoder
        clas_embedder = clas_model.embedder

        # Critics
        disc_crit_layer = tf.layers.Dense(**config["disc_crit_hparams"])
        disc_crit = tf.keras.layers.TimeDistributed(disc_crit_layer)
        clas_crit_layer = tf.layers.Dense(**config["clas_crit_hparams"])
        clas_crit = tf.keras.layers.TimeDistributed(clas_crit_layer)

        logger.info("Creating Generator MLE training subgraph...")
        # Pre-train Generator subgraph
        with g.name_scope('gen_mle'):
            x = inp[:, 0:(tf.shape(inp)[1]-2)]
            y = inp[:, 1:(tf.shape(inp)[1]-1)]
            y_onehot = tf.one_hot(y, vocab_size)
            
            x_lengths = tf.clip_by_value(seq_lengths, 0, tf.shape(x)[1]) # Trim non-ending sentences. 

            context_size = config["noise_size"]
            class_size = config["class_size"]
            context = tf.random.normal((batch_size, context_size))
            reclass_unlab = tf.zeros_like(all_data_labels, dtype=tf.float32)
            true_classes = tf.where(labeled, tf.cast(all_data_labels, tf.float32), reclass_unlab)
            tiled_true_classes = tf.tile(tf.expand_dims(true_classes, -1), [1, class_size]) # Increase the possibility of exposure
            context = tf.concat([context, tiled_true_classes], axis=1)
            tiled_context = tf.reshape(
                tf.tile(context, [1, tf.shape(x)[1]]), [-1, tf.shape(x)[1], context_size + class_size])

            
            outputs_mle = generator(
                decoding_strategy='train_greedy',
                inputs=x,
                mode=generator_dropout,
                mle_context=tiled_context
                )           
            logits_mle = outputs_mle.logits
            
            observed_logits = tf.zeros_like(y_onehot)

            loss_mle_full = tx.losses.sequence_sparse_softmax_cross_entropy(
                labels=y,
                logits=logits_mle,
                sequence_length=x_lengths,
                average_across_timesteps=False,
                sum_over_timesteps=False,
                average_across_batch=False,
                sum_over_batch=False
                )
            loss_mle = tf.reduce_mean(loss_mle_full)
            
            
            mle_optimizer = tx.core.get_optimizer(global_step=global_step,
                                                  hparams=config["g_opt_mle_hparams"])
            if config["is_gpt2_trainable"] is True:
                g_variables = tx.utils.collect_trainable_variables([generator])
            else:
                raise ValueError(f"Generator is frozen")
                
            mle_train_op = mle_optimizer.minimize(loss_mle,
                                                  global_step=global_step,
                                                  var_list=g_variables)
                
            mean_max_logit_mle = tf.reduce_mean(tf.reduce_max(logits_mle, axis = -1))
            mean_min_logit_mle = tf.reduce_mean(tf.reduce_min(logits_mle, axis = -1))
            mean_logit_mle = tf.reduce_mean(logits_mle)
            logit_sd_mle = tf.sqrt(tf.reduce_mean(tf.square(logits_mle)) - tf.square(mean_logit_mle))
            tf.summary.scalar('mean_logit_mle', mean_logit_mle)
            tf.summary.scalar("mean_max_logit_mle", mean_max_logit_mle)
            tf.summary.scalar("mean_min_logit_mle", mean_min_logit_mle)
            tf.summary.scalar('loss_mle', loss_mle)
            tf.summary.scalar('logit_sd_mle', logit_sd_mle)
            perplexity = tf.exp(loss_mle)
            tf.summary.scalar('perplexity', perplexity)
            mle_summaries = tf.summary.merge_all(scope='gen_mle')
        # MLE Validate summaries
        with g.name_scope('val_mle_summaries'):
            tf.summary.scalar('val_logit_sd_mle', logit_sd_mle)
            tf.summary.scalar("val_mean_max_logit_mle", mean_max_logit_mle)
            tf.summary.scalar("val_mean_min_logit_mle", mean_min_logit_mle)
            tf.summary.scalar('val_mean_logit_mle', mean_logit_mle)
            tf.summary.scalar('val_loss_mle', loss_mle)
            tf.summary.scalar('val_perplexity', perplexity)
            val_mle_summaries = tf.summary.merge_all(scope='val_mle_summaries')



        # Generate subgraph
        with g.name_scope('gen_sample'):
            if config["annealing_length"] > 0:
                max_length = tf.Variable(config["annealing_length"], dtype=tf.int32)
            else:
                max_length = tf.Variable(config["max_decoding_length_infer"], dtype=tf.int32)
            logger.info("Creating token sequence sampling subgraph...")
            start_tokens = tf.cast(tf.fill([batch_size], 
                                   vocab.bos_token_id),
                                   dtype=tf.int32)
            random_context = tf.random.normal([batch_size, config["noise_size"]])
            class_prior = tf.distributions.Bernoulli(probs=config["prior_prob"])
            random_classes = class_prior.sample((batch_size, 1))
            tiled_random_classes = tf.tile(random_classes, [1, class_size]) # Increase the possibility of exposure
            random_vector = tf.concat([random_context, 
                                       tf.cast(tiled_random_classes, tf.float32)], 
                                       axis=-1)
            random_class_onehots = tf.one_hot(random_classes, 2, axis=-1)
            end_token = vocab.eos_token_id
            
            # Specify sample strategy
            sample_strategy = config["sample_strategy"]
            sample_helper = config["sample_helper"]
            logger.info("Sampling using strategy: {}_{}...".format(sample_strategy, sample_helper))
            
            # Obtain lengths of generated sentences 
            def get_gen_lengths(gen_sample_ids):
                gen_lengths_list = tf.map_fn(
                    fn=lambda sample_id: tf.shape(sample_id),
                    elems=gen_sample_ids,
                    dtype=tf.int32
                    )
                gen_lengths = tf.squeeze(gen_lengths_list, axis=1)
                return gen_lengths
            
            if sample_strategy == "infer":
                softmax_temperature = config["sample_temperature"]
                
                if sample_helper == "greedy":
                    gpt2_context_helper = custom_helpers.GPT2ContextGreedyEmbeddingHelper(
                        embedding=gen_embedder, 
                        mode=generator_dropout, 
                        context=random_vector, 
                        max_decoding_length=max_length,
                        start_tokens=start_tokens, 
                        end_token=end_token, 
                        softmax_temperature=softmax_temperature
                        )
                elif sample_helper == "sample":
                    gpt2_context_helper = custom_helpers.GPT2ContextSampleEmbeddingHelper(
                        embedding=gen_embedder, 
                        mode=generator_dropout, 
                        context=random_vector, 
                        start_tokens=start_tokens, 
                        end_token=end_token, 
                        softmax_temperature=softmax_temperature
                        )
                elif sample_helper == "topk_sample":
                    topk = config["sample_topk"]
                    gpt2_context_helper = custom_helpers.GPT2ContextTopKSampleEmbeddingHelper(
                        embedding=gen_embedder, 
                        mode=generator_dropout, 
                        context=random_vector, 
                        start_tokens=start_tokens, 
                        end_token=end_token, 
                        top_k=topk,
                        softmax_temperature=softmax_temperature 
                        )
                elif sample_helper == "softmax":
                    gpt2_context_helper = custom_helpers.GPT2ContextSoftmaxEmbeddingHelper(
                        embedding=gen_embedder, 
                        mode=generator_dropout, 
                        context=random_vector, 
                        start_tokens=start_tokens, 
                        end_token=end_token, 
                        embedding_size=vocab_size,
                        tau=softmax_temperature 
                        )
                elif sample_helper == "gumbel_softmax":
                    gpt2_context_helper = custom_helpers.GPT2ContextGumbelSoftmaxEmbeddingHelper(
                        embedding=gen_embedder, 
                        mode=generator_dropout, 
                        context=random_vector, 
                        start_tokens=start_tokens, 
                        end_token=end_token, 
                        embedding_size=vocab_size,
                        tau=softmax_temperature 
                        )
                
                gen_outputs, gen_lengths = generator(
                    decoding_strategy="infer",
                    helper=gpt2_context_helper,
                    mode=generator_dropout,
                    max_decoding_length=max_length
                    )
                gen_logits = gen_outputs.logits
                gen_sample_ids = gen_outputs.sample_id
                
            elif sample_strategy == "train":
                softmax_temperature = config["sample_temperature"]
                gen_inputs = inp[:, 1:tf.shape(inp)[1]-1]
                gen_inputs_lengths = tf.clip_by_value(seq_lengths, 0, tf.shape(gen_inputs)[1]) # Trim non-ending sentences. 
                tiled_random_vector = tf.reshape(
                    tf.tile(random_vector, [1, tf.shape(x)[1]]), [-1, tf.shape(x)[1], context_size+class_size])
                
                if sample_helper == "topk_sample":
                    top_k = config["sample_topk"]
                    gen_inputs_time = tf.expand_dims(tf.range(tf.shape(gen_inputs)[1] ), 0)
                    gen_inputs_time = tf.broadcast_to(gen_inputs_time, [tf.shape(gen_inputs)[0], tf.shape(gen_inputs)[1] ])
                    gen_inputs_emb = gen_embedder(gen_inputs, gen_inputs_time, generator_dropout)
                    gen_inputs_emb = tf.concat(
                        [gen_inputs_emb[:, :, :(gen_inputs_emb.shape[-1]-tiled_random_vector.shape[-1])], tiled_random_vector], axis = -1)
                    
                    gpt2_context_helper = custom_helpers.GPT2TopKTrainingHelper(
                        inputs=gen_inputs_emb,
                        sequence_length=gen_inputs_lengths,
                        top_k=top_k, 
                        softmax_temperature=softmax_temperature 
                        )
                    
                    gen_outputs, gen_lengths = generator(
                        decoding_strategy="train_topk_sample",
                        helper=gpt2_context_helper,
                        mode=generator_dropout,
                        max_decoding_length=max_length
                        )
                    gen_logits = gen_outputs.logits
                    gen_sample_ids = gen_outputs.sample_id
                else:
                    gen_inputs = inp[:, 1:tf.shape(inp)[1]-1]
                    gen_inputs_lengths = tf.clip_by_value(seq_lengths, 0, tf.shape(gen_inputs)[1]) # Trim non-ending sentences. 
                    tiled_random_vector = tf.reshape(
                        tf.tile(random_vector, [1, tf.shape(x)[1]]), [-1, tf.shape(x)[1], context_size+class_size])
                    
                    gen_outputs = generator(
                        decoding_strategy="train_sample" if sample_helper == "sample" else "train_greedy",
                        inputs=gen_inputs,
                        softmax_temperature=softmax_temperature,
                        mode=generator_dropout,
                        mle_context=tiled_random_vector
                        )
                    gen_logits = gen_outputs.logits
                    gen_sample_ids = gen_outputs.sample_id
                    gen_lengths = get_gen_lengths(gen_sample_ids)
                    
            elif sample_strategy == "scheduled":
                gen_inputs = inp[:, 1:tf.shape(inp)[1]-1]
                gen_inputs_lengths = tf.clip_by_value(seq_lengths, 0, tf.shape(gen_inputs)[1]) # Trim non-ending sentences. 
                tiled_random_vector = tf.reshape(
                    tf.tile(random_vector, [1, tf.shape(x)[1]]), [-1, tf.shape(x)[1], context_size+class_size])
                gen_inputs_time = tf.expand_dims(tf.range(tf.shape(gen_inputs)[1] ), 0)
                gen_inputs_time = tf.broadcast_to(gen_inputs_time, [tf.shape(gen_inputs)[0], tf.shape(gen_inputs)[1] ])
                gen_inputs_emb = gen_embedder(gen_inputs, gen_inputs_time, generator_dropout)
                gen_inputs_emb = tf.concat(
                    [gen_inputs_emb[:, :, :(gen_inputs_emb.shape[-1]-tiled_random_vector.shape[-1])], tiled_random_vector], axis = -1)
                    
                if sample_helper == "sample":
                    sample_probability = config["sample_probability"]
                    gpt2_context_helper = custom_helpers.GPT2ScheduledEmbeddingTrainingHelper(
                        inputs=gen_inputs_emb,
                        sequence_length=gen_inputs_lengths,
                        embedding=gen_embedder, 
                        mode=generator_dropout,
                        context=random_vector, 
                        sampling_probability=sample_probability, 
                        )
                    
                    gen_outputs, gen_lengths = generator(
                        decoding_strategy="scheduled",
                        helper=gpt2_context_helper,
                        mode=generator_dropout,
                        max_decoding_length=max_length
                        )
                    gen_logits = gen_outputs.logits
                    gen_sample_ids = gen_outputs.sample_id
                else:
                    raise KeyError(f"Scheduled strategy only supports token-level")
            
            elif sample_strategy == "beam_search":
                beam_width = config["beam_width"]
                beam_random_context = tf.random.normal([beam_width * batch_size, config["noise_size"]])
                beam_class_prior = tf.distributions.Bernoulli(probs=config["prior_prob"])
                beam_random_classes = class_prior.sample((beam_width * batch_size, 1))
                beam_tiled_random_classes = tf.tile(beam_random_classes, [1, class_size]) # Increase the possibility of exposure
                beam_random_vector = tf.concat([beam_random_context, 
                                           tf.cast(beam_tiled_random_classes, tf.float32)], 
                                           axis=-1)
                gen_outputs = generator(
                    decoding_strategy="beam_search",
                    mode=generator_dropout,
                    beam_width=beam_width,
                    start_tokens=start_tokens,
                    end_token=end_token,
                    max_decoding_length=max_length,
                    sample_context=beam_random_vector
                    )
                
                if beam_width == 1:
                    gen_logits = gen_outputs.logits # Only best beam streamed out 
                    gen_sample_ids = gen_outputs.sample_id # Only best beam streamed out 
                else:
                    gen_logits = gen_outputs.logits[:, :, 0, :] # Only take the best beam
                    gen_sample_ids = gen_outputs.sample_id[:, :, 0] # Only take the best beam
                gen_lengths = get_gen_lengths(gen_sample_ids)
                
            else:
                raise KeyError(f'Unknown sampling strategy')
            
            # Inefficient, use tf.gather
            observed_gen_logits = tf.zeros_like(gen_sample_ids)
            
            
            mean_max_logit = tf.reduce_mean(tf.reduce_max(gen_logits, axis = -1))
            mean_min_logit = tf.reduce_mean(tf.reduce_min(gen_logits, axis = -1))
            mean_logit_gen = tf.reduce_mean(gen_logits)
            logit_sd_gen = tf.sqrt(tf.reduce_mean(tf.square(gen_logits)) -\
                                   tf.square(mean_logit_gen))
            mean_length = tf.reduce_mean(gen_lengths)
            max_gen_length = tf.reduce_max(gen_lengths)
            min_length = tf.reduce_min(gen_lengths)

            sample_text = vocab.map_ids_to_tokens(gen_sample_ids)
            sample_text = tf.reduce_join(sample_text, axis=-1, separator=' ')
            original_text = vocab.map_ids_to_tokens(inp)
            original_text = tf.reduce_join(original_text, axis=-1, separator=' ')

            tf.summary.scalar("mean_max_logit", mean_max_logit)
            tf.summary.scalar("mean_min_logit", mean_min_logit)
            tf.summary.scalar('logit_sd', logit_sd_gen)
            tf.summary.scalar('mean_length', mean_length)
            tf.summary.scalar('max_length', max_gen_length)
            tf.summary.scalar('min_length', min_length)
            gen_sample_summaries = tf.summary.merge_all(scope='gen_sample')
            gen_sample_summaries = tf.summary.merge([gen_sample_summaries, mle_summaries])
        # capture text
        sample_text_summary = tf.summary.text('sample_text', sample_text)
        original_text_summary = tf.summary.text('original_text', original_text)

        # Train Discriminator Subgraph
        with g.name_scope("disc_train"):
            logger.info("Creating discriminator training subgraph...")
            fake_seq = gen_sample_ids
            real_seq = inp[:, 1:-1] # remove BOS EOS token as fake does not have.
            real_seq_lengths = tf.clip_by_value(seq_lengths, 0, tf.shape(real_seq)[1])
            
            r_disc_pos_init_time = tf.expand_dims(tf.range(tf.shape(real_seq)[1]), 0)
            r_disc_pos_init_time = tf.broadcast_to(
                r_disc_pos_init_time, [tf.shape(real_seq)[0], tf.shape(real_seq)[1]])
            f_disc_pos_init_time = tf.expand_dims(tf.range(tf.shape(fake_seq)[1]), 0)
            f_disc_pos_init_time = tf.broadcast_to(
                f_disc_pos_init_time, [tf.shape(fake_seq)[0], tf.shape(fake_seq)[1]])
            real_inp = disc_embedder(real_seq, r_disc_pos_init_time, discriminator_dropout)
            fake_inp = disc_embedder(fake_seq, f_disc_pos_init_time, discriminator_dropout)

            fake_seq = fake_seq[:, :max_length]
            real_seq = real_seq[:, :max_length]
            real_inp = real_inp[:, :max_length, :]
            real_seq_lengths = tf.clip_by_value(real_seq_lengths, 0, tf.shape(real_inp)[1])
            fake_inp = fake_inp[:, :max_length, :]
            gen_lengths = tf.clip_by_value(gen_lengths, 0, tf.shape(fake_inp)[1])

            if config["add_sentence_progress"]:
                # Array of  [batch_size, tstep, 1] like 
                #  [[1, 2, 3, 4...]
                #   [1, 2, 3, 4...]]
                b_f = tf.shape(fake_inp)[0]
                t_f = tf.shape(fake_inp)[1]
                b_r = tf.shape(real_inp)[0]
                t_r = tf.shape(real_inp)[1]
                f_nsteps = tf.reshape(
                    tf.tile( 
                        tf.range(start=1, limit=(t_f + 1)),
                        [b_f]),
                    [b_f, t_f, 1])
                r_nsteps = tf.reshape(
                    tf.tile(
                        tf.range(start=1, limit=(t_r + 1)),
                        [b_r]),
                    [b_r, t_r, 1])
                f_nsteps = tf.cast(f_nsteps, tf.float32)
                r_nsteps = tf.cast(r_nsteps, tf.float32)

                gen_lengths_reshape = tf.cast(tf.reshape(
                    tf.tile(gen_lengths, [t_f]), 
                    [b_f, t_f, 1]), dtype=tf.float32)
                real_seq_lengths_reshape = tf.cast(tf.reshape(
                    tf.tile(real_seq_lengths, [t_r]),
                    [b_r, t_r, 1]), dtype=tf.float32)

                f_progress_vector = tf.ones_like(gen_lengths_reshape) -\
                    (tf.multiply(1/gen_lengths_reshape, f_nsteps))
                r_progress_vector = tf.ones_like(real_seq_lengths_reshape) -\
                    tf.multiply(1/real_seq_lengths_reshape , r_nsteps)
                f_progress_vector = tf.clip_by_value(f_progress_vector, 0, 1e8)
                r_progress_vector = tf.clip_by_value(r_progress_vector, 0, 1e8)
                real_inp = tf.concat([real_inp[:, :, :-1], r_progress_vector], axis = -1)
                fake_inp = tf.concat([fake_inp[:, :, :-1], f_progress_vector], axis = -1)
                
            if "decoder" in config["disc_hparams"]["gpt2_stack"].keys():
                r_disc_outputs = discriminator(
                    decoding_strategy='train_greedy',
                    inputs=real_inp,
                    max_decoding_length=real_seq_lengths,
                    mode=discriminator_dropout,
                    )
                r_disc_q_logit = r_disc_outputs.logits
                r_disc_cell_outputs = r_disc_outputs.cell_outputs
                
                f_disc_outputs = discriminator(
                    decoding_strategy='train_greedy',
                    inputs=fake_inp,
                    max_decoding_length=gen_lengths,
                    mode=discriminator_dropout,
                    )
                f_disc_q_logit = f_disc_outputs.logits
                f_disc_cell_outputs = f_disc_outputs.cell_outputs
            else:
                r_disc_outputs = discriminator(
                    inputs=real_inp,
                    sequence_length=real_seq_lengths,
                    mode=discriminator_dropout,
                    )
                r_disc_q_logit = r_disc_outputs.logits
                r_disc_cell_outputs = r_disc_outputs.cell_outputs
                
                f_disc_outputs = discriminator(
                    inputs=fake_inp,
                    sequence_length=gen_lengths,
                    mode=discriminator_dropout,
                    )
                f_disc_q_logit = f_disc_outputs.logits
                f_disc_cell_outputs = f_disc_outputs.cell_outputs

            
            
            r_disc_q_logit_sq = tf.squeeze(r_disc_q_logit)
            f_disc_q_logit_sq = tf.squeeze(f_disc_q_logit)
            r_disc_score = tx.losses.mask_and_reduce(r_disc_q_logit_sq,
                                              real_seq_lengths,
                                              average_across_batch=False,
                                              average_across_timesteps=True,
                                              sum_over_batch=False,
                                              sum_over_timesteps=False)
            f_disc_score = tx.losses.mask_and_reduce(f_disc_q_logit_sq,
                                              gen_lengths,
                                              average_across_batch=False,
                                              average_across_timesteps=True,
                                              sum_over_batch=False,
                                              sum_over_timesteps=False)



            r_disc_score = tf.expand_dims(r_disc_score, 1)
            f_disc_score = tf.expand_dims(f_disc_score, 1)
            true_labs = tf.ones_like(r_disc_score)
            fake_labs = tf.zeros_like(f_disc_score)
            r_disc_loss = tf.losses.sigmoid_cross_entropy(
                logits = r_disc_score,
                multi_class_labels=true_labs, 
                label_smoothing = config["disc_label_smoothing_epsilon"],
                reduction=tf.losses.Reduction.MEAN)
            f_disc_loss = tf.losses.sigmoid_cross_entropy(
                logits=f_disc_score,
                multi_class_labels=fake_labs, 
                label_smoothing = config["disc_label_smoothing_epsilon"],
                reduction=tf.losses.Reduction.MEAN)
            disc_loss = r_disc_loss + f_disc_loss
            disc_loss.set_shape(disc_loss.shape)
            
            
            d_variables = tx.utils.collect_trainable_variables([discriminator])
                
            disc_optimizer = tx.core.get_optimizer(hparams=config["d_opt_hparams"])

            disc_train_op = disc_optimizer.minimize(disc_loss,
                                                    var_list=d_variables)

            # Discriminator Critic
            r_disc_crit_inp = r_disc_cell_outputs[:, :-1]
            r_disc_crit_target = r_disc_q_logit[:, :]
            f_disc_crit_inp = f_disc_cell_outputs[:, :-1]
            f_disc_crit_target = f_disc_q_logit[:, :]
            r_disc_crit_baselines = disc_crit(r_disc_crit_inp)
            f_disc_crit_baselines = disc_crit(f_disc_crit_inp)
            
            # Initially have to predict value based on no input
            init_pred = tf.Variable(0, dtype=tf.float32)
            init_pred_tile = tf.reshape(
                tf.tile(tf.expand_dims(init_pred, 0), [batch_size]), [-1, 1, 1])
            r_disc_crit_baselines = tf.concat([init_pred_tile, r_disc_crit_baselines], axis = 1)
            f_disc_crit_baselines = tf.concat([init_pred_tile, f_disc_crit_baselines], axis = 1)
            
        
            r_disc_crit_loss = tf.losses.mean_squared_error(labels=r_disc_crit_target,
                                                          predictions=r_disc_crit_baselines,
                                                           reduction=tf.losses.Reduction.MEAN)
            f_disc_crit_loss = tf.losses.mean_squared_error(labels=f_disc_crit_target,
                                                          predictions=f_disc_crit_baselines,
                                                           reduction=tf.losses.Reduction.MEAN)

            if config["disc_crit_train_on_fake_only"]:
                disc_crit_loss = f_disc_crit_loss
            else:
                disc_crit_loss = r_disc_crit_loss + f_disc_crit_loss
            disc_crit_optimizer = tx.core.get_optimizer(hparams=config["d_crit_opt_hparams"])
            disc_crit_train_op = disc_crit_optimizer.minimize(disc_crit_loss,
                                                     var_list=[disc_crit.trainable_variables,
                                                               init_pred])


            r_probs = tf.math.sigmoid(r_disc_score)
            f_probs = tf.math.sigmoid(f_disc_score)
            r_preds = tf.cast(tf.round(r_probs), tf.int32)
            f_preds = tf.cast(tf.round(f_probs), tf.int32)
            mean_r_disc_score = tf.reduce_mean(r_disc_score)
            mean_f_disc_score = tf.reduce_mean(f_disc_score)
            mean_r_prob = tf.reduce_mean(r_probs)
            mean_f_prob = tf.reduce_mean(f_probs)
            disc_acc = tf.reduce_mean(tf.contrib.metrics.accuracy(
                tf.concat([r_preds, f_preds], axis=0), 
                tf.concat([tf.ones_like(r_preds), tf.zeros_like(f_preds)], axis=0)))
            mean_f_disc_crit_baselines = tf.reduce_mean(f_disc_crit_baselines)
            mean_r_disc_crit_baselines = tf.reduce_mean(r_disc_crit_baselines)
            f_disc_crit_rmse = tf.sqrt(f_disc_crit_loss)
            r_disc_crit_rmse = tf.sqrt(r_disc_crit_loss)
            tf.summary.scalar("disc_acc", disc_acc)
            tf.summary.scalar("disc_loss", disc_loss)
            tf.summary.scalar('mean_r_disc_score', mean_r_disc_score)
            tf.summary.scalar('mean_f_disc_score', mean_f_disc_score)
            tf.summary.scalar('mean_r_prob', mean_r_prob)
            tf.summary.scalar('mean_f_prob', mean_f_prob)
            tf.summary.scalar('f_disc_crit_rmse', f_disc_crit_rmse)
            tf.summary.scalar('mean_f_disc_crit_baselines', mean_f_disc_crit_baselines)
            if not config["disc_crit_train_on_fake_only"]:
                tf.summary.scalar('r_disc_crit_rmse', r_disc_crit_rmse)
                tf.summary.scalar('mean_r_disc_crit_baselines',mean_r_disc_crit_baselines)
            
            disc_summaries = tf.summary.merge_all(scope='disc_train')
            
        with g.name_scope('disc_val_summaries'):
            tf.summary.scalar('val_loss', disc_loss)
            tf.summary.scalar('val_acc', disc_acc)
            tf.summary.scalar('val_disc_crit_rmse', f_disc_crit_rmse)
            disc_val_summaries = tf.summary.merge_all(scope='disc_val_summaries')
        # Train Classifier Subgraph
        with g.name_scope("clas_train"):
            clas_use_fake_data = tf.placeholder(tf.bool)
            logger.info("Creating classifier training subgraph...")

            # Designate clas input
            real_label_inp = label_inp[:, 1:-1]
            
            r_clas_pos_init_time = tf.expand_dims(tf.range(tf.shape(real_label_inp)[1]), 0)
            r_clas_pos_init_time = tf.broadcast_to(
                r_clas_pos_init_time, [tf.shape(real_label_inp)[0], tf.shape(real_label_inp)[1]])
            f_clas_pos_init_time = tf.expand_dims(tf.range(tf.shape(fake_seq)[1]), 0)
            f_clas_pos_init_time = tf.broadcast_to(
                f_clas_pos_init_time, [tf.shape(fake_seq)[0], tf.shape(fake_seq)[1]])
            real_label_inp_emb = clas_embedder(real_label_inp, r_clas_pos_init_time, classifier_dropout)
            fake_label_inp_emb = clas_embedder(fake_seq, f_clas_pos_init_time, classifier_dropout)
            
            real_label_inp = real_label_inp[:, :max_length]
            real_label_inp_emb = real_label_inp_emb[:, :max_length, :]
            label_seq_lengths = tf.clip_by_value(label_seq_lengths, 0, max_length)


            if "decoder" in config["clas_hparams"]["gpt2_stack"].keys():
                r_clas_outputs = classifier(
                    decoding_strategy='train_greedy',
                    inputs=real_label_inp_emb,
                    max_decoding_length=label_seq_lengths,
                    mode=classifier_dropout,
                    )
                r_clas_q_logit = r_clas_outputs.logits
                r_clas_cell_outputs = r_clas_outputs.cell_outputs
                
                f_clas_outputs = classifier(
                    decoding_strategy='train_greedy',
                    inputs=fake_label_inp_emb,
                    max_decoding_length=gen_lengths,
                    mode=classifier_dropout,
                    )
                f_clas_q_logit = f_clas_outputs.logits
                f_clas_cell_outputs = f_clas_outputs.cell_outputs
            else:
                r_clas_outputs = classifier(
                    inputs=real_label_inp_emb,
                    sequence_length=label_seq_lengths,
                    mode=classifier_dropout,
                    )
                r_clas_q_logit = r_clas_outputs.logits
                r_clas_cell_outputs = r_clas_outputs.cell_outputs
                
                f_clas_outputs = classifier(
                    inputs=fake_label_inp_emb,
                    sequence_length=gen_lengths,
                    mode=classifier_dropout,
                    )
                f_clas_q_logit = f_clas_outputs.logits
                f_clas_cell_outputs = f_clas_outputs.cell_outputs
            


            r_clas_q_logit_sq = tf.squeeze(r_clas_q_logit)
            f_clas_q_logit_sq = tf.squeeze(f_clas_q_logit)
            # Random stopping
            r_clas_score = tx.losses.mask_and_reduce(r_clas_q_logit_sq,
                                              label_seq_lengths,
                                              average_across_batch=False,
                                              average_across_timesteps=True,
                                              sum_over_batch=False,
                                              sum_over_timesteps=False)
            f_clas_score = tx.losses.mask_and_reduce(f_clas_q_logit_sq,
                                              gen_lengths,
                                              average_across_batch=False,
                                              average_across_timesteps=True,
                                              sum_over_batch=False,
                                              sum_over_timesteps=False)
            
            
            
            random_class_labels = tf.squeeze(tf.cast(random_classes, tf.float32))
            r_clas_loss = tf.losses.sigmoid_cross_entropy(
                logits=r_clas_score, 
                multi_class_labels=data_labels,
                reduction=tf.losses.Reduction.MEAN)
            f_clas_loss = tf.losses.sigmoid_cross_entropy(
                logits=f_clas_score, 
                multi_class_labels=random_class_labels,
                reduction=tf.losses.Reduction.MEAN)
            if  config["clas_loss_on_fake_lambda"] > 0 :
                clas_loss =  tf.cond(clas_use_fake_data,
                                     lambda : r_clas_loss + config["clas_loss_on_fake_lambda"] * f_clas_loss,
                                     lambda : r_clas_loss)

            else:
                clas_loss = r_clas_loss
            
            if config["clas_min_ent_lambda"] > 0 :
                # binary entropy
                ent = tf.multiply(-tf.nn.sigmoid(f_clas_score), tf.log(tf.nn.sigmoid(f_clas_score)+ 1e-8)) -\
                        tf.multiply(1 - tf.nn.sigmoid(f_clas_score), 
                                    tf.log(1 - tf.nn.sigmoid(f_clas_score) + 1e-8))
                f_clas_ent = tf.reduce_mean(ent)
                clas_loss = tf.cond(clas_use_fake_data,
                                    lambda: clas_loss + config["clas_min_ent_lambda"] * f_clas_ent,
                                    lambda :clas_loss)


                    
            clas_loss.set_shape(clas_loss.shape)

            c_variables = tx.utils.collect_trainable_variables([classifier])
                
            clas_optimizer = tx.core.get_optimizer(hparams=config["c_opt_hparams"])
            clas_train_op = clas_optimizer.minimize(clas_loss,
                                                    var_list=c_variables)
            # Classifier critic
            r_clas_crit_inp = r_clas_cell_outputs[:, :-1]
            r_clas_crit_target = r_clas_q_logit[:, :]
            f_clas_crit_inp = f_clas_cell_outputs[:, :-1]
            f_clas_crit_target = f_clas_q_logit[:, :]
            r_clas_crit_baselines = clas_crit(r_clas_crit_inp)
            f_clas_crit_baselines = clas_crit(f_clas_crit_inp)
            init_clas_pred = tf.Variable(0, dtype=tf.float32)
            init_clas_pred_tile1 = tf.reshape(
                tf.tile(tf.expand_dims(init_clas_pred, 0), [tf.shape(r_clas_crit_inp)[0]]),
                       [-1, 1, 1])
            init_clas_pred_tile2 = tf.reshape(
                tf.tile(tf.expand_dims(init_clas_pred, 0), [tf.shape(f_clas_crit_inp)[0]]),
                        [-1, 1, 1])
            r_clas_crit_baselines = tf.concat([init_clas_pred_tile1, r_clas_crit_baselines], axis=1)
            f_clas_crit_baselines = tf.concat([init_clas_pred_tile2, f_clas_crit_baselines], axis=1)

            r_clas_crit_loss = tf.losses.mean_squared_error(labels=r_clas_crit_target,
                                                          predictions=r_clas_crit_baselines,
                                                           reduction=tf.losses.Reduction.MEAN)
            f_clas_crit_loss = tf.losses.mean_squared_error(labels=f_clas_crit_target,
                                                          predictions=f_clas_crit_baselines,
                                                           reduction=tf.losses.Reduction.MEAN)
            if config["clas_crit_train_on_fake_only"]:
                clas_crit_loss = f_clas_crit_loss
            else:
                clas_crit_loss = r_clas_crit_loss + f_clas_crit_loss

            clas_crit_optimizer = tx.core.get_optimizer(hparams=config["c_crit_opt_hparams"])
            clas_crit_train_op = clas_crit_optimizer.minimize(clas_crit_loss,
                                                              var_list=[clas_crit.trainable_variables,
                                                                       init_clas_pred])

            

            r_probs = tf.math.sigmoid(r_clas_score)
            f_probs = tf.math.sigmoid(f_clas_score)
            r_clas_preds = tf.cast(tf.round(r_probs), tf.int32)
            f_clas_preds = tf.cast(tf.round(f_probs), tf.int32)
            random_class_labels_ints = tf.cast(random_class_labels, tf.int32)
            r_clas_acc = tf.reduce_mean(tf.contrib.metrics.accuracy(
                predictions=r_clas_preds, labels=data_labels))
            f_clas_acc = tf.reduce_mean(tf.contrib.metrics.accuracy(
                predictions=f_clas_preds, labels=random_class_labels_ints))
            
            # Preparations for calculating f1 scores
            r_tp = tf.reduce_mean(tf.metrics.true_positives(
                predictions=r_clas_preds, labels=data_labels))
            f_tp = tf.reduce_mean(tf.metrics.true_positives(
                predictions=f_clas_preds, labels=random_class_labels_ints))
            r_tn = tf.reduce_mean(tf.metrics.true_negatives(
                predictions=r_clas_preds, labels=data_labels))
            f_tn = tf.reduce_mean(tf.metrics.true_negatives(
                predictions=f_clas_preds, labels=random_class_labels_ints))
            r_fp = tf.reduce_mean(tf.metrics.false_positives(
                predictions=r_clas_preds, labels=data_labels))
            f_fp = tf.reduce_mean(tf.metrics.false_positives(
                predictions=f_clas_preds, labels=random_class_labels_ints))
            r_fn = tf.reduce_mean(tf.metrics.false_negatives(
                predictions=r_clas_preds, labels=data_labels))
            f_fn = tf.reduce_mean(tf.metrics.false_negatives(
                predictions=f_clas_preds, labels=random_class_labels_ints))
            
            tf.summary.scalar('clas_loss', clas_loss)
            tf.summary.scalar('r_clas_loss', r_clas_loss)
            tf.summary.scalar('f_clas_loss', f_clas_loss)
            tf.summary.scalar('r_clas_acc', r_clas_acc)
            tf.summary.scalar('f_clas_acc', f_clas_acc)
            tf.summary.scalar('r_tp', r_tp)
            tf.summary.scalar('f_tp', f_tp)
            tf.summary.scalar('r_tn', r_tn)
            tf.summary.scalar('f_tn', f_tn)
            tf.summary.scalar('r_fp', r_fp)
            tf.summary.scalar('f_fp', f_fp)
            tf.summary.scalar('r_fn', r_fn)
            tf.summary.scalar('f_fn', f_fn)
            clas_crit_rmse = tf.sqrt(clas_crit_loss)
            r_clas_crit_rmse = tf.sqrt(r_clas_crit_loss)
            f_clas_crit_rmse = tf.sqrt(f_clas_crit_loss)
            mean_f_clas_crit_baselines = tf.reduce_mean(f_clas_crit_baselines)
            mean_r_clas_crit_baselines = tf.reduce_mean(r_clas_crit_baselines)
            tf.summary.scalar('clas_crit_rmse', clas_crit_rmse)
            tf.summary.scalar('f_clas_crit_rmse',  f_clas_crit_rmse)
            tf.summary.scalar('mean_f_clas_crit_baselines', mean_f_clas_crit_baselines)
            if not config["clas_crit_train_on_fake_only"]:
                tf.summary.scalar('r_clas_crit_rmse', r_clas_crit_rmse)
                tf.summary.scalar('mean_r_clas_crit_baselines',mean_r_clas_crit_baselines)
            if config["clas_min_ent_lambda"] >0:
                tf.summary.scalar('clas_min_ent', f_clas_ent)
            clas_summaries = tf.summary.merge_all(scope='clas_train')
            
        # Validate clas summaries
        with g.name_scope('clas_val_sum'):
            tf.summary.scalar('val_clas_loss', clas_loss)
            tf.summary.scalar('val_r_clas_loss', r_clas_loss)
            tf.summary.scalar('val_f_clas_loss', f_clas_loss)
            tf.summary.scalar('val_r_clas_acc', r_clas_acc)
            tf.summary.scalar('val_f_clas_acc', f_clas_acc)
            val_clas_summaries = tf.summary.merge_all(scope='clas_val_sum')

            
        # Generator Policy Gradient Training
        with g.name_scope('pg_train'):
            logger.info("Creating policy gradient subgraph...")

            def blend(dscore, cscore, blend_factor):
                return dscore +  blend_factor * cscore
            # Convert gen logits to selected action log probs 
            p = tf.print(tf.shape(gen_logits), 
                         tf.shape(gen_sample_ids),
                         gen_logits, gen_sample_ids)

            log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=gen_logits,
                                                                           labels=gen_sample_ids)
            
            # Note the negation here.
            log_probs = -tf.clip_by_value(log_probs, config["min_log_prob"], config["max_log_prob"])

            # Critic baselines
            disc_baseline = f_disc_crit_baselines
            clas_baseline = f_clas_crit_baselines

            disc_rewards = f_disc_q_logit


            clas_rewards = tf.where(tf.squeeze(tf.cast(random_classes, tf.bool)),
                                    f_clas_q_logit, # Random class is 1
                                    -f_clas_q_logit) # Random class is 0
            # Flip baselines to match
            clas_baseline = tf.where(tf.squeeze(tf.cast(random_classes, tf.bool)),
                                     clas_baseline, # random clas is 1
                                     -clas_baseline) # random class is 0
                                     

            if config["classifier_loss_lambda"] >0 and config["reward_blending"] == 'additive':
                rewards = config["discriminator_loss_lambda"] * tf.nn.sigmoid(disc_rewards) +\
                        config["classifier_loss_lambda"] * tf.nn.sigmoid(clas_rewards)
                advantages = (config["discriminator_loss_lambda"] *
                    (tf.nn.sigmoid(disc_rewards) - tf.nn.sigmoid(disc_baseline))) +\
                    config["classifier_loss_lambda"] * (tf.nn.sigmoid(clas_rewards) - tf.nn.sigmoid(clas_baseline))
            else:
                rewards = tf.nn.sigmoid(disc_rewards)
                advantages = rewards - tf.nn.sigmoid(disc_baseline)
            
            if config["reward_blending"] == 'f1':
                rewards = 2 *tf.multiply(config["discriminator_loss_lambda"] * tf.nn.sigmoid(disc_rewards),
                                      config["classifier_loss_lambda"] * tf.nn.sigmoid(clas_rewards))
                rewards = tf.divide(rewards, (config["discriminator_loss_lambda"] * tf.nn.sigmoid(disc_rewards) + 
                                              config["classifier_loss_lambda"] * tf.nn.sigmoid(clas_rewards)))
                baseline = 2 * tf.multiply(config["discriminator_loss_lambda"] * tf.nn.sigmoid(disc_baseline),
                                       config["classifier_loss_lambda"] * tf.nn.sigmoid(clas_baseline))
                baseline = tf.divide(baseline, (config["discriminator_loss_lambda"] * tf.nn.sigmoid(disc_baseline) + 
                                                config["classifier_loss_lambda"] *  tf.nn.sigmoid(clas_baseline)))
                advantages = rewards - baseline
            
            
            advantages = tf.squeeze(advantages)
            if config["linear_decay_pg_weights"]:
                steps = tf.reshape(
                    tf.tile(
                        tf.range(0.0, tf.cast(tf.shape(advantages)[1], dtype=tf.float32), dtype=tf.float32),
                        [tf.shape(advantages)[0]]),
                    [-1, tf.shape(advantages)[1]])
                alpha = tf.cast(tf.expand_dims(gen_lengths, 1), dtype=tf.float32) - steps
                alpha = tx.utils.mask_sequences(alpha, gen_lengths)
                advantages = tf.multiply(advantages, alpha)
    
            
            if config["norm_advantages"]:
                advantages = tx.losses.discount_reward(advantages, 
                                                       sequence_length=gen_lengths,
                                                       discount=0,
                                                       normalize=True,
                                                       tensor_rank=2)
            
            advantages = tf.clip_by_value(advantages, -config["adv_max_clip"], config["adv_max_clip"])

            
            pg_loss_full = tx.losses.pg_loss_with_log_probs(
                log_probs=log_probs, 
                advantages=advantages,
                sequence_length = gen_lengths,
                average_across_batch=False, 
                average_across_timesteps=False,
                rank=1,
                sum_over_batch=False,
                sum_over_timesteps=False)


            pg_loss = tx.losses.pg_loss_with_log_probs(
                log_probs=log_probs, 
                advantages=advantages,
                sequence_length = gen_lengths,
                average_across_batch=True, 
                average_across_timesteps=True,
                rank=1,
                sum_over_batch=False,
                sum_over_timesteps=False)
            
            
            pg_loss.set_shape((pg_loss.shape))

            if config["is_gpt2_trainable"] is True:
                pg_variables = tx.utils.collect_trainable_variables([generator])
            else:
                raise ValueError(f"Generator is frozen")
                
            
            pg_optimizer = tx.core.get_optimizer(global_step=global_step,
                                                 hparams=config["g_opt_pg_hparams"])
            pg_train_op = pg_optimizer.minimize(pg_loss,
                                                global_step=global_step,
                                                var_list=pg_variables)
            

            mean_reward = tf.reduce_mean(rewards)
            mean_adv = tf.reduce_mean(advantages)
            adv_sd = tf.reduce_mean(tf.square(advantages)) - tf.square(mean_adv)
            reward_sd = tf.reduce_mean(tf.square(rewards)) - tf.square(mean_reward)
            mean_log_prob= tf.reduce_mean(log_probs)
            max_log_prob = tf.reduce_max(log_probs)
            pg_loss_sd = tf.reduce_mean(tf.square(pg_loss_full)- tf.square(pg_loss))
            
            # Rank of pg_loss must be 0
            pg_loss = tf.reduce_mean(pg_loss)
            
            tf.summary.scalar('mean_reward', mean_reward)
            tf.summary.scalar('mean_adv', mean_adv)
            tf.summary.scalar('adv_sd', adv_sd)
            tf.summary.scalar('pg_loss', pg_loss)
            tf.summary.scalar('pg_loss_sd', pg_loss_sd)
            tf.summary.scalar('mean_logit_gen', mean_logit_gen)
            tf.summary.scalar('mean_log_prob', mean_log_prob)
            tf.summary.scalar('max_log_prob', max_log_prob)
            pg_summaries = tf.summary.merge_all(scope='pg_train')
            pg_summaries = tf.summary.merge([gen_sample_summaries, pg_summaries])
    # END GRAPH 


    # Summary slicing ops
            y_sl = y[0, :]
            observed_logits_sl = observed_logits[0, :]
            loss_mle_full_sl = loss_mle_full[0, :]

            gen_sample_ids_sl = gen_sample_ids[0, :]
            observed_gen_logits_sl = observed_gen_logits[0, :]
            log_probs_sl= log_probs[0, :]
            disc_rewards_sl = disc_rewards[0, :]
            clas_rewards_sl = clas_rewards[0, :]
            disc_baseline_sl = disc_baseline[0, :]
            clas_baseline_sl = clas_baseline[0, :]
            rewards_sl = rewards[0, :]
            advantages_sl = advantages[0, :]
            pg_loss_full_sl = pg_loss_full[0, :]

            fake_seq_sl = fake_seq[0, :]
            real_seq_sl = real_seq[0, :]
            r_disc_q_logit_sl = r_disc_q_logit[0, :]
            f_disc_q_logit_sl = f_disc_q_logit[0, :]
            r_disc_score_sl = r_disc_score[0, 0]
            f_disc_score_sl = f_disc_score[0, 0]
            r_disc_crit_baselines_sl = r_disc_crit_baselines[0, :]

            real_label_inp_sl = real_label_inp[0, :]
            r_clas_q_logit_sl = r_clas_q_logit[0, :]
            f_clas_q_logit_sl = f_clas_q_logit[0, :]
            r_clas_score_sl = r_clas_score[0]
            f_clas_score_sl = f_clas_score[0]

            r_clas_crit_baselines_sl = r_clas_crit_baselines[0, :]
            data_labels_sl = data_labels[0]
            random_classes_sl = random_classes[0]
            true_class_sl = all_data_labels[0]


    # Epoch running
    def gen_run_epoch(sess, mode_string, writer, train_with_unsup = True):
        if mode_string == 'train' or mode_string == 'pretrain':
            unsup_iterator.switch_to_train_data(sess)
            nounsup_iterator.switch_to_train_data(sess)
            modekey = tf.estimator.ModeKeys.TRAIN
            size = get_size(train_data)
        elif mode_string == 'val':
            unsup_iterator.switch_to_val_data(sess)
            nounsup_iterator.switch_to_val_data(sess)
            modekey = tf.estimator.ModeKeys.EVAL
            size = get_size(val_data)
        elif mode_string == 'test':
            unsup_iterator.switch_to_test_data(sess)
            nounsup_iterator.switch_to_test_data(sess)
            modekey = tf.estimator.ModeKeys.EVAL
            size = get_size(test_data)
            total_perp=0
        progbar = tf.keras.utils.Progbar(size, 30, 1, 0.05)
        nexamples = 0.01 # avoid a divide by zero
        total_loss = 0
        gen_step = 0
        total_runtime = 0
        if config["log_verbose_mle"] or config["log_verbose_rl"]:
            fl = fakelogger('{}/logs.txt'.format(config["log_dir"]))
        while True:
            try:
                log_mle = False
                log_rl = False
                start_time = time.time()
                
                if mode_string == 'pretrain':
                    fetches = {
                        'loss' : loss_mle,
                        'batch_size' : batch_size,
                        'global_step' : global_step,
                        'mle_train_op' : mle_train_op,
                        'summaries' : mle_summaries,

                    }
                    if  gen_step % config["batches_per_summary"] == 0:
                        fetches['summaries'] = mle_summaries
                    if gen_step % config["batches_per_text_summary"] == 0 and config["log_verbose_mle"]:
                        log_mle = True
                        fetches['sentence'] = y_sl
                        fetches['logits'] = observed_logits_sl
                        fetches['full_cross_ent'] = loss_mle_full_sl
                        fetches['class'] = true_class_sl
                    

                elif mode_string == 'train':
                    fetches = {
                        'mean_adv' : mean_adv,
                        'batch_size' : batch_size,
                        'mean_reward' : mean_reward,
                        'loss' : pg_loss,
                        'train_op' : pg_train_op,
                        'global_step' : global_step,
                    }
                    if  gen_step % config["batches_per_summary"] == 0:
                        fetches['summaries'] = pg_summaries
                    if gen_step % config["batches_per_text_summary"] == 0 and config["log_verbose_rl"]:
                        log_rl = True
                        fetches['sentence'] = gen_sample_ids_sl
                        fetches['logits'] = observed_gen_logits_sl
                        fetches['log_probs'] = log_probs_sl
                        fetches['disc_reward'] = disc_rewards_sl
                        fetches['clas_reward'] = clas_rewards_sl
                        fetches['disc_crit'] = disc_baseline_sl
                        fetches['clas_crit'] = clas_baseline_sl
                        fetches['qvalues'] = rewards_sl
                        fetches['advantages'] = advantages_sl
                        fetches['pg_loss_full'] = pg_loss_full_sl
                        fetches['fake_class'] = random_classes_sl
                    
                elif mode_string == 'val' or mode_string == 'test':
                    fetches = {
                        'loss' : loss_mle,
                        'batch_size' : batch_size,
                        'global_step' : global_step,
                        'summaries' : val_mle_summaries,
                        'perplexity' : perplexity
                    }
                    if  gen_step % config["batches_per_summary"] == 0:
                        fetches['summaries'] = val_mle_summaries
                    if  mode_string == 'test':
                        fetches['gen_sentences'] = gen_sample_ids
                        fetches['random_classes'] = random_classes

                if mode_string == 'train' or mode_string == 'pretrain':
                    feed_dict = {global_mode : modekey,
                                 generator_dropout : tf.estimator.ModeKeys.TRAIN,
                                 discriminator_dropout : tf.estimator.ModeKeys.PREDICT,
                                 classifier_dropout : tf.estimator.ModeKeys.PREDICT}
                if mode_string == 'val' or mode_string == 'test': 
                    feed_dict = {global_mode : modekey,
                                generator_dropout : tf.estimator.ModeKeys.EVAL,
                                discriminator_dropout : tf.estimator.ModeKeys.PREDICT,
                                classifier_dropout : tf.estimator.ModeKeys.PREDICT}
                feed_dict[use_unsup] = train_with_unsup

                rtns = sess.run(fetches, feed_dict=feed_dict, options=run_options)
                glob_step = rtns['global_step']
                loss = rtns['loss']
                bs = rtns['batch_size']
                # Summaries
                if gen_step % config["batches_per_summary"] == 0:
                    writer.add_summary(rtns['summaries'], glob_step)
                if gen_step % config["batches_per_text_summary"] == 0:
                    writer.add_summary(sess.run(sample_text_summary, 
                                                {generator_dropout : tf.estimator.ModeKeys.EVAL,
                                                 use_unsup : True}), glob_step)
                    writer.add_summary(sess.run(original_text_summary, {use_unsup : True}), glob_step)
                    # Write verbose Summaries
                    if mode_string == 'pretrain' and config["log_verbose_mle"]:
                        header = ['tkn', 'logit', 'crossent']
                        values = [list(vocab.map_ids_to_tokens_py(rtns['sentence'])), 
                                  rtns['logits'].tolist(),
                                  rtns['full_cross_ent'].tolist()]
                        final_line = 'True class: {}'.format(rtns['class'].tolist())
                        print_out_array(header, values, fl, final_line)
                    if mode_string == 'train' and config["log_verbose_rl"]:
                        header = ['tkn', 'logit', 'log_prob', 'Q_d', 'Q_c', 'V_d',
                                  'V_c', 'Q', 'A', 'pgloss']
                        values = [list(vocab.map_ids_to_tokens_py(rtns['sentence'])),
                                  rtns['logits'].squeeze().tolist(),
                                  rtns['log_probs'].squeeze().tolist(),
                                  rtns['disc_reward'].squeeze().tolist(),
                                  rtns['clas_reward'].squeeze().tolist(), 
                                  rtns['disc_crit'].squeeze().tolist(), 
                                  rtns['clas_crit'].squeeze().tolist(),
                                  rtns['qvalues'].squeeze().tolist(),
                                  rtns['advantages'].squeeze().tolist(),
                                  rtns['pg_loss_full'].squeeze().tolist(),
                                 ]
                        final_line='mean_pg_loss: {:0.02f} class {}'.format(loss, rtns['fake_class'])
                        print_out_array(header, values, fl, final_line)
                if mode_string == 'test':
                    total_perp += rtns['perplexity'] * bs

                    with open('./generated_sentences.txt', 'a') as f:
                        for i in range(rtns['gen_sentences'].shape[0]):
                            s = list(vocab.map_ids_to_tokens_py(rtns['gen_sentences'][i, :]))
                            c = rtns['random_classes'][i]
                            f.write(str(c) + '\n')
                            f.write(' '.join(s) + '\n')
                            



                total_loss += loss * bs
                nexamples += bs
                gen_step += 1
                #Update progbar
                end_time = time.time()
                per_step_time = round(end_time - start_time, 2)
                total_runtime = total_runtime + per_step_time
                progbar.update(nexamples,[('loss', loss), 
                                          ('batch_time', per_step_time), 
                                          ("total_runtime", total_runtime)]) 

                if mode_string == 'train' and nexamples > config["adv_train_max_gen_examples"]:
                    break
            except tf.errors.OutOfRangeError:
                trying_unsup=False
                break
        if mode_string == 'test':
            return {'perp' : total_perp/nexamples, 
                    'loss' : total_loss/nexamples, 
                    "total_runtime": total_runtime}
            
        return {'loss' : total_loss/nexamples, "total_runtime": total_runtime}

    def disc_run_epoch(sess, mode_string, writer, disc_step):
        if mode_string == 'train' or mode_string == 'train_critic':
            unsup_iterator.switch_to_train_data(sess)
            modekey = tf.estimator.ModeKeys.TRAIN
            size = get_size(train_data)

        elif mode_string == 'val':
            unsup_iterator.switch_to_val_data(sess)
            modekey = tf.estimator.ModeKeys.EVAL
            size = get_size(val_data)

        elif mode_string == 'test':
            unsup_iterator.switch_to_test_data(sess)
            modekey = tf.estimator.ModeKeys.EVAL
            size = get_size(test_data)
        if config["log_verbose_mle"] or config["log_verbose_rl"]:
            fl = fakelogger('{}/logs.txt'.format(config["log_dir"]))

        nexamples = 0.01
        total_loss = 0
        total_acc = 0
        total_runtime = 0
        progbar = tf.keras.utils.Progbar(size, 30, 1, 0.05)
        while True:
            try:
                start_time = time.time()
                if mode_string == 'train':
                    fetches = {
                        'disc_loss' : disc_loss,
                        'train_op' : disc_train_op,
                        'crit_train_op' : disc_crit_train_op,
                        'disc_acc' : disc_acc,
                        'real_loss' : r_disc_loss,
                        'fake_loss' : f_disc_loss,
                        'global_step' : global_step,
                        'batch_size' : batch_size,
                    }
                    if disc_step % config["batches_per_summary"] == 0:
                        fetches['summaries'] = disc_summaries
                    if disc_step % config["batches_per_text_summary"] == 0:
                        fetches['fake_sentence'] = fake_seq_sl
                        fetches['real_sentence'] = real_seq_sl
                        fetches['r_disc_q_logit'] = r_disc_q_logit_sl
                        fetches['f_disc_q_logit'] = f_disc_q_logit_sl
                        fetches['r_disc_score'] = r_disc_score_sl
                        fetches['f_disc_score'] = f_disc_score_sl
                        fetches['r_disc_loss']  = r_disc_loss
                        fetches['f_disc_lostr30_usp42s']  = f_disc_loss
                        fetches['disc_baseline'] =  disc_baseline_sl
                        fetches['r_disc_baseline'] = r_disc_crit_baselines_sl
                        
                if mode_string == 'val':
                    fetches = {
                        'disc_loss' : disc_loss,
                        'real_loss' : r_disc_loss,
                        'fake_loss' : f_disc_loss,
                        'disc_acc' : disc_acc,
                        'batch_size' : batch_size,
                        'global_step' : global_step,
                    }
                    if disc_step % config["batches_per_summary"] == 0:
                        fetches['summaries'] = disc_val_summaries
                if mode_string == 'train_critic':
                    fetches = {
                        'disc_loss' : disc_loss,
                        'crit_train_op' : disc_crit_train_op,
                        'disc_acc' : disc_acc,
                        'real_loss' : r_disc_loss,
                        'fake_loss' : f_disc_loss,
                        'global_step' : global_step,
                        'batch_size' : batch_size
                    }
                    if disc_step % config["batches_per_summary"] == 0:
                        fetches['summaries'] = disc_summaries

                if mode_string == 'train' or mode_string == 'pretrain':
                    feed_dict = {global_mode : modekey,
                                 generator_dropout : tf.estimator.ModeKeys.EVAL,
                                 discriminator_dropout : tf.estimator.ModeKeys.TRAIN,
                                 use_unsup : True}
                else:
                    feed_dict = {global_mode : modekey,
                                 generator_dropout : tf.estimator.ModeKeys.EVAL,
                                 discriminator_dropout : tf.estimator.ModeKeys.EVAL,
                                 use_unsup : True}
                rtns = sess.run(fetches, feed_dict=feed_dict)
                glob_step = rtns['global_step']
                loss = rtns['disc_loss']
                r_loss = rtns['real_loss']
                f_loss = rtns['fake_loss']
                acc = rtns['disc_acc']
                bs = rtns['batch_size']

                if disc_step % config["batches_per_summary"] == 0:
                    writer.add_summary(
                        rtns['summaries'], disc_step)
                if disc_step % config["batches_per_text_summary"] == 0 and mode_string=='train':
                    r_header = ['tkn', 'logit', 'v_d']
                    f_header = ['tkn', 'logit', 'v_d']
                    r_values = [list(vocab.map_ids_to_tokens_py(rtns['real_sentence'])),
                                rtns['r_disc_q_logit'].squeeze().tolist(),
                                rtns['r_disc_baseline'].squeeze().tolist()
                               ]
                    f_values = [list(vocab.map_ids_to_tokens_py(rtns['fake_sentence'])),
                                rtns['f_disc_q_logit'].squeeze().tolist(),
                                rtns['disc_baseline'].squeeze().tolist()
                               ]
                    r_final_line = 'r_disc_loss: {:0.02f} r_disc_score: {:0.02f}'.format(
                        r_loss, rtns['r_disc_score'])
                    f_final_line = 'f_disc_loss: {:0.02f} f_disc_score: {:0.02f}'.format(
                        f_loss, rtns['f_disc_score'])
                    fl.debug('REAL SENTENTCE')
                    print_out_array(r_header, r_values, fl, r_final_line)
                    fl.debug('FAKE SENTENCE')
                    print_out_array(f_header, f_values, fl, f_final_line)

                if config["adv_disc_max_ex"] is not None and nexamples > config["adv_disc_max_ex"]:
                    break

                disc_step += 1
                nexamples += bs
                total_loss += loss * bs
                total_acc += rtns['disc_acc'] * bs

                #Update progbar
                end_time = time.time()
                per_step_time = round(end_time - start_time, 2)
                total_runtime = total_runtime + per_step_time
                progbar.update(nexamples, [('loss', loss), 
                                           ('batch_time', per_step_time), 
                                           ('acc', acc),
                                           ("total_runtime", total_runtime)]) 
            except tf.errors.OutOfRangeError:
                break
        
        return {'loss' : total_loss/nexamples, 
                'acc' : total_acc/nexamples, 
                'step' : disc_step,
                "total_runtime": total_runtime}
    
    def clas_run_epoch(sess, mode_string, writer, clas_step):
        total_loss = 0
        total_real_acc = 0
        total_fake_acc = 0
        nexamples = 0.01
        total_runtime = 0
        if mode_string == 'train' or mode_string == 'pretrain':
            modekey = tf.estimator.ModeKeys.TRAIN
            nounsup_iterator.switch_to_train_data(sess)
            size = get_size(clas_train_data)
        elif mode_string == 'val':
            modekey = tf.estimator.ModeKeys.EVAL
            nounsup_iterator.switch_to_val_data(sess)
            size = get_size(val_data)
            total_real_tp = 0
            total_real_tn = 0
            total_real_fp = 0
            total_real_fn = 0
        elif mode_string == 'test':
            modekey = tf.estimator.ModeKeys.EVAL
            nounsup_iterator.switch_to_test_data(sess)
            size = get_size(test_data)
            test_sent_count = 0
            total_real_tp = 0
            total_real_tn = 0
            total_real_fp = 0
            total_real_fn = 0
            preds = []
            
        progbar = tf.keras.utils.Progbar(size, 30, 1, 0.05)
        previous_acc = 0
        
        if config["log_verbose_mle"] or config["log_verbose_rl"]:
            fl = fakelogger('{}/logs.txt'.format(config["log_dir"]))
        while True:
            try:
                clas_step += 1
                start_time = time.time()
                if mode_string == 'pretrain':                   
                    fetches = {
                        'clas_loss' : clas_loss,
                        'train_op' : clas_train_op,
                        'crit_train_op' : clas_crit_train_op,
                        'r_clas_acc' : r_clas_acc,
                        'real_loss' : r_clas_loss,
                        'batch_size' : label_batch_size,
                        'global_step' : global_step,
                    }
                    if  clas_step % config["batches_per_summary"] == 0:
                        fetches['summaries'] = clas_summaries
                    if clas_step % config["batches_per_text_summary"] == 0:
                        fetches['real_sentence'] = real_label_inp_sl
                        fetches['r_clas_q_logit'] = r_clas_q_logit_sl
                        fetches['r_clas_score'] = r_clas_score_sl
                        fetches['r_clas_loss']  = r_clas_loss
                        fetches['r_baseline'] =  r_clas_crit_baselines_sl
                        fetches['real_class'] = data_labels_sl


                if mode_string == 'train':
                    fetches = {
                        'clas_loss' : clas_loss,
                        'train_op' : clas_train_op,
                        'crit_train_op' : clas_crit_train_op,
                        'r_clas_acc' : r_clas_acc,
                        'f_clas_acc' : f_clas_acc,
                        'real_loss' : r_clas_loss,
                        'fake_loss' : f_clas_loss,
                        'batch_size' : label_batch_size,
                        'global_step' : global_step,
                    }
                    if  clas_step % config["batches_per_summary"] == 0:
                        fetches['summaries'] = clas_summaries
                    if clas_step % config["batches_per_text_summary"] == 0:
                        fetches['fake_sentence'] = fake_seq_sl
                        fetches['real_sentence'] = real_label_inp_sl
                        fetches['r_clas_q_logit'] = r_clas_q_logit_sl
                        fetches['f_clas_q_logit'] = f_clas_q_logit_sl
                        fetches['r_clas_score'] = r_clas_score_sl
                        fetches['f_clas_score'] = f_clas_score_sl
                        fetches['r_clas_loss']  = r_clas_loss
                        fetches['f_clas_loss']  = f_clas_loss
                        fetches['clas_baseline'] =  clas_baseline_sl
                        fetches['fake_class'] = random_classes_sl
                        fetches['real_class'] = data_labels_sl

                if mode_string == 'val':
                    fetches = {
                        'clas_loss' : clas_loss,
                        'real_loss' : r_clas_loss,
                        'fake_loss' : f_clas_loss,
                        'r_clas_acc' : r_clas_acc,
                        'f_clas_acc' : f_clas_acc,
                        'batch_size' : label_batch_size,
                        'global_step' : global_step,
                        "r_tp": r_tp,
                        "f_tp": f_tp,
                        "r_tn": r_tn,
                        "f_tn": f_tn,
                        "r_fp": r_fp,
                        "f_fp": f_fp,
                        "r_fn": r_fn,
                        "f_fn": f_fn,
                    }
                    if  clas_step % config["batches_per_summary"] == 0:
                        fetches['summaries'] = val_clas_summaries

                if mode_string == 'test':
                    fetches = {
                        'clas_loss' : clas_loss,
                        'real_loss' : r_clas_loss,
                        'r_clas_score' : r_clas_score,
                        'r_clas_preds' : r_clas_preds,
                        'fake_loss' : f_clas_loss,
                        'r_clas_acc' : r_clas_acc,
                        'f_clas_acc' : f_clas_acc,
                        'r_clas_q_logit' : r_clas_q_logit,
                        'real_class' : data_labels,
                        "r_tp": r_tp,
                        "f_tp": f_tp,
                        "r_tn": r_tn,
                        "f_tn": f_tn,
                        "r_fp": r_fp,
                        "f_fp": f_fp,
                        "r_fn": r_fn,
                        "f_fn": f_fn,
                        'real_sentence' : real_label_inp,
                        'batch_size' : label_batch_size,
                        'global_step' : global_step,
                        'summaries' : val_clas_summaries,
                    }
                if mode_string == 'train' or mode_string == 'pretrain':
                    feed_dict = {global_mode: modekey,
                                 classifier_dropout : tf.estimator.ModeKeys.TRAIN,
                                 generator_dropout : tf.estimator.ModeKeys.EVAL,
                                 discriminator_dropout : tf.estimator.ModeKeys.TRAIN,
                                 use_unsup :False}
                else:
                    feed_dict = {global_mode: modekey,
                                 classifier_dropout : tf.estimator.ModeKeys.EVAL,
                                 generator_dropout : tf.estimator.ModeKeys.EVAL,
                                 discriminator_dropout : tf.estimator.ModeKeys.EVAL,
                                 use_unsup : False}
                if mode_string == 'pretrain':
                    feed_dict[clas_use_fake_data] = False
                elif mode_string == 'train':
                    feed_dict[clas_use_fake_data] = True
                else:
                    feed_dict[clas_use_fake_data] = False
                    
                try:
                    rtns = sess.run(fetches, feed_dict = feed_dict)
                except tf.errors.InvalidArgumentError:
                    clas_step = clas_step - 1
                    continue
                else:
                    pass
                
                glob_step = rtns['global_step']
                loss = rtns['clas_loss']
                r_loss = rtns['real_loss']
                bs = rtns['batch_size']
                acc = rtns['r_clas_acc']
                if mode_string != 'pretrain':
                    f_acc = rtns['f_clas_acc']

                if  clas_step % config["batches_per_summary"] == 0:
                    writer.add_summary(
                        rtns['summaries'], clas_step)
                if clas_step % config["batches_per_text_summary"] == 0 and mode_string == 'pretrain':
                    r_header = ['tkn', 'logit', 'V_c']
                    r_values = [list(vocab.map_ids_to_tokens_py(rtns['real_sentence'])),
                                rtns['r_clas_q_logit'].squeeze().tolist(),
                                rtns['r_baseline'].squeeze().tolist(),
                               ]
                    fl.debug('REAL SENTENTCE')
                    r_final_line = 'class: {} r_clas_loss: {:0.02f} r_clas_score: {:0.02f}'.format(
                        rtns['real_class'], rtns['r_clas_loss'], rtns['r_clas_score'])
                    print_out_array(r_header, r_values, fl, r_final_line)



                if clas_step % config["batches_per_text_summary"] == 0 and mode_string == 'train' :
                    r_header = ['tkn', 'logit']
                    f_header = ['tkn', 'logit', 'V_c']
                    r_values = [list(vocab.map_ids_to_tokens_py(rtns['real_sentence'])),
                                rtns['r_clas_q_logit'].squeeze().tolist()
                               ]
                    f_values = [list(vocab.map_ids_to_tokens_py(rtns['fake_sentence'])),
                                rtns['f_clas_q_logit'].squeeze().tolist(),
                                rtns['clas_baseline'].squeeze().tolist()
                               ]
                    fl.debug('REAL SENTENTCE')
                    r_final_line = 'class: {} r_clas_loss: {:0.02f} r_clas_score: {:0.02f}'.format(
                        rtns['real_class'], rtns['real_loss'], rtns['r_clas_score'])
                    f_final_line = 'class: {} f_clas_loss: {:0.02f} f_clas_score: {:0.02f}'.format(
                        rtns['fake_class'], rtns['f_clas_loss'], rtns['f_clas_score'])
                    print_out_array(r_header, r_values, fl, r_final_line)
                    fl.debug('FAKE SENTENCE')
                    print_out_array(f_header, f_values, fl, f_final_line)

                if clas_step % config["batches_per_text_summary"] == 0 and mode_string == 'pretrain':
                    r_header = ['tkn', 'logit']
                    f_header = ['tkn', 'logit', 'V_c']
                    r_values = [list(vocab.map_ids_to_tokens_py(rtns['real_sentence'])),
                                rtns['r_clas_q_logit'].squeeze().tolist()
                               ]
                    fl.debug('REAL SENTENTCE')
                    r_final_line = 'class: {} r_clas_loss: {:0.02f} r_clas_score: {:0.02f}'.format(
                        rtns['real_class'], rtns['real_loss'], rtns['r_clas_score'])
                    print_out_array(r_header, r_values, fl, r_final_line)
                if mode_string == 'test':
                    header = ['tkn', 'logit']
                    nsent = rtns['real_sentence'].shape[0]
                    for i in range(nsent):
                        r_values = [list(vocab.map_ids_to_tokens_py(rtns['real_sentence'][i, :])),
                                    (rtns['r_clas_q_logit'][i, :]).squeeze().tolist()
                                   ]
                        test_sent_count += 1
                        fl.debug('TEST SENT {}'.format(test_sent_count))
                        final_lines = ('class: {} r_clas_loss {:0.02f} r_clas_score {:0.02f}'
                                       'r_clas_acc {:0.02f}').format(
                                           rtns['real_class'][i], rtns['real_loss'], 
                                           rtns['r_clas_score'][i], rtns['r_clas_acc'])

                        print_out_array(header, r_values, fl, final_lines)
                    
                
                nexamples += rtns['batch_size']
                total_loss += loss * rtns['batch_size']
                total_real_acc  += rtns['r_clas_acc'] * rtns['batch_size']
                if mode_string == 'train':
                    total_fake_acc += rtns['f_clas_acc'] * rtns['batch_size']
                else:
                    total_fake_acc = 0
                if mode_string == 'test':
                    total_real_tp += rtns["r_tp"]
                    total_real_tn += rtns["r_tn"]
                    total_real_fp += rtns["r_fp"]
                    total_real_fn += rtns["r_fn"]
                    preds.extend(rtns['r_clas_preds'].squeeze().tolist())
                if  mode_string == 'val':
                    total_real_tp += rtns["r_tp"]
                    total_real_tn += rtns["r_tn"]
                    total_real_fp += rtns["r_fp"]
                    total_real_fn += rtns["r_fn"]
                
                # Update progbar
                end_time = time.time()
                per_step_time = round(end_time - start_time, 2)
                total_runtime = total_runtime + per_step_time
                if mode_string == 'pretrain':
                    progbar.update(nexamples, [('loss', loss), 
                                               ('batch_time', per_step_time), 
                                               ('acc', acc),
                                               ("total_runtime", total_runtime)])
                else:
                    progbar.update(nexamples, [('loss', loss), 
                                               ('batch_time', per_step_time),
                                               ('r_acc', acc), 
                                               ('f_acc', f_acc),
                                               ("total_runtime", total_runtime)])
                
            except tf.errors.OutOfRangeError:
                break
         
        output = {'loss' : total_loss/nexamples, 
                  'real_acc' : total_real_acc/nexamples,
                  'fake_acc' : total_fake_acc/nexamples, 
                  'step' : clas_step,
                  "total_runtime": total_runtime}
        
        if mode_string == 'test':
            if total_real_tp + total_real_fp == 0:
                real_prec = 0
            else:
                real_prec = total_real_tp / (total_real_tp + total_real_fp)
            if total_real_tp + total_real_fn == 0:
                real_recl = 0
            else:
                real_recl = total_real_tp / (total_real_tp + total_real_fn)
            if real_prec + real_recl == 0:
                real_f1 = 0
            else:
                real_f1 = 2 * real_prec * real_recl / (real_prec + real_recl)
                
            output["real_prec"] = real_prec
            output["real_recl"] = real_recl
            output["real_f1"] =  real_f1
            output['r_clas_preds'] = preds
            
        if mode_string == 'val':
            if total_real_tp + total_real_fp == 0:
                real_prec = 0
            else:
                real_prec = total_real_tp / (total_real_tp + total_real_fp)
            if total_real_tp + total_real_fn == 0:
                real_recl = 0
            else:
                real_recl = total_real_tp / (total_real_tp + total_real_fn)
            if real_prec + real_recl == 0:
                real_f1 = 0
            else:
                real_f1 = 2 * real_prec * real_recl / (real_prec + real_recl)
                
            output["real_prec"] = real_prec
            output["real_recl"] = real_recl
            output["real_f1"] =  real_f1

        return output

    run_options = tf.RunOptions()
    # Begin training loop
    sess = tf.Session(graph=g)
    breaking_gen_now = False
    with sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        logger.info("Beginning data flow...")

        # Checkpoints
        checkpoint_dir = os.path.abspath(config["checkpoint_dir"])
        checkpoint_file = config["load_checkpoint_file"]
        checkpoint = tf.train.Saver()
        if config["restore_model"]:
            checkpoint.restore(sess, checkpoint_file)
            logger.info("Checkpoint restored from {}".format(checkpoint_file))

        if config["clear_run_logs"]:
            logfiles = os.listdir(config["log_dir"])
            [os.unlink(os.path.join(config["log_dir"], f)) for f in logfiles]


        # Summaries
        sum_writer = tf.summary.FileWriter(config["log_dir"],graph=g, session=sess, flush_secs=30)
        with sum_writer:
            
            # Check if testing
            if config["gen_clas_test"] is not None and config["gen_clas_test"]:
                dict_all_res = {}
                dict_bestclas_acc_res = {}
                dict_bestclas_mixed_res = {}
                
                for ckpt in config["clas_test_ckpts"]:
                    logger.info("\nRestoring {}: ".format(ckpt))
                    checkpoint.restore(sess, os.path.join(config["checkpoint_dir"], ckpt))
                    
                    logger.info("\nTest Classifier: \n")
                    clas_test_outputs = clas_run_epoch(sess, "test", sum_writer, 0)
                    logger.info("\nClassifier test result: \n{}".format(clas_test_outputs))
                    with open(os.path.join(config["checkpoint_dir"], config["clas_pred_output"]), 'w') as f:
                        for p in clas_test_outputs['r_clas_preds']:
                            f.write(str(round(p)) + '\n')
                    
                    logger.info("\nTest Generator: \n")
                    gen_test_outputs = gen_run_epoch(sess, 'test', sum_writer, False)
                    logger.info("\nGenerator test result: \n{}".format(gen_test_outputs))
                    with open(os.path.join(config["checkpoint_dir"], config["gen_perp_output"]), 'a') as f:
                        f.write(os.getcwd() + '\n')
                        f.write(str(gen_test_outputs['perp']) + '\n')
                        
                    if "ckpt-all" in ckpt:
                        dict_all_res["accuracy"] = clas_test_outputs["real_acc"]
                        dict_all_res["f1 score"] = clas_test_outputs["real_f1"]
                        dict_all_res["perplexity"] = gen_test_outputs["perp"]
                    elif "ckpt-bestclas-acc" in ckpt:
                        dict_bestclas_acc_res["accuracy"] = clas_test_outputs["real_acc"]
                        dict_bestclas_acc_res["f1 score"] = clas_test_outputs["real_f1"]
                        dict_bestclas_acc_res["perplexity"] = gen_test_outputs["perp"]
                    elif "ckpt-bestclas-mixed" in ckpt:
                        dict_bestclas_mixed_res["accuracy"] = clas_test_outputs["real_acc"]
                        dict_bestclas_mixed_res["f1 score"] = clas_test_outputs["real_f1"]
                        dict_bestclas_mixed_res["perplexity"] = gen_test_outputs["perp"]
                    
                return dict_all_res, dict_bestclas_acc_res, dict_bestclas_mixed_res
            

            g.finalize()
            # Gen Pre-training
            logger.info("\nStarting generator pretraining...")
            min_gen_val_loss = 1e8
            patience = 0
            
            total_runtime = 0
            gen_pretrain_time = 0
            disc_pretrain_time = 0
            clas_pretrain_time = 0
            gen_adv_time = 0
            disc_adv_time = 0
            clas_adv_time = 0
            test_time = 0
            
            for e in range(config["g_pretrain_epochs"]):
                logger.info('\nGen Pretrain Epoch {}'.format(e))
                train_with_unsup = True
                gen_rtns = gen_run_epoch(sess, 'pretrain', sum_writer, train_with_unsup)
                total_runtime = total_runtime + gen_rtns["total_runtime"]
                gen_pretrain_time = gen_pretrain_time + gen_rtns["total_runtime"]
                logger.info('\nGen Validate Pretrain Epoch {}'.format(e))
                gen_rtns = gen_run_epoch(sess, 'val', sum_writer)
                total_runtime = total_runtime + gen_rtns["total_runtime"]
                gen_pretrain_time = gen_pretrain_time + gen_rtns["total_runtime"]
                logger.info('\nGen Val loss:{}'.format(gen_rtns['loss']))
                if not config["gen_patience"] <= 0:
                    pass
                if gen_rtns['loss'] < (min_gen_val_loss - config["gen_es_tolerance"]):
                    min_gen_val_loss = gen_rtns['loss']
                    patience = 0
                else:
                    patience += 1
               
                if patience > config["gen_patience"]:
                    logger.info("\nGen Early Stopping Reached at val loss {:0.02f}".format(
                        min_gen_val_loss))
                    break
            logger.info('\nMin Gen MLE val loss: {}'.format(min_gen_val_loss))
            logger.info("\nTotal runtime after generator pretrain: {}".format(total_runtime))
            
            min_disc_val_loss = 1e8
            patience = 0
            # Disc Pretraining
            logger.info("\nStarting discriminator pretraining...")
            disc_rtns = {'step' : 0}
            for e in range(config["d_pretrain_epochs"]):
                logger.info('\nDisc Pretrain Epoch {} '.format(e))
                disc_rtns = disc_run_epoch(
                    sess, 'train', sum_writer, disc_rtns['step'])
                total_runtime = total_runtime + disc_rtns["total_runtime"]
                disc_pretrain_time = disc_pretrain_time + disc_rtns["total_runtime"]
                logger.info('\nDisc Val Epoch {} '.format(e))
                disc_rtns = disc_run_epoch(
                    sess, 'val', sum_writer, disc_rtns['step'])
                total_runtime = total_runtime + disc_rtns["total_runtime"]
                disc_pretrain_time = disc_pretrain_time + disc_rtns["total_runtime"]
                
                if not config["disc_patience"] <= 0:
                    pass
                if disc_rtns['loss'] < (min_disc_val_loss - config["disc_es_tolerance"]):
                    min_disc_val_loss = disc_rtns['loss']
                    patience = 0
                else:
                    patience += 1
               
                if patience > config["disc_patience"]:
                    logger.info("\nDisc Early Stopping Reached at val loss {:0.02f}".format(
                        min_disc_val_loss))
                    break
                
            min_disc_crit_val_loss = 1e8
            patience = 0   
            logger.info('\nDiscriminator critic pretraining...')
            for e in range(config["d_pretrain_critic_epochs"]):
                logger.info('\nDisc-Crit Pretrain Epoch {}'.format(e))
                disc_rtns = disc_run_epoch(
                    sess, 'train_critic', sum_writer, disc_rtns['step'])
                total_runtime = total_runtime + disc_rtns["total_runtime"]
                disc_pretrain_time = disc_pretrain_time + disc_rtns["total_runtime"]
                
                if not config["disc_patience"] <= 0:
                    pass
                if disc_rtns['loss'] < (min_disc_crit_val_loss - config["disc_es_tolerance"]):
                    min_disc_crit_val_loss = disc_rtns['loss']
                    patience = 0
                else:
                    patience += 1
               
                if patience > config["disc_patience"]:
                    logger.info("\nDisc Early Stopping Reached at val loss {:0.02f}".format(
                        min_disc_crit_val_loss))
                    break
                
            logger.info("\nTotal runtime after discriminator pretrain: {}".format(total_runtime))


            logger.info("\nStarting classifier pretraining...")
            min_loss = 1e8
            patience = 0
            clas_rtns = {'step' : 0, 'real_acc': 0}
            for e in range(config["c_pretrain_epochs"]):
                logger.info('\nClas Pretrain Epoch {}'.format(e))
                clas_rtns = clas_run_epoch(sess, 'pretrain', sum_writer, clas_rtns['step'])
                total_runtime = total_runtime + clas_rtns["total_runtime"]
                clas_pretrain_time = clas_pretrain_time + clas_rtns["total_runtime"]
                logger.info('\nClas Validate Pretrain Epoch {}'.format(e))
                clas_rtns = clas_run_epoch(sess, 'val', sum_writer, clas_rtns['step'])
                total_runtime = total_runtime + clas_rtns["total_runtime"]
                clas_pretrain_time = clas_pretrain_time + clas_rtns["total_runtime"]
                if clas_rtns['loss'] < min_loss:
                    min_loss = clas_rtns['loss']

            logger.info('\nMin Clas Pretrain val loss: {}'.format(min_loss))
            logger.info("\nTotal runtime after classifier pretrain: {}".format(total_runtime))
            
            logger.info("\nStarting adversarial training...")
            prev_gen_val = 1e8
            extra_disc = False


            clas_adv_max_acc = 0
            clas_adv_max_mixed = 0
            breaking_gen_now = True
            
            for e in range(config["adversarial_epochs"]):
                cur_epoch = e 
                # Generator Train
                logger.info('\nGen Adv-Train Epoch {}'.format(cur_epoch))
                for i in range(config["gen_adv_epochs"]):
                    gen_rtns = gen_run_epoch(sess, 'train', sum_writer) 
                    total_runtime = total_runtime + gen_rtns["total_runtime"]
                    gen_adv_time = gen_adv_time + gen_rtns["total_runtime"]
                logger.info("\nTotal runtime after generator adv-train: {}".format(total_runtime))
                if config["mle_loss_in_adv"]:
                    for i in range(config["gen_mle_adv_epochs"]): 
                        if config["g_unlab_every_n_adv"] > 0:
                            train_with_unsup = i % config["g_unlab_every_n_adv"] == 0
                        else:
                            train_with_unsup = False
                        logger.info('\nGen Adv-MLE Train Epoch {}'.format(cur_epoch))
                        gen_rtns = gen_run_epoch(sess, 'pretrain', sum_writer, config["adv_gen_train_with_unsup"])
                        total_runtime = total_runtime + gen_rtns["total_runtime"]
                        gen_adv_time = gen_adv_time + gen_rtns["total_runtime"]
                        gen_rtns = gen_run_epoch(sess, 'val', sum_writer)
                        total_runtime = total_runtime + gen_rtns["total_runtime"]
                        gen_adv_time = gen_adv_time + gen_rtns["total_runtime"]
                        logger.info('Gen Val loss:{}'.format(gen_rtns['loss']))
                logger.info("\nTotal runtime after generator mle-train: {}".format(total_runtime))
                
                # Check discriminator loss
                if config["discriminator_loss_lambda"] > 0:
                    logger.info('\nDisc Adv-Valid Epoch {}'.format(cur_epoch))
                    disc_rtns = disc_run_epoch(
                        sess, 'val', sum_writer, disc_rtns['step'])
                    total_runtime = total_runtime + disc_rtns["total_runtime"]
                
                # Train Disc
                disc_e = 0
                while config["discriminator_loss_lambda"] > 0 and disc_e < config["disc_adv_epochs"]:
                    logger.info('\nDisc Adv-Train Epoch: {}+{}'.format(cur_epoch, disc_e))
                    disc_rtns = disc_run_epoch(sess, 'train', sum_writer, disc_rtns['step'])
                    total_runtime = total_runtime + disc_rtns["total_runtime"]
                    disc_adv_time = disc_adv_time + disc_rtns["total_runtime"]
                    logger.info('Disc Adv-Valid Epoch {}+{}'.format(cur_epoch, disc_e))
                    disc_rtns = disc_run_epoch(sess, 'val', sum_writer, disc_rtns['step'])
                    total_runtime = total_runtime + disc_rtns["total_runtime"]
                    disc_adv_time = disc_adv_time + disc_rtns["total_runtime"]
                    disc_e += 1
                logger.info("\nTotal runtime after discriminator adv-train: {}".format(total_runtime))
                
                # Generator validate
                logger.info('\nGen Adv-Valid Epoch {}'.format(cur_epoch))
                gen_rtns = gen_run_epoch(sess, 'val', sum_writer)
                total_runtime = total_runtime + gen_rtns["total_runtime"]
                gen_adv_time = gen_adv_time + gen_rtns["total_runtime"]

                # Check Clas Acc
                if config["classifier_loss_lambda"] > 0:
                    logger.info('\nClas Adv-Val Epoch {}'.format(cur_epoch))
                    clas_rtns = clas_run_epoch(sess, 'val', sum_writer, clas_rtns['step'])
                    total_runtime = total_runtime + clas_rtns["total_runtime"]
                    clas_adv_time = clas_adv_time + clas_rtns["total_runtime"]
                # Train Clas
                clas_e = 0
                while config["classifier_loss_lambda"] > 0 and clas_e < config["clas_adv_epochs"]: 
                    logger.info('\nClas Adv-Train Epoch {}+{}'.format(cur_epoch, clas_e))
                    clas_rnts = clas_run_epoch(sess, 'train', sum_writer, clas_rtns['step'])
                    total_runtime = total_runtime + clas_rtns["total_runtime"]
                    clas_adv_time = clas_adv_time + clas_rtns["total_runtime"]
                    logger.info('Clas Adv-Val Epoch {}+{}'.format(cur_epoch, clas_e))
                    clas_rtns = clas_run_epoch(sess, 'val', sum_writer, clas_rtns['step'])
                    total_runtime = total_runtime + clas_rtns["total_runtime"]
                    clas_adv_time = clas_adv_time + clas_rtns["total_runtime"]
                    clas_e += 1
                    
                    current_acc = clas_rtns['real_acc']
                    current_f1 = clas_rtns['real_f1']
                    current_mixed = 2 * current_acc * current_f1 / (current_acc + current_f1)
                    if current_acc > clas_adv_max_acc:
                        clas_adv_max_acc = current_acc
                        checkpoint.save(sess, os.path.join(checkpoint_dir, 'ckpt-bestclas-acc'))
                    if current_mixed > clas_adv_max_mixed:
                        clas_adv_max_mixed = current_mixed
                        checkpoint.save(sess, os.path.join(checkpoint_dir, 'ckpt-bestclas-mixed'))
                            
                logger.info('\nMax Clas Adv val acc: {}'.format(clas_adv_max_acc))
                logger.info('Max Clas Adv val mixed: {}'.format(clas_adv_max_mixed))
                
                logger.info("\nTotal runtime after classifier adv-train: {}".format(total_runtime))
                checkpoint.save(sess, os.path.join(checkpoint_dir, 'ckpt-all'))


            # Test perplexity of Generator and accuracy of Classifier after whole experiment
            logger.info("\nTest after the experiment: ") 
            
            logger.info("\nTest Classifier: \n")
            clas_test_outputs = clas_run_epoch(sess, "test", sum_writer, 0)
            logger.info("\nClassifier test result: \n{}".format(clas_test_outputs))
            total_runtime = total_runtime + clas_test_outputs["total_runtime"]
            test_time = test_time + clas_test_outputs["total_runtime"]
            logger.info("\nTest Generator: \n")
            gen_test_outputs = gen_run_epoch(sess, 'test', sum_writer, False)
            logger.info("\nGenerator test result: \n{}".format(gen_test_outputs))
            total_runtime = total_runtime + gen_test_outputs["total_runtime"]
            test_time = test_time + gen_test_outputs["total_runtime"]
            
            logger.info("\n*****************************************************************")
            logger.info("\n*****************************************************************")
            logger.info("\n*****************************************************************")
            logger.info("\nTotal GPU Time: {}".format(round(total_runtime, 2)))
            logger.info("\nGen Pretrain Time: {}".format(round(gen_pretrain_time, 2)))
            logger.info("\nDisc Pretrain Time: {}".format(round(disc_pretrain_time, 2)))
            logger.info("\nClas Pretrain Time: {}".format(round(clas_pretrain_time, 2)))
            logger.info("\nGen Adv Time: {}".format(round(gen_adv_time, 2)))
            logger.info("\nDisc Adv Time: {}".format(round(disc_adv_time, 2)))
            logger.info("\nClas Adv Time: {}".format(round(clas_adv_time, 2)))
            logger.info("\nTest Time: {}".format(round(test_time, 2)))
            
            
#             cpu_time = timer.GetCounter()
#             cpu_time = cpu_time/1000.0
#             logger.info("Total CPU time: {} s".format(cpu_time))
            
            dict_time_res = {
                "GPU_train_time" : round(total_runtime, 2),
                "gen_pretrain_time": round(gen_pretrain_time, 2),
                "disc_pretrain_time": round(disc_pretrain_time, 2),
                "clas_pretrain_time": round(clas_pretrain_time, 2),
                "gen_adv_time": round(gen_adv_time, 2),
                "disc_adv_time": round(disc_adv_time, 2),
                "clas_adv_time": round(clas_adv_time, 2),
                "GPU_test_time"  : round(test_time, 2),
#                 "CPU_time" : cpu_time
            }
            return dict_time_res
            

if __name__ == "__main__":
    if len(sys.argv) == 1:
        config_file = None
    else:
        config_file = sys.argv[1]
        
    # Setup
    if config_file is None:
        config_file = "spamGAN_config_smallunsup_opspam.json"
#         config_file = "spamGAN_config_smallunsup_yelp.json"
        print('No config given, using {}'.format(config_file))
        config = json.loads(open(config_file).read())
    else:
        print('Using config: {}'.format(config_file))
        config = json.loads(open(config_file).read())    
    
    # Get proportions if it's from final_runner.py
    if "tr" in config_file and "usp" in config_file:
        BASEDIR = "/home/hanfeiyu/Pretrained-spamGAN/final_experiments/opspam/spamGAN_output"
        data_paths = {}
        
        for train_pcent in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            if "tr{}".format(int(train_pcent*100)) in config_file:
                for unsup_pcent in [0.0, 0.5, 0.7, 1.0]:
                    if "usp{}".format(int(unsup_pcent * 100)) in config_file:
                        time_result_file = 'tr{}_usp{}_time'.format(int(train_pcent*100), int(unsup_pcent * 100))
                        all_result_file = 'tr{}_usp{}_all'.format(int(train_pcent*100), int(unsup_pcent * 100))
                        bestclas_acc_result_file = 'tr{}_usp{}_bestclas_acc'.format(int(train_pcent*100), int(unsup_pcent * 100))
                        bestclas_mixed_result_file = 'tr{}_usp{}_bestclas_mixed'.format(int(train_pcent*100), int(unsup_pcent * 100))
                        break
                    else:
                        continue
                break
            else:
                continue
        
        resultdir = os.path.join(BASEDIR, "result")
        data_paths["time_result_file"] = os.path.join(resultdir, time_result_file)
        data_paths["all_result_file"] = os.path.join(resultdir, all_result_file)
        data_paths["bestclas_acc_result_file"] = os.path.join(resultdir, bestclas_acc_result_file)
        data_paths["bestclas_mixed_result_file"] = os.path.join(resultdir, bestclas_mixed_result_file)
    
        # Train or test
        if config["gen_clas_test"] is False:
            dict_time_res = main(config)
            
            # Record time metrics
            time_file_exists = os.path.isfile(data_paths["time_result_file"])
            f = open(data_paths["time_result_file"],'a')
            w = csv.DictWriter(f, dict_time_res.keys())
            if not time_file_exists:
                print("Writing time header...")
                w.writeheader()
            w.writerow(dict_time_res)
            f.close()
            
        else:
            dict_all_res, dict_bestclas_pretrain_res, dict_bestclas_acc_res, dict_bestclas_f1_res, dict_bestclas_mixed_res = main(config)

            all_file_exists = os.path.isfile(data_paths["all_result_file"])
            f = open(data_paths["all_result_file"],'a')
            w = csv.DictWriter(f, dict_all_res.keys())
            if not all_file_exists:
                print("Writing all header...")
                w.writeheader()
            w.writerow(dict_all_res)
            f.close()
            
            bestclas_acc_file_exists = os.path.isfile(data_paths["bestclas_acc_result_file"])
            f = open(data_paths["bestclas_acc_result_file"],'a')
            w = csv.DictWriter(f, dict_bestclas_acc_res.keys())
            if not bestclas_acc_file_exists:
                print("Writing bestclas-acc header...")
                w.writeheader()
            w.writerow(dict_bestclas_acc_res)
            f.close()
            
            bestclas_mixed_file_exists = os.path.isfile(data_paths["bestclas_mixed_result_file"])
            f = open(data_paths["bestclas_mixed_result_file"],'a')
            w = csv.DictWriter(f, dict_bestclas_mixed_res.keys())
            if not bestclas_mixed_file_exists:
                print("Writing bestclas-mixed header...")
                w.writeheader()
            w.writerow(dict_bestclas_mixed_res)
            f.close()
    
    # Not from final_runner.py
    else:
        main(config)

    
             
