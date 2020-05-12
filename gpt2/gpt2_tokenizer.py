import json
import codecs
import regex as re


__all__ = ['BytePairEncoding', 'get_bpe_from_files']


class BytePairEncoding(object):

    def __init__(self,
                 token_dict,
                 bpe_rank):
        """Encode and decode of BPE.
        :param token_dict: Maps from encoded token to indices.
        :param bpe_rank: Maps from byte pair to an integer rank.
        """
        self.token_dict = token_dict
        self.token_dict_inv = {v: k for k, v in self.token_dict.items()}
        self.bpe_rank = bpe_rank
        self.byte_encoder = self.init_byte_encoder()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.token_pattern = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
        self.cache = {}

    @staticmethod
    def init_byte_encoder():
        codes = list(range(ord("!"), ord("~") + 1)) +\
                list(range(ord("¡"), ord("¬") + 1)) +\
                list(range(ord("®"), ord("ÿ") + 1))
        byte_encoder = {code: chr(code) for code in codes}
        shift = 0
        for code in range(2 ** 8):
            if code not in byte_encoder:
                byte_encoder[code] = chr(2 ** 8 + shift)
                shift += 1
        return byte_encoder

    def get_bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        chars = list(token)
        while len(chars) > 0:
            min_pair, min_rank = None, float('inf')
            for i in range(1, len(chars)):
                pair = (chars[i - 1], chars[i])
                rank = self.bpe_rank.get(pair, float('inf'))
                if rank < min_rank:
                    min_rank = rank
                    min_pair = pair
            if min_pair is None or min_pair not in self.bpe_rank:
                break
            last, tail = chars[0], 1
            for index in range(1, len(chars)):
                if (last, chars[index]) == min_pair:
                    chars[tail - 1] = last + chars[index]
                    last = last + chars[index]
                else:
                    chars[tail - 1] = last
                    tail += 1
                    last = chars[index]
            chars[tail - 1] = last
            chars = chars[:tail]
        self.cache[token] = chars
        return chars

    def encode(self, text):
        indices = []
        tokens = []
        for token in re.findall(self.token_pattern, text):
            token = bytearray(token.encode('utf-8'))
            chars = ''.join(self.byte_encoder[code] for code in token)
            indices += [self.token_dict[token] for token in self.get_bpe(chars)]
            tokens.append(chars)
        return indices, tokens

    def decode(self, tokens):
        text = ''.join([self.token_dict_inv[token] for token in tokens])
        return bytearray([self.byte_decoder[byte] for byte in text]).decode('utf-8', errors='replace')


def get_bpe_from_files(encoder_path, vocab_path):
    """Get initialized BPE.
    :param encoder_path: Path to 'encoder.json'.
    :param vocab_path: Path to 'vocab.bpe'
    :return: The object from encode and decode strings.
    """
    with codecs.open(encoder_path, 'r', 'utf8') as reader:
        token_dict = json.load(reader)
    bpe_rank = {}
    with codecs.open(vocab_path, 'r', 'utf8') as reader:
        reader.readline()
        for rank, line in enumerate(reader):
            line = line.strip()
            if line:
                bpe_rank[tuple(line.split())] = rank
    return BytePairEncoding(token_dict, bpe_rank)


def make_bpe_file(input_path, output_path, encoder_path="./gpt2-small/encoder.json", vocab_path="./gpt2-small/vocab.bpe"):
    bpe = get_bpe_from_files(encoder_path=encoder_path, vocab_path=vocab_path)
    
    with open(input_path, "r") as input, open(output_path, "w") as output:
        lines = input.readlines()
        for line in lines:
            ids, tokens = bpe.encode(line[:-1])
            sequences = " ".join(tokens)
            output.write(sequences + "\n")
            
def decode_from_bpe(
        input_path, output_path, 
        encoder_path="./gpt2-small/encoder.json", vocab_path="./gpt2-small/vocab.bpe"):
    bpe = get_bpe_from_files(encoder_path=encoder_path, vocab_path=vocab_path)
    
    with open(input_path, "r") as input, open(output_path, "w") as output:
        lines = input.readlines()
        for line in lines:
            ids, tokens = bpe.encode(line[:-1])
            sequences = " ".join(tokens)
            output.write(sequences + "\n")
            
def make_gpt2_vocab(vocab_path="./gpt2-small/encoder.json", output_path="./gpt2-small/gpt2_vocab.txt"):
    with codecs.open(vocab_path, "r", "utf8") as vocab, open(output_path, "w") as output:
        token_dict = json.load(vocab)
        for token in token_dict.keys():
            output.write(token + "\n")
            
def wipe_out_short_sentences(
        input_review_path, input_label_path, output_review_path, output_label_path,
        threshold=15
        ):
    with open(input_review_path, "r") as input_review, open(input_label_path, "r") as input_label, open(output_review_path, "w") as output_review, open(output_label_path, "w") as output_label:
        review_lines = input_review.readlines()
        label_lines = input_label.readlines()
        for i in range(len(review_lines)):
            review = review_lines[i]
            label = label_lines[i]
            if len(review.split(" ")) < threshold:
                continue
            else:
                output_review.write(review)
                output_label.write(label)

def split_nounsup(
    input_review_path, 
    input_label_path, 
    output_nounsup_review_path, 
    output_nounsup_label_path
        ):
    with open(input_review_path, "r") as input_review, open(input_label_path, "r") as input_label, open(output_nounsup_review_path, "w") as output_nounsup_review, open(output_nounsup_label_path, "w") as output_nounsup_label:
        review_lines = input_review.readlines()
        label_lines = input_label.readlines()
        for i in range(len(review_lines)):
            review = review_lines[i]
            label = label_lines[i]
            if int(label) != -1:
                output_nounsup_review.write(review)
                output_nounsup_label.write(label)

def count_words(sup_path, unsup_path, just_sup, threshold=0):
    with open(sup_path, "r") as sup, open(unsup_path, "r") as unsup:
        if just_sup is True:
            lines = sup.readlines()
        else:
            lines = sup.readlines() + unsup.readlines()
        
        cnt_words = 0
        cnt_lines = 0
        total_lines = 0
        max_words = 0
        min_words = 1e6
        dict_words = {}
        mode_words = 0
        mode_num = 0
        len_list = []
        
        for line in lines:
            words = len(line.split(" "))
            len_list.append(words)
            total_lines = total_lines + 1
            
            if words > threshold:
                cnt_words = cnt_words + words
                cnt_lines = cnt_lines + 1
            
                if max_words < words:
                    max_words = words
                if min_words > words:
                    min_words = words
                
                if str(words) in dict_words:
                    dict_words[str(words)] = dict_words[str(words)] + 1 
                else:
                    dict_words[str(words)] = 1
            
        for words in dict_words.keys():
            if dict_words[words] > mode_num:
                mode_words = int(words)
                mode_num = dict_words[words]
        
        rec_rate = 0.9
        rec_len = sorted(len_list)[int(rec_rate*cnt_lines)]
        
        print("Total words: {}".format(cnt_words))
        print("Total reviews: {}, percentage: {}".format(cnt_lines, cnt_lines/total_lines))
        print("Average words: {}".format(cnt_words/cnt_lines))
        print("Mode words: {}, number of mode: {}".format(mode_words, mode_num))
        print("Max words: {}".format(max_words))
        print("Min words: {}".format(min_words))
        print("Recommand length {} of corpus: {}".format(rec_rate, rec_len))
        
        
        


if __name__ == '__main__':
    
#     count_words("../data/opspam_reviews.txt",
#                 "../data/chicago_unlab_reviews.txt", 
#                 False)
#     count_words("../data/yelp/train_review.txt", 
#                 "../data/yelp/train_review.txt",
#                 False,
#                 256)
    
#     make_bpe_file("../minrun_train_reviews.txt", "./minrun_train_reviews_bpe.txt")
#     make_bpe_file("../minrun_val_reviews.txt", "./minrun_val_reviews_bpe.txt")
#     make_bpe_file("../minrun_test_reviews.txt", "./minrun_test_reviews_bpe.txt")

#     make_bpe_file("../data/yelp/labeled10/train_review.txt", "./yelp_train_reviews_bpe.txt")
#     make_bpe_file("../data/yelp/labeled10/val_review.txt", "./yelp_val_reviews_bpe.txt")
#     make_bpe_file("../data/yelp/test_review.txt", "./yelp_test_reviews_bpe.txt")
#     make_gpt2_vocab()
    
#     wipe_out_short_sentences(
#         "./yelp_train_reviews_bpe.txt",
#         "../data/yelp/labeled10/train_label.txt",
#         "./yelp_train_reviews_bpe_over15.txt",
#         "./yelp_train_labels_over15.txt"
#         )
#     wipe_out_short_sentences(
#         "./yelp_val_reviews_bpe.txt",
#         "../data/yelp/labeled10/val_label.txt",
#         "./yelp_val_reviews_bpe_over15.txt",
#         "./yelp_val_labels_over15.txt"
#         )
#     wipe_out_short_sentences(
#         "./yelp_test_reviews_bpe.txt",
#         "../data/yelp/test_label.txt",
#         "./yelp_test_reviews_bpe_over15.txt",
#         "./yelp_test_labels_over15.txt"
#         )
    
#     split_nounsup(
#         "./minrun_train_reviews_bpe.txt",
#         "../minrun_train_labels.txt",
#         "./minrun_train_reviews_bpe_nounsup.txt",
#         "./minrun_train_labels_nounsup.txt"
#         )
    
#     split_nounsup(
#         "./yelp_train_reviews_bpe.txt",
#         "../data/yelp/labeled10/train_label.txt",
#         "./yelp_train_reviews_bpe_nounsup.txt",
#         "./yelp_train_labels_nounsup.txt"
#         )



