import numpy as np
from transformers import XLNetTokenizer, logging
from configs import *
logging.set_verbosity_error()


# 数据的input features 为了契合多模态在一般语言模型的基础上加上了visual, acoustic其他两种模态
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, visual, acoustic, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


# 将样例转换为Token，最大句子长度是max_seq_length(50)
def convert_to_features(examples, max_seq_length, tokenizer):
    features = []
    for (ex_index, example) in enumerate(examples):
        (words, visual, acoustic), label_id, segment = example
        tokens, inversions = [], []
        for idx, word in enumerate(words):
            tokenized = tokenizer.tokenize(word)
            tokens.extend(tokenized)
            inversions.extend([idx] * len(tokenized))

        # Check inversion
        assert len(tokens) == len(inversions)

        aligned_visual = []
        aligned_audio = []

        for inv_idx in inversions:
            aligned_visual.append(visual[inv_idx, :])
            aligned_audio.append(acoustic[inv_idx, :])

        visual = np.array(aligned_visual)
        acoustic = np.array(aligned_audio)

        # Truncate input if necessary
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: max_seq_length - 2]
            acoustic = acoustic[: max_seq_length - 2]
            visual = visual[: max_seq_length - 2]

        prepare_input = prepare_xlnet_input

        input_ids, visual, acoustic, input_mask, segment_ids = prepare_input(
            tokens, visual, acoustic, tokenizer
        )

        # Check input length
        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        assert len(segment_ids) == args.max_seq_length
        assert acoustic.shape[0] == args.max_seq_length
        assert visual.shape[0] == args.max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                visual=visual,
                acoustic=acoustic,
                label_id=label_id,
            )
        )
    return features


# xlnet 的 input embedding
def prepare_xlnet_input(tokens, visual, acoustic, tokenizer):
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    PAD_ID = tokenizer.pad_token_id

    # PAD special tokens
    tokens = tokens + [SEP] + [CLS]
    audio_zero = np.zeros((1, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic, audio_zero, audio_zero))
    visual_zero = np.zeros((1, VISUAL_DIM))
    visual = np.concatenate((visual, visual_zero, visual_zero))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * (len(tokens) - 1) + [2]

    pad_length = (args.max_seq_length - len(segment_ids))

    # then zero pad the visual and acoustic
    audio_padding = np.zeros((pad_length, ACOUSTIC_DIM))
    acoustic = np.concatenate((audio_padding, acoustic))

    video_padding = np.zeros((pad_length, VISUAL_DIM))
    visual = np.concatenate((video_padding, visual))

    input_ids = [PAD_ID] * pad_length + input_ids
    input_mask = [0] * pad_length + input_mask
    segment_ids = [3] * pad_length + segment_ids

    return input_ids, visual, acoustic, input_mask, segment_ids


def get_tokenizer(model):
    return XLNetTokenizer.from_pretrained(model)

