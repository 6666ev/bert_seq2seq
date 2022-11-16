
import torch
import numpy as np
from bert_seq2seq.seq2seq_model import top_k_top_p_filtering
import torch.nn.functional as F
import torch.nn as nn
from transformers.generation_beam_search import BeamScorer, BeamSearchScorer
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch.nn import functional as F

from transformers.file_utils import ModelOutput
from transformers.generation_beam_search import BeamScorer, BeamSearchScorer
from transformers.generation_logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.generation_stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation_utils import (
    GreedySearchEncoderDecoderOutput,
    GreedySearchDecoderOnlyOutput,
    SampleEncoderDecoderOutput,
    SampleDecoderOnlyOutput,
    BeamSearchEncoderDecoderOutput,
    BeamSearchDecoderOnlyOutput,
    BeamSampleEncoderDecoderOutput,
    BeamSampleDecoderOnlyOutput
)
from transformers.utils import logging

logger = logging.get_logger(__name__)
GreedySearchOutput = Union[GreedySearchEncoderDecoderOutput,
                           GreedySearchDecoderOnlyOutput]
SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]
BeamSearchOutput = Union[BeamSearchEncoderDecoderOutput,
                         BeamSearchDecoderOnlyOutput]
BeamSampleOutput = Union[BeamSampleEncoderDecoderOutput,
                         BeamSampleDecoderOnlyOutput]


class ExtendModel:
    def __init__(self, model: nn.Module, tokenizer, bos_id, eos_id, device="cpu") -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bos_id = bos_id
        self.eos_id = eos_id

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def to(self, device):
        self.model.to(device)

    def load_state_dict(self, model_param, strict=True):
        self.model.load_state_dict(model_param, strict=strict)

    def state_dict(self):
        return self.model.state_dict()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()

    def sample_generate_autoregressive(self, text, input_max_length=256, out_max_length=200, top_k=30, top_p=0.0, add_eos=False):

        token_ids = self.tokenizer.encode(
            text, max_length=input_max_length, truncation=True)
        if add_eos:
            token_ids = token_ids + [self.eos_id]
        token_ids = torch.tensor(
            token_ids, device=self.device, dtype=torch.long).view(1, -1)
        output_ids = []

        with torch.no_grad():
            for step in range(out_max_length):
                scores = self.model(input_ids=token_ids)[0]
                logit_score = torch.log_softmax(
                    scores[:, -1], dim=-1).squeeze(0)
                if self.tokenizer.unk_token_id is not None:
                    logit_score[self.tokenizer.unk_token_id] = -float('Inf')
                filtered_logits = top_k_top_p_filtering(
                    logit_score, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(
                    F.softmax(filtered_logits, dim=-1), num_samples=1)
                if self.eos_id == next_token.item():
                    break
                output_ids.append(next_token.item())
                token_ids = torch.cat(
                    (token_ids, next_token.long().unsqueeze(0)), dim=1)

        return self.tokenizer.decode(output_ids)

    def sample_generate_encoder_decoder(self, text, input_max_length=256, out_max_length=200, top_k=30, top_p=0.0, add_eos=False):
        token_out = self.tokenizer.encode(text, max_length=input_max_length)
        if len(token_out) == 2:
            token_ids = token_out[0]
        else:
            token_ids = token_out
        if add_eos:
            token_ids = token_ids + [self.eos_id]

        token_ids = torch.tensor(
            token_ids, device=self.device, dtype=torch.long).view(1, -1)
        output_ids = []
        input_decoder_ids = torch.tensor(
            self.bos_id, device=self.device, dtype=torch.long).view(1, -1)
        with torch.no_grad():
            for step in range(out_max_length):
                scores = self.model(input_ids=token_ids,
                                    decoder_input_ids=input_decoder_ids)[0]
                logit_score = torch.log_softmax(
                    scores[:, -1], dim=-1).squeeze(0)
                filtered_logits = top_k_top_p_filtering(
                    logit_score, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(
                    F.softmax(filtered_logits, dim=-1), num_samples=1)
                if self.eos_id == next_token.item():
                    break
                output_ids.append(next_token.item())
                input_decoder_ids = torch.cat(
                    (input_decoder_ids, next_token.long().unsqueeze(0)), dim=1)

        return self.tokenizer.decode(output_ids)

    def my_generate_text_beam(self, text, gen_type="zm", input_max_length=300, output_max_length=200, add_eos=False):
        # text = [text]
        inputs = self.tokenizer(
            text, max_length=input_max_length,  return_tensors="pt",padding="max_length",truncation=True)

        for k in inputs.keys():
            inputs[k] = inputs[k].to(self.device)

        num_beams = 4
        gen_kwargs = {
            "max_length": output_max_length,
            "num_beams": num_beams
        }

        generated_tokens = self.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            gen_type=gen_type,
            **gen_kwargs,
        )
        generated_text =  self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return generated_text


    def sample_generate_encoder_decoder2(self, text, input_max_length=300, out_max_length=200, top_k=30, top_p=0.0, add_eos=False):
        token_out = self.tokenizer.encode(text, max_length=input_max_length)
        if len(token_out) == 2:
            token_ids = token_out[0]
        else:
            token_ids = token_out
        if add_eos:
            token_ids = token_ids + [self.eos_id]

        # batch = 2
        token_ids = torch.tensor(
            [token_ids, token_ids], device=self.device, dtype=torch.long).view(2, -1)

        zm_output_ids, xq_output_ids = [], []
        zm_input_decoder_ids = torch.tensor(
            [self.bos_id, self.bos_id], device=self.device, dtype=torch.long).view(2, -1)
        xq_input_decoder_ids = torch.tensor(
            [self.bos_id, self.bos_id], device=self.device, dtype=torch.long).view(2, -1)

        # batch = 1
        # token_ids = torch.tensor(
        #     token_ids, device=self.device, dtype=torch.long).view(1, -1)

        # zm_output_ids, xq_output_ids = [], []
        # zm_input_decoder_ids = torch.tensor(
        #     self.bos_id, device=self.device, dtype=torch.long).view(1, -1)
        # xq_input_decoder_ids = torch.tensor(
        #     self.bos_id, device=self.device, dtype=torch.long).view(1, -1)

        with torch.no_grad():
            for step in range(out_max_length):
                outputs = self.model(input_ids=token_ids, decoder_input_ids=(
                    zm_input_decoder_ids, zm_input_decoder_ids))
                scores = outputs.logits[0]  # 生成罪名 [1, 1, 21128]
                logit_score = torch.log_softmax(
                    scores[:, -1], dim=-1).squeeze(0)  # [21128]
                filtered_logits = top_k_top_p_filtering(
                    logit_score, top_k=top_k, top_p=top_p)  # [21128]
                next_token = torch.multinomial(
                    F.softmax(filtered_logits, dim=-1), num_samples=1)  # [1]
                zm_output_ids.append(next_token.item())
                zm_input_decoder_ids = torch.cat(
                    (zm_input_decoder_ids, next_token.long().unsqueeze(0)), dim=1)

            for step in range(out_max_length):
                outputs = self.model(input_ids=token_ids, decoder_input_ids=(
                    xq_input_decoder_ids, xq_input_decoder_ids))
                scores = outputs.logits[1]  # 生成刑期
                logit_score = torch.log_softmax(
                    scores[:, -1], dim=-1).squeeze(0)
                filtered_logits = top_k_top_p_filtering(
                    logit_score, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(
                    F.softmax(filtered_logits, dim=-1), num_samples=1)
                xq_output_ids.append(next_token.item())
                xq_input_decoder_ids = torch.cat(
                    (xq_input_decoder_ids, next_token.long().unsqueeze(0)), dim=1)

        return self.tokenizer.decode(zm_output_ids), self.tokenizer.decode(xq_output_ids)

    def generate_unilm(self, text, out_max_length=40, beam_size=1, max_length=256):
        # 对 一个 句子生成相应的结果
        self.out_max_length = out_max_length
        input_max_length = max_length - out_max_length
        tokenizer_out = self.tokenizer.encode(
            text, max_length=input_max_length)
        if len(tokenizer_out) != 1:
            token_ids = tokenizer_out[0]
        else:
            token_ids = tokenizer_out
        token_ids = torch.tensor(token_ids, device=self.device).view(1, -1)
        token_type_ids = torch.zeros_like(token_ids, device=self.device)

        out_puts_ids = self.beam_search(token_ids, token_type_ids, beam_size=beam_size,
                                        device=self.device)

        return self.tokenizer.decode(out_puts_ids.cpu().numpy())

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, input_ids: torch.LongTensor, model_kwargs
    ) -> Dict[str, Any]:
        if "encoder_outputs" not in model_kwargs:
            # retrieve encoder hidden states
            encoder = self.model.get_encoder()
            encoder_kwargs = {
                argument: value for argument, value in model_kwargs.items() if not argument.startswith("decoder_")
            }
            model_kwargs["encoder_outputs"]: ModelOutput = encoder(
                input_ids, return_dict=True, **encoder_kwargs)
        return model_kwargs

    def _get_decoder_start_token_id(self, decoder_start_token_id: int = None, bos_token_id: int = None) -> int:
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.model.config.decoder_start_token_id
        )
        bos_token_id = bos_token_id if bos_token_id is not None else self.model.config.bos_token_id

        if decoder_start_token_id is not None:
            return decoder_start_token_id
        elif (
            hasattr(self.model.config, "decoder")
            and hasattr(self.model.config.decoder, "decoder_start_token_id")
            and self.model.config.decoder.decoder_start_token_id is not None
        ):
            return self.model.config.decoder.decoder_start_token_id
        elif bos_token_id is not None:
            return bos_token_id
        elif (
            hasattr(self.model.config, "decoder")
            and hasattr(self.model.config.decoder, "bos_token_id")
            and self.model.config.decoder.bos_token_id is not None
        ):
            return self.model.config.decoder.bos_token_id
        raise ValueError(
            "`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation."
        )

    def _prepare_decoder_input_ids_for_generation(
        self, input_ids: torch.LongTensor, decoder_start_token_id: int = None, bos_token_id: int = None
    ) -> torch.LongTensor:
        decoder_start_token_id = self._get_decoder_start_token_id(
            decoder_start_token_id, bos_token_id)
        decoder_input_ids = (
            torch.ones((input_ids.shape[0], 1), dtype=torch.long,
                       device=input_ids.device) * decoder_start_token_id
        )
        return decoder_input_ids

    def _get_logits_processor(
        self,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        encoder_no_repeat_ngram_size: int,
        encoder_input_ids: torch.LongTensor,
        bad_words_ids: List[List[int]],
        min_length: int,
        max_length: int,
        eos_token_id: int,
        forced_bos_token_id: int,
        forced_eos_token_id: int,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        num_beams: int,
        num_beam_groups: int,
        diversity_penalty: float,
    ) -> LogitsProcessorList:
        """
        This class returns a :obj:`~transformers.LogitsProcessorList` list object that contains all relevant
        :obj:`~transformers.LogitsProcessor` instances used to modify the scores of the language model head.
        """

        # init warp parameters
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.model.config.repetition_penalty
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else self.model.config.no_repeat_ngram_size
        )
        encoder_no_repeat_ngram_size = (
            encoder_no_repeat_ngram_size
            if encoder_no_repeat_ngram_size is not None
            else self.model.config.encoder_no_repeat_ngram_size
        )
        bad_words_ids = bad_words_ids if bad_words_ids is not None else self.model.config.bad_words_ids
        min_length = min_length if min_length is not None else self.model.config.min_length
        eos_token_id = eos_token_id if eos_token_id is not None else self.model.config.eos_token_id
        diversity_penalty = diversity_penalty if diversity_penalty is not None else self.model.config.diversity_penalty
        forced_bos_token_id = (
            forced_bos_token_id if forced_bos_token_id is not None else self.model.config.forced_bos_token_id
        )
        forced_eos_token_id = (
            forced_eos_token_id if forced_eos_token_id is not None else self.model.config.forced_eos_token_id
        )
        # instantiate processors list
        processors = LogitsProcessorList()

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        if diversity_penalty is not None and diversity_penalty > 0.0:
            processors.append(
                HammingDiversityLogitsProcessor(
                    diversity_penalty=diversity_penalty, num_beams=num_beams, num_beam_groups=num_beam_groups
                )
            )
        if repetition_penalty is not None and repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(
                penalty=repetition_penalty))
        if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
            processors.append(
                NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
        if encoder_no_repeat_ngram_size is not None and encoder_no_repeat_ngram_size > 0:
            if self.model.config.is_encoder_decoder:
                processors.append(EncoderNoRepeatNGramLogitsProcessor(
                    encoder_no_repeat_ngram_size, encoder_input_ids))
            else:
                raise ValueError(
                    "It's impossible to use `encoder_no_repeat_ngram_size` with decoder-only architecture"
                )
        if bad_words_ids is not None:
            processors.append(NoBadWordsLogitsProcessor(
                bad_words_ids, eos_token_id))
        if min_length is not None and eos_token_id is not None and min_length > -1:
            processors.append(MinLengthLogitsProcessor(
                min_length, eos_token_id))
        if prefix_allowed_tokens_fn is not None:
            processors.append(PrefixConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn, num_beams // num_beam_groups))
        if forced_bos_token_id is not None:
            processors.append(
                ForcedBOSTokenLogitsProcessor(forced_bos_token_id))
        if forced_eos_token_id is not None:
            processors.append(ForcedEOSTokenLogitsProcessor(
                max_length, forced_eos_token_id))
        return processors

    def _get_stopping_criteria(
        self,
        max_length: Optional[int],
        max_time: Optional[float],
    ) -> StoppingCriteriaList:

        stopping_criteria = StoppingCriteriaList()
        if max_length is not None:
            stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
        if max_time is not None:
            stopping_criteria.append(MaxTimeCriteria(max_time=max_time))
        return stopping_criteria

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs: ModelOutput = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1,
                                                                expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(
                0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(
                0, expanded_return_idx)

        if is_encoder_decoder:
            assert encoder_outputs is not None
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(
                    encoder_outputs.last_hidden_state.device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
        return input_ids, model_kwargs

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        max_time: Optional[float] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        gen_type="zm",
        **model_kwargs,
    ) -> Union[GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, torch.LongTensor]:

        # set init values
        num_beams = num_beams if num_beams is not None else self.model.config.num_beams
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.model.config.num_beam_groups
        max_length = max_length if max_length is not None else self.model.config.max_length
        do_sample = do_sample if do_sample is not None else self.model.config.do_sample
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.model.config.num_return_sequences
        )

        pad_token_id = pad_token_id if pad_token_id is not None else self.model.config.pad_token_id
        bos_token_id = bos_token_id if bos_token_id is not None else self.model.config.bos_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.model.config.eos_token_id

        output_scores = output_scores if output_scores is not None else self.model.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.model.config.return_dict_in_generate
        )

        model_kwargs["output_attentions"] = output_attentions
        model_kwargs["output_hidden_states"] = output_hidden_states

        if input_ids is None:
            # init `input_ids` with bos_token_id
            input_ids = self._prepare_input_ids_for_generation(
                bos_token_id, model_kwargs.get("encoder_outputs"))

        if model_kwargs.get("attention_mask", None) is None:
            # init `attention_mask` depending on `pad_token_id`
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                input_ids, pad_token_id, eos_token_id
            )

        # special case if pad_token_id is not defined
        if pad_token_id is None and eos_token_id is not None:
            logger.warning(
                f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            pad_token_id = eos_token_id

        # Storing encoder_input_ids for logits_processor that could use them
        encoder_input_ids = input_ids if self.model.config.is_encoder_decoder else None

        if self.model.config.is_encoder_decoder:
            # add encoder_outputs to model_kwargs
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                input_ids, model_kwargs)

            # set input_ids as decoder_input_ids
            if "decoder_input_ids" in model_kwargs:
                input_ids = model_kwargs.pop("decoder_input_ids")
            else:
                input_ids = self._prepare_decoder_input_ids_for_generation(
                    input_ids, decoder_start_token_id=decoder_start_token_id, bos_token_id=bos_token_id
                )

            if "encoder_outputs" not in model_kwargs or not isinstance(model_kwargs["encoder_outputs"], ModelOutput):
                raise ValueError(
                    "Make sure that `model_kwargs` include `encoder_outputs` of type `ModelOutput`.")

        if input_ids.shape[-1] >= max_length:
            input_ids_string = "decoder_input_ids" if self.model.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids.shape[-1]}, but ``max_length`` is set to {max_length}."
                "This can lead to unexpected behavior. You should consider increasing ``config.max_length`` or ``max_length``."
            )

        # determine generation mode
        is_group_beam_gen_mode = (num_beams > 1) and (num_beam_groups > 1)
        if num_beam_groups > num_beams:
            raise ValueError(
                "`num_beam_groups` has to be smaller or equal to `num_beams`")
        if is_group_beam_gen_mode and do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
            )

        # set model_kwargs
        model_kwargs["use_cache"] = use_cache

        # get distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            encoder_input_ids=encoder_input_ids,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
        )

        stopping_criteria = self._get_stopping_criteria(
            max_length=max_length,
            max_time=max_time,
        )

        # elif is_beam_gen_mode:
        batch_size = input_ids.shape[0]

        length_penalty = length_penalty if length_penalty is not None else self.model.config.length_penalty
        early_stopping = early_stopping if early_stopping is not None else self.model.config.early_stopping

        if num_return_sequences > num_beams:
            raise ValueError(
                "`num_return_sequences` has to be smaller or equal to `num_beams`.")

        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            max_length=max_length,
            num_beams=num_beams,
            device=self.device,
            length_penalty=length_penalty,
            do_early_stopping=early_stopping,
            num_beam_hyps_to_keep=num_return_sequences,
        )
        # interleave with `num_beams`
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids, expand_size=num_beams, is_encoder_decoder=self.model.config.is_encoder_decoder, **model_kwargs
        )
        return self.beam_search_hf(
            input_ids,
            beam_scorer,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            max_length=max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            gen_type=gen_type,
            **model_kwargs,
        )

    # def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs) -> Dict[str, Any]:
    #     """
    #     Implement in subclasses of :class:`~transformers.PreTrainedModel` for custom behavior to prepare inputs in the
    #     generate method.
    #     """
    #     return {"input_ids": input_ids}

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            # change this to avoid caching (presumably for debugging)
            "use_cache": use_cache,
        }

    def adjust_logits_during_generation(self, logits: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        """
        Implement in subclasses of :class:`~transformers.PreTrainedModel` for custom behavior to adjust the logits in
        the generate method.
        """
        return logits

    @staticmethod
    def _update_model_kwargs_for_generation(
        outputs: ModelOutput, model_kwargs: Dict[str, Any], is_encoder_decoder: bool = False
    ) -> Dict[str, Any]:
        # update past
        if "past_key_values" in outputs:
            model_kwargs["past"] = outputs.past_key_values
        elif "mems" in outputs:
            model_kwargs["past"] = outputs.mems
        elif "past_buckets_states" in outputs:
            model_kwargs["past"] = outputs.past_buckets_states
        else:
            model_kwargs["past"] = None

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # update attention mask
        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        return model_kwargs

    # def _reorder_cache(self, past, beam_idx):
    #     raise NotImplementedError(
    #         f"Make sure that a `_reorder_cache` function is correctly implemented in {self.__class__.__module__} to enable beam search for {self.__class__}"
    #     )

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx)
                      for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

    def beam_search_hf(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        gen_type="zm",
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        max_length = max_length if max_length is not None else self.model.config.max_length
        validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.model.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.model.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.model.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.model.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (
            return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.model.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get(
                "attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get(
                    "hidden_states") if output_hidden_states else None
            )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        assert (
            num_beams * batch_size == batch_beam_size
        ), "Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."

        beam_scores = torch.zeros(
            (batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, **model_kwargs)

            # single模型的得到next token方法。得到outputs的时候不能加is_train这个参数。
            # 因为是针对dec2模型的参数，在modeling_bart中修改了源码，实际transformers库中没有is_train这个参数
            if gen_type=="single":
                outputs = self(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
                next_token_logits = outputs.logits[:, -1, :]
            else:
                outputs = self(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    is_train=False,
                )
                if gen_type=="zm":
                    next_token_logits = outputs.logits[0][:, -1, :]
                elif gen_type=="xq":
                    next_token_logits = outputs.logits[1][:, -1, :]

            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `F.log_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(
                next_token_logits, cur_len=cur_len, max_length=max_length
            )

            # (batch_size * num_beams, vocab_size)
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)

            next_token_scores = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores + \
                beam_scores[:, None].expand_as(next_token_scores)

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(
                batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat(
                [input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            cur_len = cur_len + 1

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.model.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(
                    model_kwargs["past"], beam_idx)

            if beam_scorer.is_done:
                break

            if stopping_criteria(input_ids, scores):
                break

        sequence_outputs = beam_scorer.finalize(
            input_ids, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id, max_length=max_length
        )

        return sequence_outputs["sequences"]

    def beam_search(self, token_ids, token_type_ids, beam_size=1, device="cpu"):
        """
        beam-search操作
        """

        # 用来保存输出序列
        output_ids = torch.empty(1, 0, device=device, dtype=torch.long)
        # 用来保存累计得分

        with torch.no_grad():
            output_scores = torch.zeros(token_ids.shape[0], device=device)
            for step in range(self.out_max_length):
                if step == 0:
                    scores = self.model(token_ids, token_type_ids)
                    # 重复beam-size次 输入ids
                    token_ids = token_ids.view(1, -1).repeat(beam_size, 1)
                    token_type_ids = token_type_ids.view(
                        1, -1).repeat(beam_size, 1)
                else:
                    scores = self.model(new_input_ids, new_token_type_ids)

                logit_score = torch.log_softmax(scores[:, -1], dim=-1)

                logit_score = output_scores.view(-1, 1) + logit_score  # 累计得分
                # 取topk的时候我们是展平了然后再去调用topk函数
                # 展平
                logit_score = logit_score.view(-1)
                hype_score, hype_pos = torch.topk(logit_score, beam_size)
                indice1 = (hype_pos // scores.shape[-1])  # 行索引
                indice2 = (hype_pos %
                           scores.shape[-1]).long().reshape(-1, 1)  # 列索引

                # 更新得分
                output_scores = hype_score
                output_ids = torch.cat(
                    [output_ids[indice1], indice2], dim=1).long()
                new_input_ids = torch.cat([token_ids, output_ids], dim=1)
                new_token_type_ids = torch.cat(
                    [token_type_ids, torch.ones_like(output_ids)], dim=1)

                end_counts = (output_ids == self.eos_id).sum(1)  # 统计出现的end标记
                best_one = output_scores.argmax()
                if end_counts[best_one] == 1:
                    # 说明出现终止了～
                    return output_ids[best_one][:-1]
                else:
                    # 保留未完成部分
                    flag = (end_counts < 1)  # 标记未完成序列
                    if not flag.all():  # 如果有已完成的
                        token_ids = token_ids[flag]
                        token_type_ids = token_type_ids[flag]
                        new_input_ids = new_input_ids[flag]
                        new_token_type_ids = new_token_type_ids[flag]
                        output_ids = output_ids[flag]  # 扔掉已完成序列
                        output_scores = output_scores[flag]  # 扔掉已完成序列
                        end_counts = end_counts[flag]  # 扔掉已完成end计数
                        beam_size = flag.sum()  # topk相应变化

            return output_ids[output_scores.argmax()]
