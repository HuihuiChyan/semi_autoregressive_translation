import os
import random
import logging
import fairseq
import itertools

from fairseq.tasks import register_task
from fairseq.criterions import register_criterion
from fairseq.tasks.translation import TranslationTask
from fairseq.data import data_utils, indexed_dataset, ConcatDataset
from user_dir.hui_sequence_generator_hui import HuiSequenceGenerator
from user_dir.hui_language_pair_dataset import HuiLanguagePairDataset
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion

import pdb

logger = logging.getLogger(__name__)

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
	if target.dim() == lprobs.dim() - 1:
		target = target.unsqueeze(-1)
	nll_loss = -lprobs.gather(dim=-1, index=target)
	smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
	if ignore_index is not None:
		pad_mask = target.eq(ignore_index)
		nll_loss.masked_fill_(pad_mask, 0.0)
		smooth_loss.masked_fill_(pad_mask, 0.0)
	else:
		nll_loss = nll_loss.squeeze(-1)
		smooth_loss = smooth_loss.squeeze(-1)
	if reduce:
		nll_loss = nll_loss.sum()
		smooth_loss = smooth_loss.sum()
	eps_i = epsilon / (lprobs.size(-1) - 1)
	loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
	return loss, nll_loss

@register_criterion('hui_translation_criterion')
class HuiTranslationCriterion(LabelSmoothedCrossEntropyCriterion):

	def get_lprobs_and_target(self, model, net_output, sample):
		lprobs = model.get_normalized_probs(net_output, log_probs=True)
		target = model.get_targets(sample, net_output)
		if self.ignore_prefix_size > 0:
			if getattr(lprobs, "batch_first", False):
				lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
				target = target[:, self.ignore_prefix_size :].contiguous()
			else:
				lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
				target = target[self.ignore_prefix_size :, :].contiguous()
		pdb.set_trace()
		return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

	def compute_loss(self, model, net_output, sample, reduce=True):
		lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
		loss, nll_loss = label_smoothed_nll_loss(
			lprobs,
			target,
			self.eps,
			ignore_index=self.padding_idx,
			reduce=reduce,
		)
		return loss, nll_loss

	def forward(self, model, sample, reduce=True):

		net_output = model(**sample["net_input"])

		loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

		sample_size = sample["ntokens"]
		
		logging_output = {
			"loss": loss.data,
			"nll_loss": nll_loss.data,
			"ntokens": sample["ntokens"],
			"nsentences": sample["target"].size(0),
			"sample_size": sample_size,
		}
		# print(loss.data/sample["ntokens"])

		return loss, sample_size, logging_output

@register_task("hui_translation_task")
class HuiTranslationTask(TranslationTask):

	@staticmethod
	def add_args(parser):
		parser.add_argument("--chunk-size", type=int, default=1)
		parser.add_argument("--beam-size", type=int, default=5)
		parser.add_argument("--initialize-encoder-and-embedings", action="store_true", default=False)
		TranslationTask.add_args(parser)

	def load_dataset(self, split, epoch=1, **kwargs):
		paths = fairseq.utils.split_paths(self.args.data)
		data_path = paths[(epoch-1) % len(paths)]
		src, tgt = self.args.source_lang, self.args.target_lang
		src_dict, tgt_dict = self.src_dict, self.tgt_dict
		
		def split_exists(split, src, tgt, lang, data_path):
			filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
			return indexed_dataset.dataset_exists(filename, impl=self.args.dataset_impl)

		src_datasets = []
		tgt_datasets = []

		for k in itertools.count():
			split_k = split + (str(k) if k > 0 else "")

			if split_exists(split_k, src, tgt, src, data_path):
				prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
			elif k > 0:
				break
			else:
				raise FileNotFoundError("Dataset not found: {} ({})".format(split, data_path))

			src_datasets.append(data_utils.load_indexed_dataset(prefix + src, src_dict, self.args.dataset_impl))
			tgt_datasets.append(data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, self.args.dataset_impl))

			logger.info("{} {} {}-{} {} examples".format(data_path, split_k, src, tgt, len(src_datasets[-1])))

		assert len(src_datasets) == len(tgt_datasets)

		if len(src_datasets) == 1:
			src_dataset = src_datasets[0]
			tgt_dataset = tgt_datasets[0]
		else:
			sample_ratios = [1] * len(src_datasets)
			src_dataset = ConcatDataset(src_datasets, sample_ratios)
			tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

		self.datasets[split] = HuiLanguagePairDataset(src_dataset,
													  src_dataset.sizes,
													  src_dict,
													  tgt_dataset,
													  tgt_dataset.sizes,
													  tgt_dict,
													  shuffle=(split == "train"),
													  chunk_size=self.args.chunk_size)

	def build_generator(self, models, args):
		if getattr(args, "score_reference", False):
			from fairseq.sequence_scorer import SequenceScorer
			return SequenceScorer(self.target_dictionary, compute_alignment=False)

		# return fairseq.sequence_generator.SequenceGenerator(models, self.target_dictionary, beam_size=getattr(args, 'beam_size', 5))
		return HuiSequenceGenerator(models, self.target_dictionary, beam_size=self.args.beam_size, chunk_size=self.args.chunk_size)