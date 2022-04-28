import math
import torch
import fairseq
from torch import Tensor
from typing import Dict, List, Optional
import pdb

class HuiBeamSearch(fairseq.search.BeamSearch):

	def __init__(self, tgt_dict):
		super().__init__(tgt_dict)

	def step(self, step: int, lprobs, scores: Optional[Tensor]):
		bsz, beam_size, vocab_size = lprobs.size()

		if step == 0:
			# at the first step all hypotheses are equally likely, so use
			# only the first beam
			lprobs = lprobs[:, ::beam_size, :].contiguous()
		else:
			# make probs contain cumulative scores for each hypothesis
			assert scores is not None
			lprobs = lprobs + scores[:, :, step - 1].unsqueeze(-1)

		# Take the best 2 x beam_size predictions. We'll choose the first
		# beam_size of these which don't predict eos to continue with.
		top_prediction = torch.topk(lprobs.view(bsz, -1), k=beam_size * 2)

		scores_buf = top_prediction[0]
		indices_buf = top_prediction[1]
		# Project back into relative indices and beams
		beams_buf = indices_buf // vocab_size
		indices_buf = indices_buf.fmod(vocab_size)

		# At this point, beams_buf and indices_buf are single-dim and contain relative indices
		return scores_buf, indices_buf, beams_buf

class HuiEnsembleModel(fairseq.sequence_generator.EnsembleModel):
	"""A wrapper around an ensemble of models."""

	@torch.jit.export
	def forward_decoder(
		self,
		tokens,
		encoder_outs: List[Dict[str, List[Tensor]]],
		incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
		temperature: float = 1.0,
		chunk_size=2,
	):
		log_probs = []
		avg_attn: Optional[Tensor] = None
		encoder_out: Optional[Dict[str, List[Tensor]]] = None
		for i, model in enumerate(self.models):
			if self.has_encoder():
				encoder_out = encoder_outs[i]
			# decode each model
			if self.has_incremental_states():
				decoder_out = model.decoder.forward(
					tokens,
					encoder_out=encoder_out,
					incremental_state=incremental_states[i],
				)
			else:
				if hasattr(model, "decoder"):
					decoder_out = model.decoder.forward(tokens, encoder_out=encoder_out)
				else:
					decoder_out = model.forward(tokens)
			attn: Optional[Tensor] = None
			decoder_len = len(decoder_out)
			if decoder_len > 1 and decoder_out[1] is not None:
				if isinstance(decoder_out[1], Tensor):
					attn = decoder_out[1]
				else:
					attn_holder = decoder_out[1]["attn"]
					if isinstance(attn_holder, Tensor):
						attn = attn_holder
					elif attn_holder is not None:
						attn = attn_holder[0]
				if attn is not None:
					attn = attn[:, -chunk_size:, :]
			decoder_out_tuple = (
				decoder_out[0][:, -chunk_size:, :].div_(temperature),
				None if decoder_len <= 1 else decoder_out[1],
			)
			probs = model.get_normalized_probs(
				decoder_out_tuple, log_probs=True, sample=None
			)
			probs = probs[:, -chunk_size:, :]
			if self.models_size == 1:
				return probs, attn

			log_probs.append(probs)
			if attn is not None:
				if avg_attn is None:
					avg_attn = attn
				else:
					avg_attn.add_(attn)
		avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(
			self.models_size
		)

		if avg_attn is not None:
			avg_attn.div_(self.models_size)
		
		return avg_probs, avg_attn

class HuiSequenceGenerator(fairseq.sequence_generator.SequenceGenerator):

	def __init__(self, models, tgt_dict, beam_size=5, max_len_a=0, 
		max_len_b=200, chunk_size=2, **kwargs):
		super(HuiSequenceGenerator, self).__init__(models, tgt_dict, beam_size, max_len_a=max_len_a, max_len_b=max_len_b, **kwargs)
		self.search = fairseq.search.BeamSearch(tgt_dict)
		self.model = HuiEnsembleModel(models)
		self.chunk_size = chunk_size

	def _generate(
		self,
		sample: Dict[str, Dict[str, Tensor]],
		prefix_tokens: Optional[Tensor] = None,
		constraints: Optional[Tensor] = None,
		bos_token: Optional[int] = None,
	):

		incremental_states = torch.jit.annotate(
			List[Dict[str, Dict[str, Optional[Tensor]]]],
			[
				torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
				for i in range(self.model.models_size)
			],
		)
		net_input = sample["net_input"]

		if "src_tokens" in net_input:
			src_tokens = net_input["src_tokens"]
			# length of the source text being the character length except EndOfSentence and pad
			src_lengths = (
				(src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
			)
		elif "source" in net_input:
			src_tokens = net_input["source"]
			src_lengths = (
				net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
				if net_input["padding_mask"] is not None
				else torch.tensor(src_tokens.size(-1)).to(src_tokens)
			)
		elif "features" in net_input:
			src_tokens = net_input["features"]
			src_lengths = (
				net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
				if net_input["padding_mask"] is not None
				else torch.tensor(src_tokens.size(-1)).to(src_tokens)
			)
		else:
			raise Exception("expected src_tokens or source in net input. input keys: " + str(net_input.keys()))

		# bsz: total number of sentences in beam
		# Note that src_tokens may have more than 2 dimensions (i.e. audio features)
		bsz, src_len = src_tokens.size()[:2]
		beam_size = self.beam_size

		if constraints is not None and not self.search.supports_constraints:
			raise NotImplementedError(
				"Target-side constraints were provided, but search method doesn't support them"
			)

		# Initialize constraints, when active
		self.search.init_constraints(constraints, beam_size)

		max_len: int = -1
		if self.match_source_len:
			max_len = src_lengths.max().item()
		else:
			max_len = int(self.max_len_a * src_len + self.max_len_b)
		assert (
			self.min_len <= max_len
		), "min_len cannot be larger than max_len, please adjust these!"
		# compute the encoder output for each beam
		with torch.autograd.profiler.record_function("EnsembleModel: forward_encoder"):
			encoder_outs = self.model.forward_encoder(net_input)

		# placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
		new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
		new_order = new_order.to(src_tokens.device).long()
		encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)
		# ensure encoder_outs is a List.
		assert encoder_outs is not None

		# initialize buffers
		scores = (
			torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
		)  # +1 for eos; pad is never chosen for scoring
		tokens = (
			torch.zeros(bsz * beam_size, max_len + 1 + self.chunk_size)
			.to(src_tokens)
			.long()
			.fill_(self.pad)
		)
		tokens[:, :self.chunk_size] = self.eos if bos_token is None else bos_token
		attn: Optional[Tensor] = None

		# A list that indicates candidates that should be ignored.
		# For example, suppose we're sampling and have already finalized 2/5
		# samples. Then cands_to_ignore would mark 2 positions as being ignored,
		# so that we only finalize the remaining 3 samples.
		cands_to_ignore = (
			torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
		)  # forward and backward-compatible False mask

		# list of completed sentences
		finalized = torch.jit.annotate(
			List[List[Dict[str, Tensor]]],
			[torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
		)  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

		# a boolean array indicating if the sentence at the index is finished or not
		finished = [False for i in range(bsz)]
		num_remaining_sent = bsz  # number of sentences remaining

		# number of candidate hypos per step
		cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

		# offset arrays for converting between different indexing schemes
		bbsz_offsets = (
			(torch.arange(0, bsz) * beam_size)
			.unsqueeze(1)
			.type_as(tokens)
			.to(src_tokens.device)
		)
		cand_offsets = torch.arange(0, cand_size).type_as(tokens).to(src_tokens.device)

		reorder_state: Optional[Tensor] = None
		batch_idxs: Optional[Tensor] = None

		original_batch_idxs: Optional[Tensor] = None
		if "id" in sample and isinstance(sample["id"], Tensor):
			original_batch_idxs = sample["id"]
		else:
			original_batch_idxs = torch.arange(0, bsz).type_as(tokens)

		for step in range(max_len + 1):  # one extra step for EOS marker
			# reorder decoder internal states based on the prev choice of beams
			
			if reorder_state is not None:
				if batch_idxs is not None:
					# update beam indices to take into account removed sentences
					corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
						batch_idxs
					)
					reorder_state.view(-1, beam_size).add_(
						corr.unsqueeze(-1) * beam_size
					)
					original_batch_idxs = original_batch_idxs[batch_idxs]
				self.model.reorder_incremental_state(incremental_states, reorder_state)
				encoder_outs = self.model.reorder_encoder_out(
					encoder_outs, reorder_state
				)
			# bos : 0
			# pad : 1
			# eos : 2
			# unk : 3
			if step % self.chunk_size == 0:
				with torch.autograd.profiler.record_function("EnsembleModelNAT: forward_decoder"):
					lprobs_all, avg_attn_scores_all = self.model.forward_decoder(
						tokens[:, : step + self.chunk_size],
						encoder_outs,
						incremental_states,
						self.temperature,
						chunk_size=self.chunk_size,
					)
			lprobs = lprobs_all[:, step % self.chunk_size]
			avg_attn_scores = None

			
			lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

			lprobs[:, self.pad] = -math.inf  # never select pad
			lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

			# handle max length constraint
			if step >= max_len:
				lprobs[:, : self.eos] = -math.inf
				lprobs[:, self.eos + 1 :] = -math.inf

			# Record attention scores, only support avg_attn_scores is a Tensor
			if avg_attn_scores is not None:
				if attn is None:
					attn = torch.empty(
						bsz * beam_size, avg_attn_scores.size(1), max_len + 1 + self.chunk_size
					).to(scores)
				attn[:, :, step + 1].copy_(avg_attn_scores)
			scores = scores.type_as(lprobs)
			eos_bbsz_idx = torch.empty(0).to(
				tokens
			)  # indices of hypothesis ending with eos (finished sentences)
			eos_scores = torch.empty(0).to(
				scores
			)  # scores of hypothesis ending with eos (finished sentences)

			if self.should_set_src_lengths:
				self.search.set_src_lengths(src_lengths)
			
			# Shape: (batch, cand_size)
			cand_scores, cand_indices, cand_beams = self.search.step(
				step,
				lprobs.view(bsz, -1, self.vocab_size),
				scores.view(bsz, beam_size, -1)[:, :, :step],
				tokens[:, : step + self.chunk_size],
				original_batch_idxs,
			)
			# cand_bbsz_idx contains beam indices for the top candidate
			# hypotheses, with a range of values: [0, bsz*beam_size),
			# and dimensions: [bsz, cand_size]
			cand_bbsz_idx = cand_beams.add(bbsz_offsets)

			# finalize hypotheses that end in eos
			# Shape of eos_mask: (batch size, beam size)
			eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
			eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)

			# only consider eos when it's among the top beam_size indices
			# Now we know what beam item(s) to finish
			# Shape: 1d list of absolute-numbered
			eos_bbsz_idx = torch.masked_select(
				cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
			)
			finalized_sents: List[int] = []
			if eos_bbsz_idx.numel() > 0:
				eos_scores = torch.masked_select(
					cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
				)

				finalized_sents = self.finalize_hypos(
					step,
					eos_bbsz_idx,
					eos_scores,
					tokens,
					scores,
					finalized,
					finished,
					beam_size,
					attn,
					src_lengths,
					max_len,
				)
				num_remaining_sent -= len(finalized_sents)
			
			assert num_remaining_sent >= 0
			if num_remaining_sent == 0:
				break
			if self.search.stop_on_max_len and step >= max_len:
				break
			assert step < max_len, f"{step} < {max_len}"

			# Remove finalized sentences (ones for which {beam_size}
			# finished hypotheses have been generated) from the batch.
			if len(finalized_sents) > 0:
				new_bsz = bsz - len(finalized_sents)

				# construct batch_idxs which holds indices of batches to keep for the next pass
				batch_mask = torch.ones(
					bsz, dtype=torch.bool, device=cand_indices.device
				)
				batch_mask[finalized_sents] = False
				# TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
				batch_idxs = torch.arange(
					bsz, device=cand_indices.device
				).masked_select(batch_mask)

				# Choose the subset of the hypothesized constraints that will continue
				self.search.prune_sentences(batch_idxs)

				eos_mask = eos_mask[batch_idxs]
				cand_beams = cand_beams[batch_idxs]
				bbsz_offsets.resize_(new_bsz, 1)
				cand_bbsz_idx = cand_beams.add(bbsz_offsets)
				cand_scores = cand_scores[batch_idxs]
				cand_indices = cand_indices[batch_idxs]


				if prefix_tokens is not None:
					prefix_tokens = prefix_tokens[batch_idxs]
				src_lengths = src_lengths[batch_idxs]
				cands_to_ignore = cands_to_ignore[batch_idxs]

				scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
				tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
				if attn is not None:
					attn = attn.view(bsz, -1)[batch_idxs].view(
						new_bsz * beam_size, attn.size(1), -1
					)
					
				lprobs_all = lprobs_all.view(bsz, -1)[batch_idxs].view(
					new_bsz * beam_size, self.chunk_size, -1
				)
				avg_attn_scores_all = None


				bsz = new_bsz
			else:
				batch_idxs = None
			
			# Set active_mask so that values > cand_size indicate eos hypos
			# and values < cand_size indicate candidate active hypos.
			# After, the min values per row are the top candidate active hypos

			# Rewrite the operator since the element wise or is not supported in torchscript.

			eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
			active_mask = torch.add(
				eos_mask.type_as(cand_offsets) * cand_size,
				cand_offsets[: eos_mask.size(1)],
			)

			# get the top beam_size active hypotheses, which are just
			# the hypos with the smallest values in active_mask.
			# {active_hypos} indicates which {beam_size} hypotheses
			# from the list of {2 * beam_size} candidates were
			# selected. Shapes: (batch size, beam size)
			new_cands_to_ignore, active_hypos = torch.topk(
				active_mask, k=beam_size, dim=1, largest=False
			)

			# update cands_to_ignore to ignore any finalized hypos.
			cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
			# Make sure there is at least one active item for each sentence in the batch.
			assert (~cands_to_ignore).any(dim=1).all()

			# update cands_to_ignore to ignore any finalized hypos

			# {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
			# can be selected more than once).
			active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
			active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

			active_bbsz_idx = active_bbsz_idx.view(-1)
			active_scores = active_scores.view(-1)

			# copy tokens and scores for active hypotheses

			# Set the tokens for each beam (can select the same row more than once)
			tokens[:, : step + self.chunk_size] = torch.index_select(
				tokens[:, : step + self.chunk_size], dim=0, index=active_bbsz_idx
			)
			# Select the next token for each of them
			tokens.view(bsz, beam_size, -1)[:, :, step + self.chunk_size] = torch.gather(
				cand_indices, dim=1, index=active_hypos
			)
			if step > 0:
				scores[:, :step] = torch.index_select(
					scores[:, :step], dim=0, index=active_bbsz_idx
				)
			scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
				cand_scores, dim=1, index=active_hypos
			)

			# Update constraints based on which candidates were selected for the next beam
			self.search.update_constraints(active_hypos)
			# copy attention for active hypotheses
			if attn is not None:
				attn[:, :, : step + 1 + self.chunk_size] = torch.index_select(
					attn[:, :, : step + 1 + self.chunk_size], dim=0, index=active_bbsz_idx
				)

			# reorder incremental state in decoder
			reorder_state = active_bbsz_idx

		# sort by score descending
		for sent in range(len(finalized)):
			scores = torch.tensor(
				[float(elem["score"].item()) for elem in finalized[sent]]
			)
			_, sorted_scores_indices = torch.sort(scores, descending=True)
			finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
			finalized[sent] = torch.jit.annotate(
				List[Dict[str, Tensor]], finalized[sent]
			)
		return finalized

	def finalize_hypos(
		self,
		step: int,
		bbsz_idx,
		eos_scores,
		tokens,
		scores,
		finalized: List[List[Dict[str, Tensor]]],
		finished: List[bool],
		beam_size: int,
		attn: Optional[Tensor],
		src_lengths,
		max_len: int,
	):
		"""Finalize hypothesis, store finalized information in `finalized`, and change `finished` accordingly.
		A sentence is finalized when {beam_size} finished items have been collected for it.

		Returns number of sentences (not beam items) being finalized.
		These will be removed from the batch and not processed further.
		Args:
			bbsz_idx (Tensor):
		"""
		assert bbsz_idx.numel() == eos_scores.numel()

		# clone relevant token and attention tensors.
		# tokens is (batch * beam, max_len). So the index_select
		# gets the newly EOS rows, then selects cols 1..{step + 2}
		tokens_clone = tokens.index_select(0, bbsz_idx)[
			:, self.chunk_size : step + 1 + self.chunk_size
		]  # skip the first index, which is EOS
		tokens_clone[:, step] = self.eos
		attn_clone = (
			attn.index_select(0, bbsz_idx)[:, :, self.chunk_size : step + 1 + self.chunk_size]
			if attn is not None
			else None
		)

		# compute scores per token position
		pos_scores = scores.index_select(0, bbsz_idx)[:, : step + 1 + self.chunk_size]
		pos_scores[:, step] = eos_scores
		# convert from cumulative to per-position scores
		pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

		# normalize sentence-level scores
		if self.normalize_scores:
			eos_scores /= (step + 1) ** self.len_penalty

		# cum_unfin records which sentences in the batch are finished.
		# It helps match indexing between (a) the original sentences
		# in the batch and (b) the current, possibly-reduced set of
		# sentences.
		cum_unfin: List[int] = []
		prev = 0
		for f in finished:
			if f:
				prev += 1
			else:
				cum_unfin.append(prev)
		cum_fin_tensor = torch.tensor(cum_unfin, dtype=torch.int).to(bbsz_idx)

		unfin_idx = bbsz_idx // beam_size
		sent = unfin_idx + torch.index_select(cum_fin_tensor, 0, unfin_idx)

		# Create a set of "{sent}{unfin_idx}", where
		# "unfin_idx" is the index in the current (possibly reduced)
		# list of sentences, and "sent" is the index in the original,
		# unreduced batch
		# For every finished beam item
		# sentence index in the current (possibly reduced) batch
		seen = (sent << 32) + unfin_idx
		unique_seen: List[int] = torch.unique(seen).tolist()

		if self.match_source_len:
			condition = step > torch.index_select(src_lengths, 0, unfin_idx)
			eos_scores = torch.where(condition, torch.tensor(-math.inf), eos_scores)
		sent_list: List[int] = sent.tolist()
		for i in range(bbsz_idx.size()[0]):
			# An input sentence (among those in a batch) is finished when
			# beam_size hypotheses have been collected for it
			if len(finalized[sent_list[i]]) < beam_size:
				if attn_clone is not None:
					# remove padding tokens from attn scores
					hypo_attn = attn_clone[i]
				else:
					hypo_attn = torch.empty(0)

				finalized[sent_list[i]].append(
					{
						"tokens": tokens_clone[i],
						"score": eos_scores[i],
						"attention": hypo_attn,  # src_len x tgt_len
						"alignment": torch.empty(0),
						"positional_scores": pos_scores[i],
					}
				)

		newly_finished: List[int] = []
		for unique_s in unique_seen:
			# check termination conditions for this sentence
			unique_sent: int = unique_s >> 32
			unique_unfin_idx: int = unique_s - (unique_sent << 32)

			if not finished[unique_sent] and self.is_finished(
				step, unique_unfin_idx, max_len, len(finalized[unique_sent]), beam_size
			):
				finished[unique_sent] = True
				newly_finished.append(unique_unfin_idx)

		return newly_finished 