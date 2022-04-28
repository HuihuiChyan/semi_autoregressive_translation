import torch
from fairseq.data import data_utils
from fairseq.data.language_pair_dataset import LanguagePairDataset
import pdb

class HuiLanguagePairDataset(LanguagePairDataset):

	def __init__(
		self,
		src,
		src_sizes,
		src_dict,
		tgt,
		tgt_sizes,
		tgt_dict,
		shuffle=True,
		input_feeding=True,
		chunk_size=5,
	):
		super(HuiLanguagePairDataset, self).__init__(
			src,
			src_sizes,
			src_dict,
			tgt,
			tgt_sizes,
			tgt_dict,
			shuffle=True,
			input_feeding=True,
		)
		self.chunk_size = chunk_size
	
	def __getitem__(self, index):
		tgt_item = self.tgt[index]
		src_item = self.src[index]
		example = {"id": index, "source": src_item, "target": tgt_item}
		return example

	def collater(self, samples):

		pad_idx=self.src_dict.pad()
		eos_idx=self.eos

		left_pad = False
		# left_pad仅仅对于LSTM非常重要，对于Transformer完全不重要，所以我们统一为False

		if len(samples) == 0:
			return {}

		def hui_collate_tokens(values, pad_idx, eos_idx, left_pad=False, move_eos_to_beginning=False, chunk_size=5):
			# 由于是半自回归的机器翻译，所以输入前补了chunk_size个eos_idx
			size = max(v.size(0) for v in values) + chunk_size - 1
			if chunk_size != 1:
				size = ((size - 1) // chunk_size + 1) * chunk_size

			batch_size = len(values)
			res = values[0].new(batch_size, size).fill_(pad_idx)

			for index, value in enumerate(values):
				if move_eos_to_beginning:
					for i in range(chunk_size):
						res[index][i] = eos_idx
					res[index][chunk_size:chunk_size+len(value[:-1])] = value[:-1]
				else:
					res[index][:len(value)].copy_(value)

			return res

		id = torch.LongTensor([s["id"] for s in samples])
		src_tokens = data_utils.collate_tokens([s["source"] for s in samples], pad_idx, eos_idx, left_pad,
											   move_eos_to_beginning=False)
		src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])
		src_lengths, sort_order = src_lengths.sort(descending=True)
		id = id.index_select(0, sort_order)
		src_tokens = src_tokens.index_select(0, sort_order)

		prev_output_tokens = None
		target = None

		target =  hui_collate_tokens([s["target"] for s in samples], pad_idx, eos_idx, left_pad,
									 move_eos_to_beginning=False, chunk_size=self.chunk_size)
		target = target.index_select(0, sort_order)
		tgt_lengths = torch.LongTensor([s["target"].ne(pad_idx).long().sum() for s in samples]).index_select(0, sort_order)
		ntokens = tgt_lengths.sum().item()

		if self.input_feeding:
			# move_eos_to_beginning仅仅用于teacher forcing
			prev_output_tokens =  hui_collate_tokens([s["target"] for s in samples], pad_idx, eos_idx, left_pad,
													 move_eos_to_beginning=True, chunk_size=self.chunk_size)
				
		batch = {
			"id": id,
			"nsentences": len(samples),
			"ntokens": ntokens,
			"net_input":{
				"src_tokens": src_tokens,
				"src_lengths": src_lengths,
			},
			"target": target,
		}

		if prev_output_tokens is not None:
			batch["net_input"]["prev_output_tokens"] = prev_output_tokens.index_select(0, sort_order)
	
		return batch