import torch
import fairseq
from torch import nn, Tensor
from typing import Hashable, Optional, Dict, List
from fairseq.models import register_model, register_model_architecture
from fairseq.modules import TransformerDecoderLayer, MultiheadAttention, SinusoidalPositionalEmbedding
from fairseq.models.transformer import TransformerModel, TransformerEncoder, TransformerDecoder, base_architecture
import pdb

# class HuiDecoderLayer(TransformerDecoderLayer):

# 	def __init__(self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False):
# 		super().__init__(
# 			args=args,
# 			no_encoder_attn=no_encoder_attn,
# 			add_bias_kv=add_bias_kv,
# 			add_zero_attn=add_zero_attn,
# 		)

# 	def forward(
# 		self,
# 		x,
# 		encoder_out,
# 		encoder_padding_mask,
# 		incremental_state,
# 		prev_self_attn_state = None,
# 		prev_attn_state = None,
# 		self_attn_mask = None,
# 		self_attn_padding_mask = None,
# 	):
# 		residual = x

# 		y = x
# 		x, attn = self.self_attn(
# 			query=x,
# 			key=y,
# 			value=y,
# 			key_padding_mask=self_attn_padding_mask,
# 			incremental_state=incremental_state,
# 			need_weights=False,
# 			attn_mask=self_attn_mask,
# 		)
# 		x = self.dropout_module(x)
# 		x = self.residual_connection(x, residual)
# 		x = self.self_attn_layer_norm(x)

# 		residual = x

# 		x, attn = self.encoder_attn(
# 			query=x,
# 			key=encoder_out,
# 			value=encoder_out,
# 			key_padding_mask=encoder_padding_mask,
# 			incremental_state=incremental_state,
# 			static_kv=True,
# 			need_weights=False,
# 			need_head_weights=False,
# 		)
# 		x = self.dropout_module(x)
# 		x = self.residual_connection(x, residual)
# 		x = self.encoder_attn_layer_norm(x)

# 		residual = x

# 		x = self.activation_fn(self.fc1(x))
# 		x = self.activation_dropout_module(x)
# 		x = self.fc2(x)
# 		x = self.dropout_module(x)
# 		x = self.residual_connection(x, residual)
# 		x = self.final_layer_norm(x)

# 		return x, attn, None

# class HuiEncoder(TransformerEncoder):

# 	def __init__(self, args, dictionary, embed_tokens):
# 		super().__init__(args, dictionary, embed_tokens)
# 		if args.initialize_encoder_and_embedings:
# 			state_dict = self.state_dict()
# 			state = fairseq.checkpoint_utils.load_checkpoint_to_cpu("/data1/huanghui/SANMT/checkpoints/en2zh02/checkpoint_best.pt")
# 			for key in state["model"].keys():
# 				if "encoder." in key:
# 					subkey = key[len("encoder."):]
# 					assert subkey in state_dict
# 					state_dict[subkey] = state["model"][key]
# 			self.load_state_dict(state_dict, strict=True)

class HuiSinusoidalPositionalEmbedding(SinusoidalPositionalEmbedding):

	def forward(
		self,
		input,
		incremental_state,
		chunk_size,
		timestep=None,
		positions=None):
		"""Input is expected to be of size [bsz x seqlen]."""
		bspair = torch.onnx.operators.shape_as_tensor(input)
		bsz, seq_len = bspair[0], bspair[1]
		max_pos = self.padding_idx + 1 + seq_len
		if self.weights is None or max_pos > self.weights.size(0):
			# recompute/expand embeddings if needed
			self.weights = SinusoidalPositionalEmbedding.get_embedding(
				max_pos, self.embedding_dim, self.padding_idx
			)
		self.weights = self.weights.to(self._float_tensor)

		if incremental_state is not None:
			# positions is the same for every token when decoding a single step
			pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
			start = self.padding_idx + pos - chunk_size + 1
			return self.weights[start : start + chunk_size, :].expand(bsz, chunk_size, -1)

		positions = fairseq.utils.make_positions(
			input, self.padding_idx, onnx_trace=self.onnx_trace
		)

		return (
			self.weights.index_select(0, positions.view(-1))
			.view(bsz, seq_len, -1)
			.detach()
		)

def HuiPositionalEmbedding(
	num_embeddings: int,
	embedding_dim: int,
	padding_idx: int,
	learned: bool = False):

	return 	HuiSinusoidalPositionalEmbedding(
				embedding_dim,
				padding_idx,
				init_size=num_embeddings + padding_idx + 1,
			)

class HuiDecoder(TransformerDecoder):

	def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
		super().__init__(args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn)
		self.chunk_size = args.chunk_size
		if args.initialize_encoder_and_embedings:
			state_dict = self.state_dict()
			state = fairseq.checkpoint_utils.load_checkpoint_to_cpu("/data1/huanghui/SANMT/checkpoints/en2zh02/checkpoint_best.pt")
			for key in state["model"].keys():
				if "decoder." in key:
					for search_key in ["embed_tokens", "embed_positions", "output_projection"]:
						if search_key in key:
							subkey = key[key.find(search_key):]
							assert subkey in state_dict
							state_dict[subkey] = state["model"][key]
			self.load_state_dict(state_dict, strict=True)

		self.embed_positions = HuiPositionalEmbedding(self.max_target_positions,
													  args.decoder_embed_dim,
													  self.padding_idx)

	# def build_decoder_layer(self, args, no_encoder_attn=False):
	# 	return TransformerDecoderLayer(args, no_encoder_attn)

	def buffered_future_mask(self, tensor):
		dim = tensor.size(0)
		chunk_num = dim // self.chunk_size + 1
		if (
			self._future_mask.size(0) == 0
			or (not self._future_mask.device == tensor.device)
			or self._future_mask.size(0) < dim
		):
			self._future_mask = torch.triu(
				fairseq.utils.fill_with_neg_inf(torch.zeros([chunk_num, chunk_num])), 1
			)
			self._future_mask = self._future_mask.repeat_interleave(self.chunk_size, dim=0)
			self._future_mask = self._future_mask.repeat_interleave(self.chunk_size, dim=1)
			# self._future_mask = torch.triu(fairseq.utils.fill_with_neg_inf(torch.zeros([dim//chunk_size+1, dim//chunk_size+1])), 1).unsqueeze(-1).repeat(1, 1, chunk_size).view(dim//chunk_size+1, -1).unsqueeze(-2).repeat(1, chunk_size, 1).view(-1, (dim//chunk_size+1)*chunk_size)
		self._future_mask = self._future_mask.to(tensor)
		return self._future_mask[:dim, :dim]

	def forward(self, prev_output_tokens, encoder_out, incremental_state=None, **kwargs):
		x, extra = self.extract_features(
			prev_output_tokens,
			encoder_out=encoder_out,
			incremental_state=incremental_state,
		)
		# z, extra_z = self.extract_features(
		# 	prev_output_tokens[:,:2],
		# 	encoder_out=encoder_out,
		# 	incremental_state=incremental_state,
		# )	
		x = self.output_layer(x)
		# z = self.output_layer(z)
		# if incremental_state is not None:
		# 	pdb.set_trace()
		return x, extra

	def extract_features(self, prev_output_tokens, encoder_out, incremental_state=None, **kwargs):
		chunk_size = self.chunk_size

		positions = None
		if self.embed_positions is not None:
			positions = self.embed_positions(prev_output_tokens, incremental_state=incremental_state, chunk_size=self.chunk_size)

		if incremental_state is not None:
			prev_output_tokens = prev_output_tokens[:, -chunk_size:]

		x = self.embed_scale * self.embed_tokens(prev_output_tokens)

		if positions is not None:
			x += positions

		x = self.dropout_module(x)
		x = x.transpose(0, 1) # B x T x C -> T x B x C

		self_attn_padding_mask = None
		if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
			self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

		for idx, layer in enumerate(self.layers):
			if incremental_state is None:
				self_attn_mask = self.buffered_future_mask(x)
			else:
				self_attn_mask = None

			x, _, _ = layer(
				x,
				encoder_out.encoder_out,
				encoder_out.encoder_padding_mask,
				incremental_state,
				self_attn_mask=self_attn_mask,
				self_attn_padding_mask=self_attn_padding_mask,
			)

		# T x B x C -> B x T x C
		x = x.transpose(0, 1)

		return x, {"attn": None, "inner_states": None}
		

@register_model("hui_translation_model")
class HuiTransformer(TransformerModel):

	def __init__(self, args, encoder, decoder):
		super().__init__(args, encoder, decoder)
		self.args = args
		self.supports_align_args = True
		self.chunk_size = args.chunk_size

	# @classmethod
	# def build_encoder(cls, args, src_dict, embed_tokens):
	# 	 return HuiEncoder(args, src_dict, embed_tokens)

	@classmethod
	def build_decoder(cls, args, tgt_dict, embed_tokens):
		return HuiDecoder(args, tgt_dict, embed_tokens)

	def forward(
		self,
		src_tokens, 
		src_lengths, 
		prev_output_tokens, 
		**kwargs
	):
		encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)
		decoder_out = self.decoder(
			prev_output_tokens,
			encoder_out=encoder_out,
			chunk_size=self.chunk_size
		)
		return decoder_out

hui_model_base = base_architecture
register_model_architecture("hui_translation_model", "hui_translation_model_base")(hui_model_base)