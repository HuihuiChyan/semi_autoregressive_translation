# sacremoses tokenize -x < demo_data/train.en.extract > demo_data/train.en.extract.tok
# python -m jieba -d ' ' demo_data/train.zh.extract > demo_data/train.zh.extract.tok

# subword-nmt learn-bpe -s 32000 < demo_data/train.en.extract.tok > demo_data/bpe_code.en
# subword-nmt learn-bpe -s 32000 < demo_data/train.zh.extract.tok > demo_data/bpe_code.zh
# subword-nmt apply-bpe -c demo_data/bpe_code.en < demo_data/train.en.extract.tok > demo_data/train.en.extract.tok.bpe
# subword-nmt apply-bpe -c demo_data/bpe_code.zh < demo_data/train.zh.extract.tok > demo_data/train.zh.extract.tok.bpe

# sacremoses tokenize -x < demo_data/test2013.en > demo_data/test2013.en.tok
# python -m jieba -d ' ' demo_data/test2013.zh > demo_data/test2013.zh.tok
# subword-nmt apply-bpe -c demo_data/bpe_code.en < demo_data/test2013.en.tok > demo_data/test2013.en.tok.bpe
# subword-nmt apply-bpe -c demo_data/bpe_code.zh < demo_data/test2013.zh.tok > demo_data/test2013.zh.tok.bpe

# sacremoses tokenize -x < demo_data/test2015.en > demo_data/test2015.en.tok
# python -m jieba -d ' ' demo_data/test2015.zh > demo_data/test2015.zh.tok
# subword-nmt apply-bpe -c demo_data/bpe_code.en < demo_data/test2015.en.tok > demo_data/test2015.en.tok.bpe
# subword-nmt apply-bpe -c demo_data/bpe_code.zh < demo_data/test2015.zh.tok > demo_data/test2015.zh.tok.bpe

# mkdir demo_data/data-bin
# cp demo_data/train.en.extract.tok.bpe demo_data/data-bin/train.en
# cp demo_data/train.zh.extract.tok.bpe demo_data/data-bin/train.zh
# cp demo_data/test2013.en.tok.bpe demo_data/data-bin/dev.en
# cp demo_data/test2013.zh.tok.bpe demo_data/data-bin/dev.zh

# fairseq-preprocess \
# 	--source-lang en \
# 	--target-lang zh \
# 	--trainpref demo_data/data-bin/train \
# 	--validpref demo_data/data-bin/dev \
# 	--destdir demo_data/data-bin \
# 	--thresholdsrc 2 \
# 	--thresholdtgt 2
 
export CUDA_VISIBLE_DEVICES=3
# export CUDA_LAUNCH_BLOCKING=1
fairseq-train demo_data/data-bin \
	--arch hui_translation_model_base \
	--task hui_translation_task \
	--criterion label_smoothed_cross_entropy \
	--user-dir user_dir \
	--share-decoder-input-output-embed \
	--clip-norm 0.0 \
	--max-tokens 2048 \
	--lr 2e-4 \
	--lr-scheduler inverse_sqrt \
	--warmup-updates 5000 \
	--optimizer adam \
	--adam-betas '(0.9, 0.98)' \
	--dropout 0.3 \
	--weight-decay 0.0001 \
	--eval-bleu \
	--eval-bleu-args '{"beam":5, "max_len_a": 1.2, "max_len_b": 10}' \
	--eval-bleu-remove-bpe \
	--eval-bleu-print-samples \
	--best-checkpoint-metric bleu \
	--maximize-best-checkpoint-metric \
	--save-dir checkpoints/en2zh01 \
	--beam-size 5 \
	--chunk-size 2 \
	--patience 5 \
	--validate-interval-updates 5

# --initialize-encoder-and-embedings \
# --validate-interval-updates 100 \
# export CUDA_VISIBLE_DEVICES=6
# fairseq-interactive demo_data/data-bin \
# 	-s en \
# 	-t zh \
# 	--user-dir user_dir \
# 	--path checkpoints/en2zh02/checkpoint_best.pt \
# 	--beam 5 \
# 	--batch-size 200 \
# 	--buffer-size 200 \
# 	--no-progress-bar \
# 	--unkpen 5 < demo_data/test2015.en.tok.bpe > demo_data/test2015.en.tok.bpe.out
 
# grep -a ^H demo_data/test2015.en.tok.bpe.out | cut -f3- > demo_data/test2015.zh.out.grep
# sed -r 's/@@ //g' demo_data/test2015.zh.out.grep > demo_data/test2015.zh.out.grep.debpe
# python2 tokenizeChinese.py demo_data/test2015.zh.out.grep.debpe demo_data/test2015.zh.out.grep.debpe.char
# python2 tokenizeChinese.py demo_data/test2015.zh demo_data/test2015.zh.char
# sacrebleu -w 2 -i demo_data/test2015.zh.out.grep.debpe.char demo_data/test2015.zh.char 
