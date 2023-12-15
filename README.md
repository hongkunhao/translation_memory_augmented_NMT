# Rethinking Translation Memory Augmented Neural Machine Translation

## Installation
* [PyTorch](http://pytorch.org/) version >= 1.10.0
* Python version >= 3.8

``` bash
git clone https://github.com/HongkunHao/translation_memory_augmented_NMT.git
cd translation_memory_augmented_NMT/fairseq
pip install --editable ./
```

## Train
### Conditioning on One TM Sentence

``` bash
sh train_condition_on_one_TM.sh
```

### TM-Augmented NMT via Weighted Ensemble

``` bash
sh train_weighted_ensemble.sh
```

## Inference

``` bash
sh inference.sh
```

