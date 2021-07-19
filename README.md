# On the Copying Behaviors of Pre-Training for Neural Machine Translation (Findings of ACL 2021)

### Citation

Please cite as:

```bibtex
@inproceedings{liu2021copying,
  title={On the Copying Behaviors of Pre-Training for Neural Machine Translation},
  author={Liu, Xuebo and Wang, Longyue and Wong, Derek F and Ding, Liang and Chao, Lidia S and Shi, Shuming and Tu, Zhaopeng},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2021},
  year={2021}
}
```


### Requirements and Installation
This implementation is based on [fairseq(v0.9.0)](https://github.com/pytorch/fairseq/tree/v0.9.0/fairseq)

* [PyTorch](http://pytorch.org/) version >= 1.2.0
* Python version >= 3.6

```
git clone https://github.com/SunbowLiu/CopyingPenalty
cd CopyingPenalty
pip install --editable .
```

### Addtional Parameters
The copying penalty can be applied to both vanilla sequence-to-sequence learning models (`--task translation`) and (m)BART-initialized models (`--task translation_from_pretrained_bart`).  
Please refer to [fairseq](https://github.com/pytorch/fairseq/tree/master/examples/translation) and [mBART](https://github.com/pytorch/fairseq/tree/master/examples/mbart) for the training of models.

#### Features
1. As **simple** and **powerful** as the [length penalty](https://arxiv.org/abs/1609.08144).
2. **Particularly effective** in the task of similar input and output domains.
3. **Trivial** computational cost.

| parameter | description |
|---        |---          |
| --copypen | Copying penalty: <1.0 favors translating (converting) source tokens, >1.0 favors copying source tokens; Default: 1.0 |
| --target-dictionary-path | Path to target side dictionary (produced by `preprocess.py`) for automatically obtaining the punctuation list. |

Mainly modified code: `fairseq/sequence_generator.py`

### Tune a copying penalty on the validation set
```
mkdir $PATH_TO_OUTPUT/validation
for cp in 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5
do
RESULT=$PATH_TO_OUTPUT/validation/$cp.txt
python generate.py \
    $PATH_TO_DATA \
    --gen-subset valid \
    --path $PATH_TO_OUTPUT/checkpoint_best.pt \
    --copypen $cp --target-dictionary-path $PATH_TO_DATA/dict.tgt.txt \
    > $RESULT
done
```

### Apply the tuned copying penalty to the test set
```
python generate.py \
    $PATH_TO_DATA \
    --gen-subset test \
    --path $PATH_TO_OUTPUT/checkpoint_best.pt \
    --copypen $cp --target-dictionary-path $PATH_TO_DATA/dict.tgt.txt \
```







