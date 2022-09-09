<h1 align="center">AARGH! </h1>
<h3 align="center">End-to-end Retrieval-Generation for Task-Oriented Dialog</h3>
<h4 align="center">~ <a href='_pdfs/aargh.pdf'>See the paper</a> ~</h4>

<p>&nbsp;</p>

_______

This repository provides the code including the training and evaluation scripts associated with the paper [AARGH! End-to-end Retrieval-Generation for Task-Oriented Dialog](_pdfs/aargh.pdf). It also contains the model weights of our best-performing Action-Aware Retrieval-Generative Hybrid model.   

##  :hourglass: Installation

``` bash
git clone https://github.com/Tomiinek/Aargh.git
cd Aargh
pip install -e .
```

## :muscle: Training

We provide model weights of our best performing AARGH model with low and high blending parameter. You can download it [here](https://drive.google.com/file/d/19vtX8ZiK1lcevinXEC40-kYUyEHY2xpV/view?usp=sharing) and [here](https://drive.google.com/file/d/18CQA4UIatKpcY2Bqm9j8FWVtXs88w86r/view?usp=sharing), respectively. The models should give Inform 82.8, Success: 71.2, and Inf: 90.3, Succ: 71.7 when evaluating in greedy mode on the test set. 

### **0. What do you want to train?**

It is possible to reproduce all model setups and experiments in the paper by selecting one of the following configurations: `t5_joint`, `bert_dual_action`, `bert_dual`, `bert_poly`, `bert_action`, or `t5_vanilla`. See the [config directory](https://github.com/Tomiinek/Dialogorum/tree/aargh-publish/aargh/config) for details. Use `t5_joint` to train the AARGH model. 

``` bash
seed="42"
model="t5_joint" 
```

### **1. Training of the AARGH model (or just the retrieval parts of the models)**
``` bash
out_path="${model}/retriever/${seed}"

python scripts/train.py \
    --deterministic \
    --gpus 1 \
    --num-workers 4
    --root-suffix ${out_path} \
    --config config/${model}.yaml \
    --set seed=${seed} \
```

### **2. Calculate train set embeddings using the retrieval parts**
``` bash
ckpt="last.ckpt"

ckpt_path="outputs/${model}/retriever/${seed}/checkpoints/${ckpt}"
out_path=$(dirname $(dirname "$ckpt_path"))

python scripts/get_embeddings.py ${ckpt_path} ${out_path} 64
```

### **3. Traning of the generative parts of the two stage models**
``` bash
# use this with on of `bert_dual_action`, `bert_dual`, `bert_poly`, and `bert_action` 

out_path="outputs/${model}/generator/${seed}"
hint_path="outputs/${model}/retriever/${seed}/hints.json"

python scripts/train.py \
    --deterministic \
    --gpus 1 \
    --num-workers 4
    --root-suffix ${out_path} \
    --config config/t5_separate.yaml \
    --set seed=${seed} hint_path=${hint_path} \
```

``` bash
# use this with `t5_vanilla`

out_path="outputs/${model}/generator/${seed}"

python scripts/train.py \
    --deterministic \
    --gpus 1 \
    --num-workers 4
    --root-suffix ${out_path} \
    --config config/${model}.yaml \
    --set seed=${seed} \
```

## :rocket: Evaluation

### **1. Generate responses on test data**

``` bash
greedy="true" # or "false"
beam_size="8"
ckpt="last.ckpt"
fold="test" # or `val`

end_path=$([ "$model" = "t5_joint" ] && echo "retriever" || echo "generator")
out_path="outputs/${model}/${end_path}/${seed}"
ckpt_path="outputs/${model}/${end_path}/${seed}/checkpoints/${ckpt}"
hint_path="outputs/${model}/retriever/${seed}/train_encodings.pkl"
ret_path="outputs/${model}/retriever/${seed}/checkpoints/${ckpt}"

python generate.py \ 
    -c ${ckpt_path} \
    -g \
    -t context api_call \
    -o $out_path/${fold}_beam_outputs.json \
    -f $fold \
    --set \
        greedy=${greedy} \
        num_beams=${beam_size} \
        retrieval_checkpoint=${ret_path} \
        support_path=${hint_path}" 
```

### **2. Evaluate the generated responses**

``` bash
mode="beam" # or `greedy`
fold="test" # or `val`

end_path=$([ "$model" = "t5_joint" ] && echo "retriever" || echo "generator")

python scripts/get_responses_2.py \
    "outputs/${model}/${end_path}" ${fold} ${mode} \
    "outputs/${model}/${end_path}/${fold}_${mode}_dst_metrics_stats.txt
```

### **3. Evaluate the retrieval parts**
``` bash
fold="test" # or `val`
out_path="outputs/${model}/retriever"

python scripts/get_stats.py ${out_path} ${fold} "${out_path}/${fold}_action_accuracy_stats.txt"
python scrripts/get_responses.py ${out_path} ${fold} "${out_path}/${fold}_mwz_metrics_stats.txt"
python scripts/eval_clustering.py ${out_path} ${fold} "${out_path}/${fold}_clustering.txt "
```

# :thought_balloon: Citation

```
@inproceedings{nekvinda_aargh_2022,
	address = {Edinburgh, Scotland},
	title = {{AARGH}! {End}-to-end {Retrieval}-{Generation} for {Task}-{Oriented} {Dialog}},
	booktitle = {Proceedings of the {SIGdial} 2022 Conference},
	author = {Nekvinda, Tomáš and Dušek, Ondřej},
	month = sep,
	year = {2022},
	pages = {283--297},
}
```
