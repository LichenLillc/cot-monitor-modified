# Can We Predict Alignment Before Models Finish Thinking? Towards Monitoring Misaligned Reasoning Models

This repository contains the experimental pipeline for our paper, ["Can We Predict Alignment Before Models Finish Thinking? Towards Monitoring Misaligned Reasoning Models"](https://arxiv.org/abs/2507.12428). Open-weight reasoning models such as s1.1 and DeepSeek-R1 generate extensive chains-of-thought (CoT) reasoning before producing final responses. This creates an opportunity to monitor safety during the reasoning process rather than only after completion. Our work systematically evaluates text-based and activation-based monitoring approaches for predicting final response alignment. We find that linear probes (logistic regression) trained on activations significantly outperform strong text classification methods, including GPT models and ModernBert, in predicting whether the model's final outputs will be safe.

### Prerequisites

```bash
# Install required packages
pip install torch transformers vllm
pip install scikit-learn nltk loguru tqdm
pip install datasets requests matplotlib pandas

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

### Pipeline

The experimental pipeline follows the steps below. All code can be found in the `\scripts` directory, organized sequentially. We first generate a complete CoT and final response for a given harmful prompt (`0_generation.py`), then truncate the CoT at sentence boundaries and generate responses at each truncation (`1_truncation.py`). For each CoT–final response pair, we evaluate the safety of the final response (`2a_evaluate_safety.py`) and collect internal activations for the CoT (`2b_get_activations.py`). In step 3, we evaluate a range of monitoring methods, including simple probes (`3_probes.py`), and text classification methods (`3b` to `3d`). "Example usage of the scripts is shown below.

0. Generate CoT reasoning and final responses
```bash
python3 scripts/0_generation.py \
    --model_name simplescaling/s1.1-7B \
    --data_file data/strongreject.jsonl \
    --output_dir raw_outputs
```

1. Create truncated CoT and responses at each truncation point
```bash
python3 scripts/1_truncation.py \
    --input_file raw_outputs/s1.1-7B_strongreject.jsonl \
    --model_name simplescaling/s1.1-7B \
    --base_output_dir processed
```

2a. Evaluate safety of responses  
```bash
python3 scripts/2a_evaluate_safety.py \
    --results_folder processed/s1.1-7B_strongreject/labels
```

2b. Extract CoT activations
```bash
python3 scripts/2b_get_activations.py \
    --results_folder processed/s1.1-7B_strongreject/labels \
    --model_name simplescaling/s1.1-7B \
    --activations_dir processed/s1.1-7B_strongreject/activations
```

3. Evaluate activation-based and text-based monitoring
```bash
# Train simple probes
python3 scripts/3a_probe.py \
    --input_folder processed/s1.1-7B_strongreject \
    -- pca

# Use text-based approaches
python3 3b_text_classifier.py \
    --input_folder processed/s1.1-7B_strongreject \
    --text_classifier_model "answerdotai/ModernBERT-large" \
    --train_bsz 4 \

python3 3c_openai_classifier.py \
    --input_folder processed/s1.1-7B_strongreject \
    --use_icl \

python3 3d_cot_harm_classifier.py \
    --input_folder processed/s1.1-7B_strongreject/labels \
    --eval_cot \
    --eval_para
```

### Datasets

Currently, the repository includes three safety evaluation datasets. The data can be found in `data/`.
- [StrongReject](https://huggingface.co/datasets/walledai/StrongREJECT)
- [SorryBench](https://huggingface.co/datasets/sorry-bench/sorry-bench-202503)
- [HarmBench](https://huggingface.co/datasets/walledai/HarmBench)

## Contact
Feel free to email Yik Siu Chan (yik_siu_chan@brown.edu) and Zheng-Xin Yong (contact.yong@brown.edu) if you have questions about the code or paper.
For bug reports or issues, please open a GitHub issue.

## Citation
```bibtex
@article{chan2025predictalignmentmodelsfinish,
    title={Can We Predict Alignment Before Models Finish Thinking? Towards Monitoring Misaligned Reasoning Models}, 
    author={Yik Siu Chan and Zheng-Xin Yong and Stephen H. Bach},
    journal={arXiv preprint arXiv:2507.12428},
    url={https://arxiv.org/abs/2507.12428}, 
    year={2025}
}
```