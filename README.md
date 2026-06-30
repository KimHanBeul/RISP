# RISP: Representative Item Summarization Prompting for LLM-based Sequential Recommendation

This repository is designed for our ISIS 2024 paper **"RISP: Representative Item Summarization Prompting for LLM-based Sequential Recommendation"**.

## Environment Setting
1. **Prepare the environment**
```
pip install -r requirement.txt
```
2. **Prepare the LLM**
    * Download the pre-trained LLM(Llama3-8B-Instruct) from [Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).

## Evaluate Model
* Replace tokenizer_path on line 105 of run_test.py with your LLM path
* Evalue the model in MovieLens(ml-1m), LastFM(lastfm), and Amazon Games(Games).
```bash
python run_test.py -d m1-1m
```

## Acknowledgement
**RISP** is built upon the **LLMSRec_Syn** framework for model construction and experimentation.  

We would like to acknowledge the authors of [LLMSRec_Syn](https://github.com/demoleiwang/LLMSRec_Syn) for their open-source contributions.
