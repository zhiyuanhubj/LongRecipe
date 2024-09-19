## Logits Computation based on Llama3-8B Example.

We use the existing next token prediction of vllm and modify the `simpler.py` file to obtain the logits for the input prompt. The version of vllm we are using is [v0.4.0](https://github.com/vllm-project/vllm/tree/v0.4.0). After successfully installing vllm, you need to replace the original `simpler.py` file in `/model_executor/layers/sampler.py` with the modified version.


Thus, you can excute the following code and adjust the file-saving paths as needed.
```
python logits_compute.py
```

Subsequently, you will have three files: `prompt_logits.json` containing the logits for the inputs,, `prompt_tokens_logits.json` mapping tokens to their corresponding logits., and `prompt_ids_logits.json` mapping token ids to their corresponding logits.
