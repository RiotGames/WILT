# WILT: Wason Inductive Logic Test

<p align="center">
    <img src="https://github.com/ambisinister/wilt/blob/main/docs/wilt.png?raw=true" width="128">
</p>

<p align="center">
    <img src="/docs/wilt_sign.png" width="256">
</p>


**WILT** a simple, general, multi-turn inductive logic test for language models. The test is comprised of simple black-box functions of three variables, where you may test up to 30 examples in order to observe the input-output relationships of these functions.

This test is based on the [Wason 2-4-6 task](https://journals.sagepub.com/doi/10.1080/17470216008416717). By asking the model to infer functions involving three variables, we get a reasoning benchmark which is both robust to memorization and reliably predictive of a model's ability to solve simple problems based on previous observations. 

⚠️ Warning: This benchmark executes code provided by language models. Please be sure your environment is properly sandboxed.

<p align="center">
    <img src="/docs/results.png">
</p>

## Installation

This project was developed using Python 3.9 and is compatible with Python 3.x versions.

```
git clone https://github.com/riotgames/wilt.git
cd wilt
pip install -r requirements.txt
```

## API Keys

This code requires API keys in your environment variables for the models you wish to test. The required environment variable depends on the model provider:

- OpenAI models (GPT-4o, o1-mini): ```OPENAI_API_KEY```
- Anthropic models (Claude Sonnet 3.5): ```ANTHROPIC_API_KEY```
- Mistral models: ```MISTRAL_API_KEY```
- DeepSeek models: ```DEEPSEEK_API_KEY```
- etc.

You can find the specific API key requirements for each model in their respective files under the /models directory (e.g., /models/deepseek_model.py).

## Usage

```
python main.py --model="gpt-4o-mini" --split lite --multi=10
```

Parameters:

- ```--model```: Model to evaluate (see Supported Models)
- ```--split```: Test split to use (```lite``` or ```full```)
- ```--multi```: number of tests to run in parallel (avoid for rate-limited models)
- ```--sleep```: sleep timer between turns (use for rate-limited models)

### Supported Models

For the full list of supported models, please see /models/model_factory.py. Some models we support in this repository are:

```
"o1-mini-2024-09-12",
"o1-preview-2024-09-12",
"llama3-70b-8192",
"llama-3.1-70b-versatile",
"llama-3.1-8b-instant",
"llama3-8b-8192",
"gemma-7b-it",
"gemma2-9b-it",
"mixtral-8x7b-32768",
"meta.llama3-1-405b-instruct-v1:0",
"meta.llama3-1-8b-instruct-v1:0",
"gemini-1.5-flash",
"gemini-1.5-pro",
"gemini-1.5-pro-exp-0801",
"gemini-1.5-pro-exp-0827",
"gemini-1.5-flash-exp-0827",
"gemini-1.5-flash-8b-exp-0827",
"mistral-large-2407",
"open-mistral-nemo",
"chatgpt-4o-latest",
"gpt-4-turbo",
"gpt-3.5-turbo",
"gpt-4o-2024-05-13",
"gpt-4o-2024-08-06",
"gpt-4o-mini",
"claude-3-5-sonnet-20240620",
"claude-3-haiku-20240307",
"deepseek-chat",
"deepseek-coder",
```

## Initial results on Light Test Split

<p align="center">
    <img src="/docs/litesplit.png">
</p>

We find the light split to be a good show of reasoning capabilities as a back-of-the-napkin test; not necessarily enough to differentiate similar models, but enough to roughly assess a model's ability to do simple reasoning.

## Initial results on Full Test Set

We find Claude Sonnet 3.5 to be the clearly highest performing model at this task. We measure two types of answers: correct, and approximately correct, in order to have some resolution into which models uncover almost-correct descriptions of the hidden rule, but fail to properly explore edge cases. For example, x < y < z instead of x <= y <= z; or x <= y <= z instead of max([x,y,z]) == z. Note that test cases are only counted once.

We find some interesting ordering from this -- mistral large performs well on multi-turn scenarios compared to other similar models, and more carefully explores the hypothesis space as a result. Llama 3.1 405B scores low due to high confidence early in the conversation, and under the same instructions sees an abnormally high approximate correctness as a result. All models make clear, unnatural reasoning failures on easy hidden rules (for example, Sonnet 3.5 guesses $x<0 \land y<0 \land z<0 \land x=10y \land y=10z$ for the hidden rule $x<0 \land y<0 \land z<0$).


## Citation

If you use WILT in your research, please cite:

```bibxtex
@article{banatt2024wilt,
  title={WILT: A Multi-Turn, Memorization-Robust Inductive Logic Benchmark for LLMs},
  author={Banatt, Eryk and Cheng, Jonathan and Vaidyanath, Skanda and Hwu, Tiffany},
  booktitle={The 4th Workshop on Mathematical Reasoning and AI, NeurIPS 2024},
  journal={arXiv preprint arXiv:2410.10998},
  year={2024}
}
```
