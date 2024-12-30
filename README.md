# IPD

## Overview

IPD is an interpretable personality detection model on social media using Type Dynamics Theory, the framework of IPD is depicted in Figure 1.
<div align=center>
    <img src="figs/framework.png" width="600px">
</div>
<div align=center>
Figure 1: The framework of IPD.
</div>


### Dataset
We evaluate our method on the [Kaggle](https://www.kaggle.com/datasnaek/mbti-type) and [Pandora](https://psy.takelab.fer.hr/datasets/all) datasets.


## Quick Start

**Step1: Installation for tools**

```python
# python 3.8
pip install -r requirements.txt
```

**Step2: Write a configuration file in yaml format**
Users can easily configure the parameters of METER in a yaml file. 
The path of the configuration file is IPD/config/config.yaml

```yaml
openai:
  api_key: # your_openai_API_key
  base_url: # url
  temperature: 0.2  
  max_tokens: 1024
prompts:
  llm_prompt: prompts/prompt.yaml
```

**Step3: Run with the code**
```python
python src/main.py
```

## Citation
Please cite our repository if you use IPD in your work.
```bibtex

```
