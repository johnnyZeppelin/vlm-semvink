# SemVink: Advancing VLMs' Semantic Understanding of Optical Illusions via Visual Global Thinking  <img src="asset/zoomout.ico" width="36"/>

<p align="center">
    <br>
    <img src="asset/cases.png"/>
    <br>
<p>

---

<p align="center">
    <br>
    <img src="asset/problem.png"/>
    <br>
<p>

This is a codebase for the tests and experiments in the paper <a href="https://arxiv.org/abs/2506.02803" target="_blank">SemVink: Advancing VLMs' Semantic Understanding of Optical Illusions via Visual Global Thinking</a>, which is accepted as a main conference long paper to EMNLP 2025.

## 📖 Contents
- [Usage](#-usage)
- [Citation](#-citation)


## ✨ Usage
For the HC-Bench dataset, please see <a href="https://huggingface.co/datasets/JohnnyZeppelin/HC-Bench" target="_blank">this repository on Hugging Face</a>.

The HC-Bench or any other hidden-content images can be put in `./data`.

```bash
git clone https://github.com/johnnyZeppelin/vlm-semvink.git
cd vlm-semvink
mkdir data
```

You can install the environment by
```bash
pip install -r requirements.txt
```
The `torch` must match your system and CUDA version.

The `qwen-vl-utils` is not on PyPI by default. If you cannot install it with `pip install qwen-vl-utils`, it’s likely part of the **Qwen-VL GitHub repo**. You will need to install from source, e.g.,

```bash
pip install git+https://github.com/QwenLM/Qwen-VL.git
```

In this repository, the inference time zoom-out and squint methods are provided in `inference/main.py`.

You can edit the model and dataset parts in `main.py` and run this inference-time solution by

```bash
python ./inference/main.py
```

The results are as follows:
| Model                    | Zero-Shot Direct Text (%) | Zero-Shot Direct Object (%) | Zero-Shot Hinted Text (%) | Zero-Shot Hinted Object (%) | Zero-Shot Prompt Text (%) | Zero-Shot Prompt Object (%) | Few-Shot Text (%) | Few-Shot Object (%) | w/ zoom-out Text (%) | w/ zoom-out Object (%) |
|--------------------------|---------------------------|-----------------------------|----------------------------|-----------------------------|----------------------------|-----------------------------|-------------------|---------------------|-----------------------|------------------------|
| o3                       | 0                         | 0                           | 0                          | 0                           | 0                          | 0                           | 0                 | 0                   | 100.0                 | 100.0                  |
| o4-mini                  | 0                         | 0                           | 0                          | 0                           | 0                          | 0                           | 0                 | 0                   | 100.0                 | 100.0                  |
| GEMINI 2.5 Pro           | 0                         | 0                           | 0                          | 0                           | 0                          | 0                           | 0                 | 0                   | 100.0                 | 100.0                  |
| Grok 3                   | 0                         | 5.36                        | 0                          | 8.93                        | 0                          | 5.36                        | 0                 | 5.36                | 98.21                 | 100.0                  |
| Mistral                  | 0                         | 0                           | 0                          | 10.71                       | 0                          | 0                           | 0                 | 5.36                | 96.43                 | 100.0                  |
| Claude 3.7 Sonnet        | 0                         | 0                           | 1.78                       | 3.57                        | 0                          | 0                           | 0                 | 0                   | 98.21                 | 100.0                  |
| LLaVA-v1.5-7B            | 0                         | 0                           | 0                          | 0                           | 0                          | 0                           | 0                 | 0                   | 91.07                 | 98.21                  |
| Doubao-1.5-pro           | 0                         | 0                           | 0                          | 0                           | 0                          | 0                           | 0                 | 0                   | 96.43                 | 98.21                  |
| Kimi-VL-A3B-Thinking     | 0                         | 0                           | 0                          | 0                           | 0                          | 0                           | 0                 | 0                   | 94.64                 | 100.0                  |
| Qwen2-VL-7B-Instruct     | 1.78                      | 3.57                        | 3.57                       | 3.57                        | 1.78                       | 3.57                        | 1.78              | 3.57                | 100.0                 | 96.43                  |
| Qwen2-VL-72B-Instruct    | 1.78                      | 1.78                        | 5.36                       | 3.57                        | 1.78                       | 3.57                        | 1.78              | 3.57                | 100.0                 | 100.0                  |
| DeepSeek-VL2             | 0                         | 0                           | 0                          | 0                           | 0                          | 0                           | 0                 | 0                   | 92.86                 | 94.64                  |

---

## 📎 Citation

```bibtex
@misc{li2025semvinkadvancingvlmssemantic,
      title={SemVink: Advancing VLMs' Semantic Understanding of Optical Illusions via Visual Global Thinking}, 
      author={Sifan Li and Yujun Cai and Yiwei Wang},
      year={2025},
      eprint={2506.02803},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.02803}, 
}
```

## Star History

<a href="https://www.star-history.com/#johnnyZeppelin/vlm-semvink&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=johnnyZeppelin/vlm-semvink&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=johnnyZeppelin/vlm-semvink&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=johnnyZeppelin/vlm-semvink&type=Date" />
 </picture>
</a>


## Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
