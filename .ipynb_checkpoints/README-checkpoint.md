# SemVink: Advancing VLMs' Semantic Understanding of Optical Illusions via Visual Global Thinking  <img src="asset/zoomout.ico" width="36"/>

<p align="center">
    <br>
    <img src="asset/cases.png"/>
    <br>
<p>

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
For the HC-Bench dataset, please see <a href="https://hf" target="_blank">this repository on Hugging Face</a>.

The HC-Bench or any other hidden-content images can be put in ./data/

In this repository, the inference time zoom-out and squint methods are provided in inference/main.py.

You can edit the model and dataset parts in main.py and run this inference-time solution by

<!-- ```bash -->
>python ./inference/main.py
<!-- ``` -->


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
