# Kernelized Activation Steering (KAS)

**Content warning**: This repository contains text that is offensive, harmful, or otherwise inappropriate in nature.

Code for [Kernelized Activation Steering](https://arxiv.org/abs/2610.01062) ([paper](paper.pdf)).
KAS steers a model by moving its activations along the gradient of a kernel score that pulls them
toward target examples and away from source examples.

The experiments are in two folders, each with its own environment and README:

| Folder | Experiments |
|---|---|
| [`Jailbreaking_CAA/`](Jailbreaking_CAA) | jailbreaking on JailbreakBench, CAA-style multiple-choice steering, and the KAS ablations, on Gemma-2-9B-IT, Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct |
| [`Image/`](Image) | image-generation style control on FLUX.1-schnell, compared with Mean-AcT |

To use each folder, please follow the bashes below:

```bash
git clone https://github.com/KleepyHunter/kernel_steering.git
cd kernel_steering/Jailbreaking_CAA   # or: cd kernel_steering/Image
```

## Citation

```bibtex
@misc{abdullaev2026kernelizedactivationsteering,
      title={Kernelized Activation Steering},
      author={Laziz U. Abdullaev and Minh-Hieu Pham and Bach Do and Khoat Than and Tan M. Nguyen},
      year={2026},
      eprint={2610.01062},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2610.01062},
}
```

## License

See [LICENSE](LICENSE).
