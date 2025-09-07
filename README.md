# VASE
Official implementation of Vision-Amplified Semantic Entropy for Hallucination Detection in Medical Visual Question Answering (MICCAI 2025).

#### [Project Page](https://github.com/Merrical/VASE)

This repo contains the official implementation of our paper: Vision-Amplified Semantic Entropy for Hallucination Detection in Medical Visual Question Answering, which proposed a hallucination detection method for medical VQA.
<p align="center"><img src="https://raw.githubusercontent.com/Merrical/VASE/master/VASE_overview.png" width="90%"></p>

#### [Paper](https://arxiv.org/pdf/2503.20504)

### Requirements

We suggest using virtual env to configure the experimental environment. Parts of the [Semantic Entropy](https://www.nature.com/articles/s41586-024-07421-0) computation code are adapted from https://github.com/lorenzkuhn/semantic_uncertainty.

1. Clone this repo:
```bash
git clone https://github.com/Merrical/VASE.git
```

2. Create an environment 'env_main' for medical MLLMs ( [MedGemma-4b-it](https://huggingface.co/google/medgemma-4b-it) used in this project).

3. Create another dedicated environment 'env_green' for the [GREEN](https://github.com/Stanford-AIMI/GREEN) model.

4. Activate env_main, generate answers for open-ended test VQA samples of the RAD-VQA dataset with MedGemma-4b-it (temperature = 0.1), and compute hallucination scores (RadFlag, Semantic Entropy, and our VASE).

```bash
python main_hall_det.py
```

5. Activate env_green, and obtain GREEN scores (hallucination labels) for all test samples.
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 green_eval.py
```

6. Evaluate hallucination detection performance (AUC, AUG) of RadFlag, SE, and VASE.
```bash
python hall_det_eval.py
```

### Bibtex
```bash
@inproceedings{Liao2025VASE,
  title={Vision-Amplified Semantic Entropy for Hallucination Detection in Medical Visual Question Answering},
  author={Liao, Zehui and Hu, Shishuai and Zou, Ke and Fu, Huazhu and Zhen, Liangli and Xia, Yong},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2025},
  organization={Springer}
}
```

### Contact Us
If you have any questions, please contact us ( merrical@mail.nwpu.edu.cn ).