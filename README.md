# About the repository
This repository implements **the Long-term Recurrent Convolutional Network (LRCN)** for the video behavior recognition task with **PyTorch**.

You can refer to the original work through the link: https://openaccess.thecvf.com/content_cvpr_2015/papers/Donahue_Long-Term_Recurrent_Convolutional_2015_CVPR_paper.pdf

It is required to download the **UCF101** dataset, which is a public behavior recognition dataset, to benchmark the code in this repository.

You can refer to the UCF101 dataset by following the link: https://www.crcv.ucf.edu/data/UCF101.php

# Quick Start
**Befroe running the script, you need to properly set up *DATA_DIR* in run_lrcn.sh according to your UCF101 dataset location.**

You can launch the experiment just running the script.
```
./run run_lrcn.sh
```

# Benchmark Result
|    Dataset        | Accuracy |
| :---------------: | :---------------: |
| UCF-101 (Split 1)|  **82.2%** |


# LRCN Variation - 2D CNN + 1D CNN version
The repository includes a variation of LRCN where LSTM in the LRCN model is replaced with the one-dimensional CNN layer.

The introduced one-dimensional CNN layer plays a role of capturing temporal dynamics in video as LSTM does.

The benefit comes from the fact that the CNN layer has a better GPU utilization than LSTM since LSTM has a sequential execution path.  

(It becomes easier to understand if one recalls one of original motivations in developing the Transformer model was to achieve parallelization by avoiding RNN)

The idea is explained in the paper:

```BibTeX
@article{
  author  = "Jeonghyun Joo, Heeyoul Choi",
  title   = "Improving Parallelism for Video Action Recognition Model Using One-dimensional Convolutional Neural Network",
  journal = "KIISE Transactions on Computing Practices (KTCP)",
  year    = 2021,
  volume  = "27",
  number  = "4",
  pages   = "216-220"
}
```

You can run this variation with the following command.
```
./run run_1dcnn.sh
```
