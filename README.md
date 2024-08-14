# MaskViM

> **Abstract:** *Domain Generalized Semantic Segmentation (DGSS) aims to utilize segmentation model training on known source domains to make predictions on unknown target domains. Currently, there are two network architectures: one based on Convolutional Neural Networks (CNNs) and the other based on Visual Transformers (ViTs). However, both CNN-based and ViT-based DGSS methods face challenges: the former lacks a global receptive field, while the latter requires more computational demands. Drawing inspiration from State Space Models (SSMs), which not only possess a global receptive field but also maintain linear complexity, we propose SSM-based method for achieving DGSS. In this work, we first elucidate why does $\textit{mask}$ make sense in SSM-based DGSS and propose our $\textit{mask}$ learning mechanism. Leveraging this mechanism, we present our $\textit{Mask Vision Mamba}$ network (MaskViM), a model for SSM-based DGSS, and design our $\textit{mask}$ loss to optimize MaskViM. Our method achieves superior performance on four diverse DGSS setting, which demonstrates the effectiveness of our method.* 
>
> <p align="center">
> <img width="800" src="figs/overview.png">
> </p>
