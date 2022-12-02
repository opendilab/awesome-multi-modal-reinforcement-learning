# Awesome Multi-Modal Reinforcement Learning 

This is a collection of research papers for **Multi-Modal reinforcement learning (MMRL)**.
And the repository will be continuously updated to track the frontier of MMRL. 
Some papers may not be relevant to RL, but we include them anyway as they may be useful for the research of MMRL.

Welcome to follow and star!

## Introduction

Multi-Modal RL agents focus on learning from video (images), language (text), or both, as humans do. We believe that it is important for intelligent agents to learn directly from images or text, since such data can be easily obtained from the Internet.

![飞书20220922-161353](https://user-images.githubusercontent.com/4834562/191696555-2ff17e41-77f4-4d04-ba2a-ea9bc8d99d96.png)

## Papers

```
format:
- [title](paper link) [links]
  - authors.
  - key words.
  - experiment environment.
```
- [End-to-end Generative Pretraining for Multimodal Video Captioning](https://arxiv.org/abs/2201.08264)
  - Paul Hongsuck Seo, Arsha Nagrani, Anurag Arnab, Cordelia Schmid
  - Key Words: Multimodal video captioning,  Pretraining using a future utterance, Multimodal Video Generative Pretraining
  - ExpEnv: [HowTo100M](https://arxiv.org/abs/1906.03327)

- [Open-vocabulary Queryable Scene Representations for Real World Planning](https://arxiv.org/abs/2209.09874)
  - Boyuan Chen, Fei Xia, Brian Ichter, Kanishka Rao, Keerthana Gopalakrishnan, Michael S. Ryoo, Austin Stone, Daniel Kappler
  - Key Words: Target Detection, Real World, Robotic Tasks
  - ExpEnv: [Say Can](https://say-can.github.io/)

- [Do As I Can, Not As I Say: Grounding Language in Robotic Affordances](https://arxiv.org/abs/2204.01691)
  - Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Daniel Ho, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Irpan, Eric Jang, Rosario Jauregui Ruano, Kyle Jeffrey, Sally Jesmonth, Nikhil J Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Kuang-Huei Lee, Sergey Levine, Yao Lu, Linda Luu, Carolina Parada, Peter Pastor, Jornell Quiambao, Kanishka Rao, Jarek Rettinghouse, Diego Reyes, Pierre Sermanet, Nicolas Sievers, Clayton Tan, Alexander Toshev, Vincent Vanhoucke, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Mengyuan Yan, Andy Zeng
  - Key Words: real world, natural language
  - ExpEnv: [Say Can](https://say-can.github.io/)


- [Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks](https://arxiv.org/abs/2208.10442)
  - Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, Furu Wei
  - Key Words: backbone architecture, pretraining task, model scaling up
  - ExpEnv: [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/), [COCO](https://cocodataset.org/), [NLVR2](https://paperswithcode.com/dataset/nlvr), [Flickr30K](https://paperswithcode.com/dataset/flickr30k)

- [LM-Nav: Robotic Navigation with Large Pre-Trained Models of Language, Vision, and Action](https://arxiv.org/abs/2207.04429)
  - Dhruv Shah, Blazej Osinski, Brian Ichter, Sergey Levine
  - Key Words: robotic navigation, goal-conditioned, unannotated large dataset
  - ExpEnv: None


- [SOAT: A Scene- and Object-Aware Transformer for Vision-and-Language Navigation](https://arxiv.org/abs/2110.14143)
  - Abhinav Moudgil, Arjun Majumdar, Harsh Agrawal, Stefan Lee, Dhruv Batra
  - Key Words: visual navigation, natural language instructions, large-scale web data
  - ExpEnv: Room-to-Room (R2R),  Room-Across-Room (RxR)

- [Zero-Shot Task Generalization with Multi-Task Deep Reinforcement Learning](https://arxiv.org/abs/1706.05064)
  - Junhyuk Oh, Satinder Singh, Honglak Lee, Pushmeet Kohli
  - Key Words: unseen instruction, sequential instruction
  - ExpEnv: [Minecraft](https://arxiv.org/abs/1605.09128)


- [MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge](https://arxiv.org/abs/2206.08853)
  - Linxi Fan, Guanzhi Wang, Yunfan Jiang, etc. *ArXiv2022*
  - Key Words: multimodal dataset, MineClip
  - ExpEnv: [Minecraft](https://minedojo.org/)

- [R3M: A Universal Visual Representation for Robot Manipulation](https://arxiv.org/abs/2203.12601)
  - Suraj Nair, Aravind Rajeswaran, Vikash Kumar, etc. *ArXiv2022*
  - Key Words: Ego4D human video dataset, pre-train visual representation
  - ExpEnv: [MetaWorld](https://github.com/suraj-nair-1/metaworld), [Franka Kitchen, Adroit](https://github.com/aravindr93/mjrl)

- [SOAT: A Scene-and Object-Aware Transformer for Vision-and-Language Navigation](https://arxiv.org/pdf/2110.14143.pdf)
  - Abhinav Moudgil, Arjun Majumdar,Harsh Agrawal, etc. *NeurIPS2021*
  - Key Words: Vision-and-Language Navigation
  - ExpEnv: [Room-to-Room](https://paperswithcode.com/dataset/room-to-room), [Room-Across-Room](https://github.com/google-research-datasets/RxR)

- [Recurrent World Models Facilitate Policy Evolution](https://papers.nips.cc/paper/2018/hash/2de5d16682c3c35007e4e92982f1a2ba-Abstract.html)
  - David Ha, Jürgen Schmidhuber. *NeurIPS2018*
  - Key Words: World model, generative RNN, VAE
  - ExpEnv: [VizDoom](https://github.com/mwydmuch/ViZDoom), [CarRacing](https://github.com/openai/gym)

- [Mastering Atari with Discrete World Models](https://arxiv.org/abs/2010.02193)
  - Danijar Hafner, Timothy Lillicrap, Mohammad Norouzi, etc. *ICLR2021*
  - Key Words: World models
  - ExpEnv: [Atari](https://github.com/openai/gym)
   
- [Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos](https://arxiv.org/abs/2206.11795)
  - Bowen Baker, Ilge Akkaya, Peter Zhokhov, etc. *ArXiv2022*
  - Key Words: Inverse Dynamics Model
  - ExpEnv: [minerl](https://github.com/minerllabs/minerl)

- [Offline Reinforcement Learning from Images with Latent Space Models](https://proceedings.mlr.press/v144/rafailov21a.html)
  - Rafael Rafailov, Tianhe Yu, Aravind Rajeswaran, etc. *L4DC2021*
  - Key Words: Latent Space Models
  - ExpEnv: [DeepMind Control](https://github.com/deepmind/dm_control), [Adroit Pen](https://github.com/Farama-Foundation/D4RL), [Sawyer Door Open](https://github.com/suraj-nair-1/metaworld), [Robel D’Claw Screw](https://github.com/google-research/robel)

- [Pretraining Representations for Data-Efﬁcient Reinforcement Learning](https://papers.nips.cc/paper/2021/hash/69eba34671b3ef1ef38ee85caae6b2a1-Abstract.html)
  - Max Schwarzer, Nitarshan Rajkumar, Michael Noukhovitch, etc. *NeurIPS2021*
  - Key Words: latent dynamics modelling, unsupervised RL
  - ExpEnv: [Atari](https://github.com/openai/gym)

- [LM-Nav: Robotic Navigation with Large Pre-Trained Models of Language, Vision, and Action](https://arxiv.org/abs/2207.04429)
  - Dhruv Shah, Blazej Osinski, Brian Ichter, etc. *ArXiv2022*
  - Key Words: CLIP, ViNG, GPT-3
  - ExpEnv: None

- [Learning Generalizable Robotic Reward Functions from “In-The-Wild” Human Videos](https://arxiv.org/abs/2103.16817)
  - Annie S. Chen, Suraj Nair, Chelsea Finn. *RSS2021*
  - Key Words: Reward Functions, “In-The-Wild” Human Videos
  - ExpEnv: None

- [Reinforcement Learning with Videos: Combining Ofﬂine Observations with Interaction](https://arxiv.org/abs/2011.06507)
  - Karl Schmeckpeper, Oleh Rybkin, Kostas Daniilidis,etc. *CoRL2020*
  - Key Words: learning from videos
  - ExpEnv: [robotic pushing task](https://github.com/openai/mujoco-py), [Meta-World](https://github.com/rlworkgroup/metaworld)

- [Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents](https://arxiv.org/pdf/2201.07207.pdf)
  - Wenlong Huang, Pieter Abbeel, Deepak Pathak, etc. *ICML2022*
  - Key Words: large language models, Embodied Agents
  - ExpEnv: [VirtualHome](https://github.com/xavierpuigf/virtualhome)

- [Reinforcement Learning with Action-Free Pre-Training from Videos](https://proceedings.mlr.press/v162/seo22a.html)
  - Younggyo Seo, Kimin Lee, Stephen L James, etc. *ICML2022*
  - Key Words: action-free pretraining, videos
  - ExpEnv: [Meta-world](https://github.com/rlworkgroup/metaworld), [DeepMind Control Suite](https://github.com/deepmind/dm_control)

- [History Compression via Language Models in Reinforcement Learning](https://arxiv.org/abs/2205.12258)
  - Fabian Paischer, Thomas Adler, Vihang Patil, etc. *ICML2022*
  - Key Words: Pretrained Language Transformer
  - ExpEnv: [Minigrid](https://github.com/maximecb/gym-minigrid), [Procgen](https://github.com/openai/procgen)

- [Learning Actionable Representations with Goal-Conditioned Policies](https://arxiv.org/abs/1811.07819)
  - Dibya Ghosh, Abhishek Gupta, Sergey Levine. *ICLR2019*
  - Key Words: Actionable Representations Learning
  - ExpEnv: 2D navigation(2D Wall, 2D Rooms, Wheeled, Wheeled Rooms, Ant, Pushing)

- [Is Cross-Attention Preferable to Self-Attention for Multi-Modal Emotion Recognition?](https://arxiv.org/abs/2202.09263)
  - Vandana Rajan, Alessio Brutti, Andrea Cavallaro. *ICASSP2022*
  - Key Words: Multi-Modal Emotion Recognition, Cross-Attention
  - ExpEnv: None

- [How Much Can CLIP Benefit Vision-and-Language Tasks?](https://openreview.net/forum?id=zf_Ll3HZWgy)
  - Sheng Shen, Liunian Harold Li, Hao Tan, etc. *ICLR2022*
  - Key Words: Vision-and-Language, CLIP
  - ExpEnv: None

- [Grounding Language to Entities and Dynamics for Generalization in Reinforcement Learning](https://arxiv.org/abs/2101.07393)
  - Austin W. Hanjie, Victor Zhong, Karthik Narasimhan. *ICML2021*
  - Key Words: Multi-modal Attention
  - ExpEnv: [Messenger](https://github.com/ahjwang/messenger-emma)

- [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)
  - Danijar Hafner, Timothy Lillicrap, Ian Fischer, etc. *ICML2019*
  - Key Words: latent dynamics model, pixel observations
  - ExpEnv: [DeepMind Control Suite](https://github.com/deepmind/dm_control)

- [Decoupling Representation Learning from Reinforcement Learning](https://arxiv.org/abs/2009.08319)
  - Adam Stooke,Kimin Lee,Pieter Abbeel, etc. *ICML2021*
  - Key Words: representation learning, unsupervised learning
  - ExpEnv: [DeepMind Control](https://github.com/deepmind/dm_control), [Atari](https://github.com/openai/gym), [DMLab](https://github.com/deepmind/lab)

- [Masked Visual Pre-training for Motor Control](https://arxiv.org/abs/2203.06173)
  - Tete Xiao, Ilija Radosavovic, Trevor Darrell, etc. *ArXiv2022*
  - Key Words: self-supervised learning, motor control
  - ExpEnv: [Isaac Gym](https://developer.nvidia.com/isaac-gym)


## Contributing
Our purpose is to make this repo even better. If you are interested in contributing, please refer to [HERE](CONTRIBUTING.md) for instructions in contribution.


## License
Awesome Multi-Modal Reinforcement Learning is released under the Apache 2.0 license.
