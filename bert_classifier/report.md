# PEFT with BERT on Multi-task Training

## Abstract
[//]: (补注释，bert，lora，prefix，以及三个数据集都要补，可以看范文，里面有三个数据集的cite)
In this project, we use the BERT model to perform multi-task training on three different datasets: SST-5 for sentiment analysis, QQP for paraphrase detection, and STS-B for semantic textual similarity. We implement two popular PEFT algorithms, LoRA tuning and prefix tuning, to fine-tune the BERT model on these tasks. We compare the performance of these two algorithms along with the performance of the BERT model with only downstream towers trained, and the BERT model with all layers fine-tuned. According to our experiments, there is no guarantee that PEFT algorithms can perform better than fine-tuning all layers, but we do find some advantages of using PEFT algorithms.


## Introduction
[//]: (补注释，bert，lora，prefix，以及三个数据集都要补，可以看范文，里面有三个数据集的cite)
Nowadays, more and more NLP tasks are adopting large pretrained language models (PLMs) as their backbone. However, fine-tuning these PLMs on downstream tasks is computationally expensive and time-consuming. There may also be other issues, such as catastrophic forgetting, and overfitting, when fine-tuning is performed on the entire model. To address these issues, researchers have proposed various methods to fine-tune PLMs more efficiently. One of the most popular methods is parameter-efficient fine-tuning (PEFT), which fine-tunes only a subset of the model's parameters. 

In this project, we implement two PEFT algorithms, LoRA tuning and prefix tuning, to fine-tune the BERT model on three different NLP tasks at the same time. We compare the performance of these two algorithms along with the performance of the BERT model with only downstream towers trained, and the BERT model with all layers fine-tuned. By comparing the performance of different training methods, we aim to understand the advantages and disadvantages of using PEFT algorithms.


## Related Works

+ [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

+ [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

+ [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)


## Approach

### Three Datasets and Downstream Towers
We train the BERT model on three different datasets at the same time.
+ SST-5: The Stanford Sentiment Treebank dataset, which contains 5 classes of sentiment labels. We use a two-layer MLP as the downstream tower for this dataset. The pooler output of the BERT model is taken as the input of the tower. The tower outputs a logit vector of 5 units, which is then passed through a softmax layer. Cross-entropy loss is used as the loss function.
+ QQP: The Quora Question Pairs dataset, which contains binary labels indicating whether two questions are paraphrases of each other. We use a three-layer MLP as the downstream tower for this dataset. The last states of the two sentences are calculated by the BERT model individually, and then concatenated and passed through the tower. The tower outputs a logit vector of 2 units, which is then passed through a softmax layer. Cross-entropy loss is used as the loss function.
+ STS-B: The Semantic Textual Similarity Benchmark dataset, which contains continuous similarity scores between two sentences. We use a three-layer MLP as the downstream tower for this dataset. The last states of the two sentences are calculated by the BERT model individually, and then concatenated and passed through the tower. The tower outputs a scalar value, which is then passed through a mean squared error loss function.
In each training step, the three losses are sumed up to do back-propagation.

### AdamW Optimizer
[//]: (写一下AdamW的主要公式，然后提一句我们用了原文里的加速方法，说一下AdamW的好处，记得cite)

### LoRA Tuning
One of the PEFT algorithms we implement is LoRA tuning. LoRA tuning is a low-rank adaptation method that fine-tunes only a subset of the model's parameters. Here we can see from the image how LoRA tuning works when fine-tuning a linear layer.

[//]: (这里可以加一张图，展示LoRA_tuning的原理)

The original weight matrix W with dimensions $d_{in} \times d_{out}$ is decomposed into two low-rank matrices $A$ and $B$ with dimensions $d_{in} \times r$ and $r \times d_{out}$, respectively. The weight matrix $W$ is then approximated by the product of $A$ and $B$. Thus, the formula of the linear layer becomes as follows:
$$
\text{original: } y = Wx + b
$$

$$
\text{LoRA: } y = (W + B A^T)x + b
$$

During training, the low-rank matrices $A$ and $B$ are updated by the optimizer, while the original weight matrix $W$ remains fixed. 

Noticing that when training with LoRA, usually a scaling factor $\gamma$ is introduced. The formula of the linear layer becomes as follows:
$$
\text{LoRA with scaling factor: } y = (W + \gamma B A^T)x + b
$$

In (cite the LoRA paper), the scaling factor $\gamma$ is adaptive to the LoRA rank $r$. The formula of the scaling factor is as follows:
$$
\gamma = \frac{\alpha}{r}
$$

where $\alpha$ is a hyperparameter that controls the scale of the scaling factor.

However, this may cause the scaling factor to be too small when the rank $r$ is large. (cite the sqrt paper) mentioned this issue and proposed a new scaling factor formula:
$$
\gamma = \frac{\alpha}{\sqrt{r}}
$$

In our experiments, to make the comparison between different LoRA ranks simpler and more fair, we directly set the scaling factor $\gamma$ to a fixed value 1.

The advantage of LoRA tuning is that the low-rank matrices can be merged into the original weight matrix, no additional modules are introduced to the prediction process.


### Prefix Tuning
Another PEFT algorithm we implement is prefix tuning. Prefix tuning is a method that optimizes continuous prompts for generation. The idea is to add a prefix token to the input sequence and fine-tune only the prefix token. The image below shows how the prefix tuning we implemented works for one layer of the BERT model. In all layers of the BERT model, the prefix token is added to the input sequence, and the prefix token is fine-tuned while the rest of the model remains fixed.

[//]: (这里可以加一张图，展示prefix_tuning的原理)

Noticing that (cite the prefix paper) mentioned that if the prefix token is directly updated by back-propagation, the performance may be sensitive to the hyper-settings, such as the initialization of the prefix token and the learning rate. To address this issue, (cite the prefix paper) feeds the parameters of the prefix token into a MLP to get the final prefix token that will be appended to the input sequence. During training, the MLP is updated, which avoids the direct update of the prefix token. Our experiments show that this method is more stable and can achieve better performance.

The advantage of prefix tuning is that the parameters trained during the tuning process are very few, and the prefix token can be easily saved and used in the prediction process, without changing anything of the original model.


## Experiments
We conduct a series of experiments to compare the performance of the BERT model with different training methods. If not explicitly mentioned, the experiments are run on the multi-task training, the hyperparameters of the training process are set to the same values for all training methods, where the learning rate is 1e-5, the batch size is 8, and the model is trained for 25 epochs. 

We have a main group of experiments that compares the performance of the BERT model with different training methods on the three datasets. We also have some additional experiments that investigate the influence of different hyperparameters on the performance of the PEFT algorithms. They are listed as follows:
+ Main experiment:
    We compare the performance for the four training methods: pretrained BERT with downstream towers trained, pretrained BERT with all layers fine-tuned, BERT with LoRA tuning, and BERT with prefix tuning.
+ Epochs needed:
    We train all the four methods for up to 75 epochs to see the performance of the models when trained for a longer time.
+ LoRA rank:
    We investigate the influence of different LoRA ranks on the performance of the LoRA tuning method.
+ Prefix token length:
    We investigate the influence of different prefix token lengths on the performance of the prefix tuning method. We also compare the performance of the prefix tuning method with and without the MLP that feeds the parameters of the prefix token into it.



## Results and Analysis

### Main Experiment
In this experiment, the LoRA rank is set to 8, and the prefix token length is set to 4.
The results of the main experiment are shown in the table below. The table shows the accuracy (for SST-5 and QQP) and the Pearson correlation coefficient (for STS-B) of the four training methods on the three datasets. The best performance for each dataset is highlighted in bold.

[//]: (需要整理表格1，只要dev_acc就行)

From the results, we can see that the performance of fine-tuning all layers is the best on SST-5 and QQP, while the performance of prefix tuning is the best on STS-B. The prefix tuning method also achieves the second-best performance on SST-5. However, the performance of LoRA tuning is not satisfying, compared to the other methods. It achieves 10% higher accuracy on SST-5, 2% lower accuracy on QQP, and relatively the same Pearson correlation coefficient on STS-B, compared to the pretrained BERT with downstream towers trained. None of the results of LoRA tuning outperforms the fine-tuning all layers method.

### Epochs Needed
In this experiment, we train all the four methods for up to 75 epochs. The LoRA rank is set to 8, and the prefix token length is set to 4. The training results are recorded at epoch 10, 25, 50, and 75. The results are shown in the table below.

[//]: (需要整理表格2，需要train_acc和dev_acc，从而说明overfitting，不要把pretrain放进来)

From the results, we can see that the fine-tuning all layers method reaches the ideal performance at an early stage. The training accuracy of it becomes very high when trained for a longer time, while the development accuracy does not increase much, indicating that the model is overfitting. The overfitting problem is not as severe for LoRA tuning, and we can see that the difference between the training accuracy and the development accuracy is smaller than the fine-tuning all layers method. The LoRA tuning method reach 10% better accuracy on STS-B than the fine-tuning all layers method when trained for 75 epochs, and the difference between the other two tasks is also less significant compared with the early stage. Thus, we can see that LoRA tuning may need more epochs to reach the ideal performance, but it does not overfit as much as the fine-tuning all layers method. For the prefix tuning method, as the parameters trained are very few, the overfitting problem for it is quite slight. The performance of it on STS-B is the best.

From the results we can also spot out an important feature of the datasets. The STS-B dataset is easier to overfit than the other two datasets, which also affects the results of the other experiments.

### LoRA Rank
In this experiment, we investigate the influence of different LoRA ranks on the performance of the LoRA tuning method. The LoRA rank is set to 2, 4, 8, 16, 32, and 64. All the linear modules in the BERT model are fine-tuned with the LoRA tuning method. The results are shown in the table below.

[//]: (需要整理表格3，需要train_acc和dev_acc)

From the results, we can see that the performance on SST-5 and QQP increases as the LoRA rank increases. However, the performance on STS-B decreases as the LoRA rank increases. The best performance on SST-5 and QQP is achieved when the LoRA rank is set to 64, while the best performance on STS-B is achieved when the LoRA rank is set to 2. Checking the training correlation coefficient, we can see that it is actually growing as the LoRA rank increases, which indicates that the model is overfitting. But to view the performance on all three datasets, we can say that bigger LoRA rank leads to better performance.

An interesting issue is that in (cite the LoRA paper), the experiments show that the performance is relatively the same or even slightly getting worse when the LoRA rank is getting bigger. As we mentioned before, this can be caused by the setting of the LoRA scaling factor. In our experiments, we set the scaling factor to a fixed value 1, and we did not observe the same issue as the original paper did.

### Prefix Token Length
In this experiment, we investigate the influence of different prefix token lengths on the performance of the prefix tuning method. The prefix token length is set to 1, 2, 4, and 8. The results are shown in the table below.

[//]: (需要整理表格4，只要dev就行)

We also compare the performance of the prefix tuning method with and without the MLP that feeds the parameters of the prefix token into it. The results are shown in the table below.

[//]: (需要整理表格5，只要dev就行，有问题的那种方法的length=1我忘记跑了，这张表里不要length=1算了)

We can see that updating the MLP of the prefix token rather than the prefix token directly can achieve significantly better performance. The best performance is achieved when the prefix token length is set to 4, and the MLP is used. From the results, we can also see that the performance of different prefix token lengths is not very different, which indicates that the performance of the prefix tuning method is not very sensitive to the prefix token length. The prefix tuning method does not really need a long prefix token to achieve good performance.



## Future Work
+ Better downstream tower for the STS-B dataset: The training of the STS-B dataset is easily overfitting. This may be caused by the simple structure of the downstream tower we used for this dataset. We may need to introduce more complex structures to the tower to achieve better performance. For example, we may need cosine similarity as the output of the tower, which is more reasonable comparing to setting the MLP to output a scalar value.
+ Linear modules chosen for LoRA tuning: In our experiments, we fine-tune all the linear modules in the BERT model with the LoRA tuning method. However, we may need to choose the linear modules that are more important for the tasks to fine-tune. This may help to achieve better performance.
+ More PEFT algorithms: In this project, we only implement two PEFT algorithms, LoRA tuning and prefix tuning, and none of them can be guaranteed to perform better than fine-tuning all layers. We may need to explore more PEFT algorithms to find a better way to fine-tune PLMs on downstream tasks.
+ Multi-task training: In this project, we have not considered the influence of training the BERT model on multiple tasks at the same time. We may need to investigate the influence of multi-task training on the performance of the PEFT algorithms, and analyze whether the model can learn better when trained on multiple tasks at the same time.

## Conclusion
In this project, we implement two PEFT algorithms, LoRA tuning and prefix tuning, to fine-tune the BERT model on three different NLP tasks at the same time. We compare the performance of these two algorithms along with the performance of the BERT model with only downstream towers trained, and the BERT model with all layers fine-tuned. According to our experiments, there is no guarantee that PEFT algorithms can perform better than fine-tuning all layers, but we do find some advantages of using PEFT algorithms. The LoRA tuning method may need more epochs to reach the ideal performance, but it does not overfit as much as the fine-tuning all layers method. The prefix tuning method can achieve good performance with very few parameters trained. We also find that the performance of the two PEFT algorithms may not be very sensitive to the hyperparameters, such as the LoRA rank and the prefix token length. Having a chance to implement the two PEFT algorithms and test them on three different datasets, we have a better understanding of the advantages and disadvantages of using PEFT algorithms.


