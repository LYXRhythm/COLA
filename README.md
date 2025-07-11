# COLA
Learning Fully Unsupervised Cross-Domain Image Retrieval through Self-supervised Pseudo-labels

Authors: Yongxiang Li, Xiaoyun Ren, Xingbo Zhao, Qianwen Lu, and Qingchuan Tao

### Abstract
As diverse image types proliferate and the demand for efficient information retrieval intensifies, large-scale cross-domain image retrieval has become an important research filed. Despite great progress in supervised methods, fully unsupervised
cross-domain image retrieval (FUCIR) remains challenging due to the lack of category annotations and cross-domain correspondences. Existing FUCIR approaches predominantly focus
on learning unified semantic representations by maximizing cooccurrence information across domains. However, they usually neglect the intrinsic relationships among instances, hindering a deeper understanding of the data and leading to suboptimal
performance. To overcome these limitations, we propose a novel coarse-to-fine learning paradigm called COLA, which leverages
self-supervised pseudo-labels as supervised information to robustly learn both discriminative and domain-invariant representations. Specifically, COLA consists of two key components: Self-supervised Pseudo-labels Annotation (SPA) and Robust Representation Learning (RRL). SPA facilitates a deep understanding
of instance-level similarities and encapsulates discrimination into
the representations in a self-supervised learning manner, thereby
generating high-quality pseudo-labels. However, the inevitable
noise in pseudo-labels can introduce significant risks of overfitting
and unreliable cross-domain correspondences. To mitigate these
issues, RRL designs a novel loss function, Adaptive Robust Loss
(ARL), and combines it with contrastive constraint to improve
discrimination from imperfect predictions, effectively reducing
the impact of uncertain samples. This strategy benefits the
identification of decision boundaries that are robust to noise,
thereby minimizing the interference of noise during the learning
process. Extensive experiments on three public cross-domain
image datasets, along with comparisons to eight state-of-the-art
methods, demonstrate the superiority of COLA.

### Framework

![framework](https://github.com/user-attachments/assets/18b135db-6b5b-438f-ab19-4a50e5fc902a)


### Requirements
```
pip install -r requirements.txt
``` 

### Datasets
* The OfficeHome dataset could be downloaded from https://www.hemanthdv.org/officeHomeDataset.html.

* The Office31 and image_CLEF dataset could be downloaded from https://github.com/jindongwang/transferlearning/tree/master/data.

* The directory structure of datasets.

```
  --officehome
       --Art
       --Clipart
       --Real_World
       --Product
  --office31
       --amazon
       --dslr
       --webcam
       --...
  --...
```
### Quickly Training

#### Preheat to get fake tags
```
python train_setup1.py
```
* At the end of the SPA step, false labels with noise and image features corresponding to the data set class are obtained.
Then RRL is used to train the noisy data.
#### Noise pseudo-label training

* Before training, you need to align the images of different domains with cross-domain labels, and obtain pseudo-label data of json type.

```
python align_train(test)_data.py  && python train(test)_creat_json_label.py
```

```
python train_setup2.py
```

* You can use tensorboard to monitor training accuracy in real time, and startup files are saved in ```./log```.

* The training save weights are located in ```./dest_pth```, and you can use the weights and apply the framework for cross-domain image retrieval.

### Commands for Test
* You can perform cross-domain image retrieval by running test_pic.py, and save the top 10 similar images in the domain. The images are saved in```./similar_images_coda```, which you can view.
```
python test_picture.py
```

### Comparison with the State-of-the-Art Methods
* We conducted COLA unsupervised cross-domain image retrieval on three datasets to evaluate the performance of COLA and other methods.The graph shows that this framework and method significantly improve the ability of cross-domain retrieval compared with other methods.Diagrams comparing the Office31 data set to other methods are in the framework.

<div align=center>
  
#### Comparison with the state-of-the-art methods on Image-CLEF dataset.
![Image-CLEF](https://github.com/user-attachments/assets/d48607a7-7c73-4226-b342-4f77950ca717)

</div>

#### Comparison with the state-of-the-art methods on OfficeHome dataset.
![OfficeHome](https://github.com/user-attachments/assets/639cac22-3fb3-402d-a5e8-9bd5c71f09c8)



### Ablation Study
* In the study of ablation experiments, the important role of each component is verified. We analyze the impact of each component of COLA on the overall framework, 
and experiments on these three datasets show that it has enhanced cross-domain retrieval capabilities.

![Ablation-all](https://github.com/user-attachments/assets/75d3cda0-0ff4-4f47-9777-00c95bb6d715)


* Moreover, we examine the benefits of pseudo-labels generated by SPA in comparison with standard K-means, DBSCAN,
Hierarchical Clustering (HC), and recent PUMR, as shown in Table.

![k-mean](https://github.com/user-attachments/assets/22acfba3-a2fc-49f7-9f2a-b85fe6d4c178)

