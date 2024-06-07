Implementation of the paper: [text](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Attributable_Visual_Similarity_Learning_CVPR_2022_paper.pdf)

Quick summary of the paper:
(The paper use the word similarity, but it's rather more of a distance than a similarity, values close to 0 indicate similar values, higher values indicate distant values)
The AVSL model is built on top of a CNN model. It learns a distance/similarity metric between images. Images of the same classes should have a small distance between them, and images of different classes should have a bigger difference. The model projects several feature maps of the CNN to several embedding spaces then computes distances between images, for different level of deepness. The intuition is that not only high-level features are important to learn similarity, but also low-level features. The AVSL then combines all of the distances at different level of deepness of the CNN model, to create a more refined distance according to the formula, (from bottom to top):

$$\hat{\delta}^1 = \delta^1 $$
$$ \forall l \geq 2 \ \ \ \ \ \hat{\delta}^l = P^l \ \delta^l + (I \ - \ P^l) \ \hat{W}^l \ \hat{\delta}^{l-1}$$


To run the model:

> create a python env
> pip install requirements.txt
> run the command in demo.sh

The source code is in the folder code, the important files are main.py, model.py, train.py, inference.py, losses.py.

- base_models.py loads a pretrained CNN model (resnet or efficientnetv2) and decomposes it, so as to be used by the AVSL model.
- model.py creates the AVSL model, it takes the feature maps of the CNN (the base model) and projects them to an embedding space
with a conv 1x1 layer (equivalent to flattening + a fc layer), it computes distances. It also computes a graph of similarity, which gives the matrices P and W of the formula above. Then the model computes the final distance from bottom to top. For the training loss we use the ProxyAnchorLoss.
- losses.py implements the ProxyAnchorLoss (paper: https://openaccess.thecvf.com/content_CVPR_2020/papers/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.pdf)
- train.py trains the model with the ProxyAnchorLoss.
- inference.py computes the distance between images for an AVSL model.
- main.py builds the complete pipeline to create the dataset, dataloader, create/load the model, train the model, infer the distances and evaluate the model. The runs are saved in runs/
- dataset.py creates the classes for the dataset
- resnet_baseline.py is simply a pretrained resnet50 then finetuned to classify the birds, it serves as a baseline.
  




