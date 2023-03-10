# Unsupervised-Domain-Adaptation-on-Adaptiope-Dataset 

*Trento University Computer Science Department Deep Learning Course Project*
*   Prof: [Elisa Ricci](http://elisaricci.eu/) 
*   Teaching Assistant: [Giacomo Zara](https://gzaragit.github.io/)

In this assignment, we build, train and evaluate a deep learning framework on a standard setting of Unsupervised Domain Adaptation (UDA)  tasks using the ResNet18 and EfficentNet pre-trained model by PyTorch framework.The Adaptiope object recognition dataset consists of images from three different domains: synthetic, product, and real-world. For this assignment, we will be working with a subset of the Adaptiope dataset, using an 80/20 training-testing split. 
The model we used is an adversarial neural network inspired by [Unsupervised Domain Adaptation by Backpropagation](https://arxiv.org/abs/1409.7495). The architecture includes a deep feature extractor and a deep label predictor, which together form a standard feed-forward architecture. In particular we wanted to test how it is possible to adapt the architecture proposed in the paper to different pre-trained networks.
We first train the model using supervised learning on the source domain and evaluate it as is on the target domain. This approach is referred to as the source-only or baseline version. To achieve this, we simply ignore the domain classifier branch and provide the model with only source domain images and labels during training. In the next step of domain adaptation, we activate the UDAB plugin by also training on the unlabeled target dataset. In this stage, we incorporate the loss of the domain classifier branch.
<p align="center">
    <img width="600" src="https://i.imgur.com/BwQZMXb.png">
</p>

**Students**
*  Tesfaye Naramo
*  Riccardo Ratta

## Pytorch version
```python
pytorch 1.9.0+cu102
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Contact
Created by [@tesfayeamare](https://github.com/tesfayeamare) - feel free to contact me!

## License
[MIT](https://choosealicense.com/licenses/mit/)

