# Localization-of-Objects-Using-Unsupervised-Representation-Learning-and-Object-Proposal-Techniques

## Training Phase
* Training a deep neural network as a classification model
* Only class level labels are available
![alt text](images/01_Introduction_Image.png)

## Testing Phase
* Trained classification model is used to localize objects
* Generate object level and pixel level labels during testing
![alt text](images/02_Introduction_Image_testing.png)

## Class Activation Mapping (CAM)
* Class Activation Mapping: the confidence score of predicted class is projected back to the preceding
convolution layer to generate the class activation maps (CAMs)
![alt text](images/cam_pipeline.png)
## Grad-CAM
* Gradient Class Activation Mapping architecture. Grad-CAM uses gradient flow passing through last
convolutional layer to generate weights
![alt text](images/grad_cam_pipeline.png)


## Requirements
```
pip install -r requirements.txt
```
* Given in requirements file

## Dataset structure
* PASCAL VOC dataset
```
--root
---VOC2012
----Annotations
----ImageSets
----JPEGImages
----SegmentationClass
----SegmentationObject
```
* YCB dataset
```
--root
---train
----001_chips_can
-----masks
---- ....
----077_rubiks_cube
-----masks
---val
----001_chips_can
-----masks
---- ....
----077_rubiks_cube
-----masks
```
* RoboCup@Work dataset
```
--root
---images
----train
-----axis
----- ....
----- motor
----val
-----axis
----- ....
----- motor
```
### Training
```
python3 train.py --batch_size: BATCH SIZE (int),
                 --save_after: Save weights after every n epochs,
                 --num_epochs: number of epochs,
                 --dataset: Have option of three datsets VOC, YCB, at_work,
                 --dataset_path: Give the path to dataset,
                 --backbone: Have option of three backbones vgg16, resnet18, squeezenet1_1,
                 --experiment_number: Give unique identity to experiment
```

### Evaluation for classification metrics
```
python3 classification_eval.py --batch_size: BATCH SIZE (int),
                 	       --dataset: Have option of three datsets VOC, YCB, at_work ,
                               --dataset_path: Give the path to dataset,
                               --backbone: Have option of three backbones vgg16, resnet18, squeezenet1_1,
                               --checkpoint_path: Path for trained checkpoint
```

### Evaluation for localization metrics
Evaluation on localization metrics for different datasets is carried out differently due to very different structure of datasets.
* PASCAL VOC dataset
```
python3 localization_eval_VOC.py --wsol_method: select cam or gradCAM,
                               --dataset_path: Give the path to dataset,
                               --backbone: Have option of three backbones vgg16, resnet18, squeezenet1_1,
                               --checkpoint_path: Path for trained checkpoint
```
* YCB dataset
```
python3 localization_eval_YCB.py --wsol_method: select cam or gradCAM,
                               --dataset_path: Give the path to dataset,
                               --backbone: Have option of three backbones vgg16, resnet18, squeezenet1_1,
                               --checkpoint_path: Path for trained checkpoint,
                               --masks_path: Give the path to masks dataset
```
* RoboCup@Work dataset
```
python3 localization_eval_atwork.py --wsol_method: select cam or gradCAM,
                               --dataset_path: Give the path to dataset,
                               --backbone: Have option of three backbones vgg16, resnet18, squeezenet1_1,
                               --checkpoint_path: Path for trained checkpoint,
                               --masks_path: Give the path to masks dataset
```

#### Code base is adapted from following repositories
* https://github.com/clovaai/wsolevaluation
* https://github.com/jacobgil/pytorch-grad-cam




