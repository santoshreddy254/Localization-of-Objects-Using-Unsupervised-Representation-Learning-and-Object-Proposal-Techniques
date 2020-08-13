# Localization-of-Objects-Using-Unsupervised-Representation-Learning-and-Object-Proposal-Techniques

## Requirements


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




