# Localization-of-Objects-Using-Unsupervised-Representation-Learning-and-Object-Proposal-Techniques

## Requirements


## Dataset structure
* PASCAL VOC dataset
--root
---VOC2012
----Annotations
----ImageSets
----JPEGImages
----SegmentationClass
----SegmentationObject

* YCB dataset
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

* RoboCup@Work dataset
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

### Training
```
python3 train.py --batch_size,
                 --save_after,
                 --num_epochs,
                 --dataset,
                 --dataset_path,
                 --backbone,
                 --experiment_number,
                 --num_epochs,
```




