# Person - Mask detector
This model analyses the photos/frames from videos whether it contains person. If a person is detected, the model then predicts if mask is present or not

Project Setup
`requirements.txt` contains all the dependencies required for the project.

`train/` dir contains the ```MobileNet``` model backed transfer learning, with  [MaskNet dataset ](https://arxiv.org/abs/2008.08016)


## Architecture used 
  1. Person Detect - [YOLO](https://arxiv.org/abs/1506.02640)
  2. Face Detection - To boost Person detection Confidence - [MTNN](https://arxiv.org/abs/1604.02878)
  3. Mask Detection - Transfer learning from [MobileNetV2](https://arxiv.org/abs/1801.04381)

