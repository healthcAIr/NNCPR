# Practical Applications of Deep Learning: Classifying the most common categories of plain radiographs in a PACS using a neural network (NNCPR)

This is the code for the European Radiology paper

[**Practical Applications of Deep Learning: Classifying the most common categories of plain radiographs in a PACS using a neural network**](https://github.com/healthcAIr/NNCPR)  
Thomas Dratsch; Michael Korenkov; David Zopfs; Sebastian Brodehl; Bettina Baessler; Daniel Giese; Sebastian Brinkmann; David Maintz; Daniel Pinto dos Santos  
European Radiology

The goal of this project was to classify the 30 most common types of plain radiographs using a neural network and to validate the network's performance on internal and external data.
Such a network could help improve various radiological workflows.

Using data from one single institution, we were able to classify the most common categories of plain radiographs with a neural network.
The network showed good generalizability on the external validation set and could be used to automatically organize a PACS, preselect radiographs so that they can be routed to more specialized networks for abnormality detection or help with other parts of the radiological workflow (e.g., automated hanging protocols; check if ordered image and performed image are the same). 

For more details on the project and performance metrics of the model, please refer to the final publication.

The model can classify the following 30 categories of plain radiographs:

| A - E                          | F - L                | O - Z                      |
| ------------------------------ | -------------------- | -------------------------- |
| abdomen_ap                     | finger_ap            | oblique_view_of_Lauenstein |
| abdomen_left_lateral_decubitus | finger_lateral       | panoramic_radiograph       |
| ankle_ap                       | foot_ap              | patella_axial              |
| ankle_lateral                  | foot_oblique         | pelvis_ap                  |
| cervical_spine_ap              | hand_ap              | shoulder_ap                |
| cervical_spine_lateral         | hand_oblique         | shoulder_outlet            |
| chest_lateral                  | knee_ap              | thoracic_spine_ap          |
| chest_pa_ap                    | knee_lateral         | thoracic_spine_lateral     |
| elbow_ap                       | lumbar_spine_ap      | wrist_ap                   |
| elbow_lateral                  | lumbar_spine_lateral | wrist_lateral              |

## Requirements

### Dependencies

- Python3
- Tensoflow v1.15.2
- Numpy
- Docker (optional)
- Virtualenv (optional)

### Setup

Clone this repository 

```bash
git clone https://github.com/healthcAIr/NNCPR
cd NNCPR
```

#### Docker

Setup the Docker container

```bash
# pull the needed image (Tensoflow v1.15.2 GPU variant with Python3)
docker pull tensorflow/tensorflow:1.15.2-gpu-py3
# run the container interactively
docker run -it -u $(id -u):$(id -g) --rm -v $PWD:/tmp -w /tmp tensorflow/tensorflow:1.15.2-gpu-py3 bash
```

#### Virtualenv

Create a virtual environment for the project and activate it:

```bash
# create a virtualenv in directory venv
virtualenv venv
# active venv
source venv/bin/activate
# install needed packages
pip install -r requirements.txt
```

#### Verify TensorFlow Installation

Let's verify the TensorFlow installation by executing a simple operation

```bash
python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

Which should output

```
Tensor("Sum:0", shape=(), dtype=float32)
```

## Usage

We will download a plain radiograph [showing a knee from Wikimedia (CC BY-SA 4.0 license)](https://commons.wikimedia.org/wiki/File:Knee_plain_X-ray.jpg) and save it as `example.jpg`.

```shell
curl -o example.jpg https://upload.wikimedia.org/wikipedia/commons/f/f0/Knee_plain_X-ray.jpg
```

Let's take a quick look and verify what we have downloaded:

![Knee plain X-ray.jpg](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f0/Knee_plain_X-ray.jpg/256px-Knee_plain_X-ray.jpg)

Now we can run this image through the trained network with the following command:

```shell
python label_image.py example.jpg
```

You will then get an output that includes the name of the image, the predicted class, the prediction value and the time it took to classify the image. 

Here is the output for the example image from above:

```shell
example.jpg 	 knee_lateral 	 0.99953 	 0.21s
```

The input image `example.jpg` was classified as `knee_lateral` with a prediciton value of `0.99953` in `0.21` seconds.

To classify multiple images, use the following shell command (just append multiple file paths):

```shell
python label_image.py image1.jpg image2.jpg image3.jpg ... imageN.jpg
```

### Important

1) The model can only classify the 30 categories of plain radiographs listed above.
   Using images from other than these 30 categories will produce incorrect results.
2) The predictions of the model can be inaccurate and the accuracy of the model varies between classes.
   However, the distribution of the sensitivity of the model was rather balanced across classes, ranging between 61.0% and 100.0%.
   18 out of 30 categories (60.0%) reached a sensitivity of over 90.0%, and 27 out of 30 categories (90.0%) reached a sensitivity over 80.0%.
   Please see the final paper for detailed performance metrics for all 30 classes.


classes.txt:
Contains the names of all 30 classes that the model can classify. This file is used by the script to label the images. If the order of the classes in this file is changed, predictions by the model will be incorrect.

network.pb:
The final network used to classify the images.

label_image.py:
Script for classifying images.

## Cite

Please cite our paper if you use this code in your own work:

```
tba
```

## License

[Apache License 2.0](LICENSE)

```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

