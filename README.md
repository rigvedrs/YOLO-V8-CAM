
# EigenCAM for YOLO V8 Interpretability

A package for applying EigenCAM on the new YOLO V8 model. Simply clone the package and import the modules to get started.

The basic structure is close to [Jacob Gil's package for AI explainability](https://github.com/jacobgil/pytorch-grad-cam) and modified to be used for the YOLO V8 model.



## Use Cases 

It can currently be used for YOLO V8 Classification and Object Detection models. Support for the other tasks will be added. You can also send pull request for adding more functions to it.

## What is EigenCAM
EigenCAM is a technique that involves computing the first principle component of the 2D activations in a neural network, without taking class discrimination into account, and has been found to produce effective results.

#### Image:

<img src="images/puppies.jpg" alt="puppies" width="240" height="240">


#### GrayScale Heatmaps:

| Object Detection  | Classification  |
| -------- | -------- |
| ![od3.png](images/od3.png)  | ![cls3.png](images/cls3.png) |

#### Combined

| Object Detection  | Classification  |
| -------- | -------- |
| ![od1.png](images/od1.png)  | ![cls1.png](images/cls1.png) |

### Object Detection model
![od1.png](images/od2.png)

### Classification model
![od1.png](images/cls2.png)



## Getting Started

#### Simply clone this repository or just download the yolo_cam folder. You must have the yolo_cam folder in the same location as your notebook

#### Import the libraries first:
```python
from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image, scale_cam_image
```

#### Call the function and print the image

```python
cam = EigenCAM(model, target_layers,task='cls')
grayscale_cam = cam(rgb_img)[0, :, :]
cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
plt.imshow(cam_image)
plt.show()
```

#### For the Object Detection Task, just change the task to 'od' and the rest is same.

```python
cam = EigenCAM(model, target_layers,task='od')
```
The default task is 'od' so it is fine even if you don't specify the task then

#### Check out the Jupyter Notebook titled YOLO v8n EigenCAM to understand it better and also handle some issues.


## ToDo:

See the [open issues](https://github.com/rigvedrs/Yolo-V8-CAM/issues) for a list of proposed features (and known issues).

- Solve the issue with having to re-run the cells
- Add the task for handling pose and segmentation model as well




## Contributing

The open source community thrives on contributions, making it an incredible space for learning, inspiration, and creativity. Please feel free to share any contributions you have for this project.

-  Create your Feature Branch (`git checkout -b feature/CoolFeature`)
-  Commit your Changes (`git commit -m 'Add some CoolFeature'`)
-  Push to the Branch (`git push origin feature/CoolFeature`)
-  Open a Pull Request



## License

Distributed under the MIT License. See `LICENSE` for more information.
## Contact
[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white "My email ID")](mailto:rigvedrs@gmail.com)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white "Visit my LinkedIn profile")](https://www.linkedin.com/in/rigvedrs/)



