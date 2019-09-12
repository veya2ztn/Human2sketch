# sketchKeras

## enhanced

This project base on [**sketchKeras**](https://github.com/lllyasviel/sketchKeras)

The original sketchKeras is proposed for animate characters like

![img](https://raw.githubusercontent.com/lllyasviel/sketchKeras/master/test1/raw.jpg)

Let's train the model for 3D real person

![](https://raw.githubusercontent.com/veya2ztn/Human2sketch/master/example.jpg)

----------

#### Goal

- human -> sketch with good performance
- underline hair so it more like a picture

#### DataSet:

we joint anime dataset and real human data.

- The anime dataset is crawled online data follow [GirlsManifold](https://github.com/shaform/GirlsManifold)
- The real human data is  [CUFS dataset](http://mmlab.ie.cuhk.edu.hk/archive/facesketch.html) 

The anime dataset is much bigger than real human data.

But the result seems good.

The weight [here](https://www.dropbox.com/s/azu6akp7mhnhb25/human2sketch.h5?dl=0)

```python
from tensorflow.keras.models import load_model
from helper import *
MODEL_PATH_1=""
mod = load_model(MODEL_PATH_1)

img=cv2.imread("1.jpg")
rgb_print(img)

sketch = get_sketch(img,mod)
rgb_print(sketch)
```

