# README.md

## Dependencies
- numpy
- scipy
- scikit-learn
- matplotlib
- opencv-python


## How to run
- run the following command
    ```bash
    python /release.py
    ```
- result will contain the composited image and the alpha matte image in the `./result` folder

## Image
if you want to use the image or parameter for KNN on your own, you can do the following steps
- put image in the `./image` folder
- put trimap image in the `./trimap` folder
- put the background image in the `./background` folder
- then modify the these line in `./release.py` file
```python
# release.py line 108
if __name__ == '__main__':
    image_name = 'troll' # your image/trimap name
    bg_image_name = 'sea' # your background image name
    features = 'HSV' # RGB or HSV color space feature
    n_neighbors = 10 # number of neighbors
```
- then run `python /release.py`
- alpha image and composting result will be in the `./result` folder

note that we only support the image with `.png` format for now. 