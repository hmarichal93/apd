# Automatic Wood Pith Detection: Local Orientation Estimation and Robust Accumulation

[ICPR 2024](https://link.springer.com/chapter/10.1007/978-3-031-78447-7_1) | [Arxiv](https://arxiv.org/abs/2404.01952) | [Slides](./assets/apd.pdf)

[link_urudendro]: https://iie.fing.edu.uy/proyectos/madera/

***
**Pith detection at the right image (blue dot)**
![F02b_input_output.png](assets%2FF02b_input_output.png)
***

## Installation
```bash
conda create --name pith python=3.11
conda activate pith
conda install -n pith pip
```
```bash
pip install .
```

## Dataset

```bash
python fetch_dataset.py
```

## Examples of usage
### Import the module
```python
from automatic_wood_pith_detector.automatic_wood_pith_detector import  apd, apd_pcl, apd_dl
import cv2 

st_sigma = 1.2
st_w = 3
lo_w = 11
percent_lo = 0.5

img_in= cv2.imread('./Input/F02c.png')
peak = apd(img_in, st_sigma, st_w, lo_w, rf = 7, percent_lo = percent_lo, max_iter = 11, epsilon =10 ** -3)

```

### CLI
Example of usage:
```bash
python main.py --filename ./Input/F02c.png --output_dir Output/ --new_shape 640 --debug 1
```

Example of usage with pclines postprocessing
```bash
python main.py --filename ./Input/F02b.png --output_dir Output/ --new_shape 640 --debug 1 --method 1
```

Example of usage with apd-dl
```bash
python main.py --filename ./Input/F02b.png --output_dir Output/ --new_shape 640 --debug 1 --method 2
```

## Citation
If you use this code, please cite the following paper:

```
@InProceedings{marichal2024automatic,
author="Marichal, Henry and Passarella, Diego and Randall, Gregory",
editor="Antonacopoulos, Apostolos and Chaudhuri, Subhasis and Chellappa, Rama and Liu, Cheng-Lin and Bhattacharya, Saumik and Pal, Umapada",
title="Automatic Wood Pith Detector: Local Orientation Estimation and Robust Accumulation",
booktitle="Pattern Recognition",
year="2025",
publisher="Springer Nature Switzerland",
address="Cham",
pages="1--15",
isbn="978-3-031-78447-7"
}
```

## License
License for th source code: [MIT](./LICENSE)


