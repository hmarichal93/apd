# Automatic Wood Pith Detection: Local Orientation Estimation and Robust Accumulation


[link_urudendro]: https://iie.fing.edu.uy/proyectos/madera/


Version 1.0
Last update: 26/11/2023. 

**Pith detection at the right image (blue dot)**
![F02b_input_output.png](assets%2FF02b_input_output.png)


## Installation
```bash
conda create --name pith python=3.11
conda activate pith
conda install -n pith pip
```
```bash
pip3 install --no-cache-dir -r requirements.txt
```

## Dataset

```bash
python fetch_dataset.py
```

## Download pretrained model
```bash
python fetch_pretrained_model.py
```

## Examples of usage

Example of usage:
```bash
python main.py --filename ./Input/F02c.png --output_dir Output/ --new_shape 640 --debug 1
```

Example of usage with pclines postprocessing
```bash
python main.py --filename ./Input/F02b.png --output_dir Output/ --new_shape 640 --debug 1 --pclines 1
```

## Citation
If you use this code, please cite the following paper:

```
@misc{marichal2024automatic,
      title={Automatic Wood Pith Detector: Local Orientation Estimation and Robust Accumulation}, 
      author={Henry Marichal and Diego Passarella and Gregory Randall},
      year={2024},
      eprint={2404.01952},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License
License for th source code: [MIT](./LICENSE)


