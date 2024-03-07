# Automatic Wood Pith Detection: Local Orientation Estimation and Robust Accumulation


[link_urudendro]: https://iie.fing.edu.uy/proyectos/madera/


Version 1.0
Last update: 26/11/2023. **Code and dataset will be available after approval of the paper.**




## Installation

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
## License
License for th source code: [MIT](./LICENSE)

