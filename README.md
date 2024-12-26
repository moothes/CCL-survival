# CCL-survival

Code for our paper ``Cohort-Individual Cooperative Learning for Multimodal Cancer Survival Analysis'' [[Paper](https://ieeexplore.ieee.org/document/10669115)]


### Prepare your data
#### WSIs
1. Download diagnostic WSIs from [TCGA](https://portal.gdc.cancer.gov/)
2. Use the WSI processing tool provided by [CLAM](https://github.com/mahmoodlab/CLAM) to extract resnet-50 pretrained 1024-dim feature for each 256 $\times$ 256 patch (20x), which we then save as `.pt` files for each WSI. So, we get one `pt_files` folder storing `.pt` files for all WSIs of one study.

The final structure of datasets should be as following:
```bash
DATA_ROOT_DIR/
    └──pt_files/
        ├── slide_1.pt
        ├── slide_2.pt
        └── ...
```

DATA_ROOT_DIR is the base directory of cancer type (e.g. the directory to TCGA_BLCA).

#### Genomics
In this work, we directly use the preprocessed genomic data provided by [MCAT](https://github.com/mahmoodlab/MCAT), stored in folder [csv](./csv).

## Usage

```
python main.py CCL
```

## Acknowledgements
Huge thanks to the authors of following open-source projects:
- [CLAM](https://github.com/mahmoodlab/CLAM)

## License & Citation 
If you find our work useful in your research, please consider citing our paper at:
```bash
@ARTICLE{zhou2024cohort,
  author={Zhou, Huajun and Zhou, Fengtao and Chen, Hao},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Cohort-Individual Cooperative Learning for Multimodal Cancer Survival Analysis}, 
  year={2024},
}
```
This code is available for non-commercial academic purposes. If you have any question, feel free to email [Huajun ZHOU](csehjzhou@ust.hk).
 
