# 农作物病害检测

微调 Inception-ResNet-V2, 解决 AI Challenger 2018 农作物病害检测问题。


## 依赖

- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [Keras](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## 数据集

我们使用AI Challenger 2018中的农作物病害检测数据集，其中包含61类47,393张图像。 数据集随机分为训练（70%）、验证（10%）、测试A（10%）与测试B（10%）四个子数据集。

### 数据预览
 ![image](https://github.com/foamliu/Crop-Disease-Detection/raw/master/images/dataset.png)

你可以到 [这里](https://challenger.ai/dataset/pdd2018) 下载。

### 性能
用 4,739 张测试图片计算平均准确率(mAP)，结果如下：

| |Test A|Test B|
|---|---|---|
|图片数|4739|4739|
|Top3准确度|n/a|n/a|

## 用法

### 数据预处理
提取60,999个训练图像，并将它们分开（53,879个用于训练，7,120个用于验证）：
```bash
$ python pre_process.py
```

### 训练
```bash
$ python train.py
```

如果想在培训期间进行可视化，请在终端中运行：
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

### Demo
下载 [pre-trained model](https://github.com/foamliu/Crop-Disease-Detection/releases/download/v1.0/model.11-0.6262.hdf5) 放在 models 目录然后执行:

```bash
$ python demo.py
```

1 | 2 | 3 | 4 |
|---|---|---|---|
|![image](https://github.com/foamliu/Crop-Disease-Detection/raw/master/images/0_out.png)  | ![image](https://github.com/foamliu/Crop-Disease-Detection/raw/master/images/1_out.png) | ![image](https://github.com/foamliu/Crop-Disease-Detection/raw/master/images/2_out.png)| ![image](https://github.com/foamliu/Crop-Disease-Detection/raw/master/images/3_out.png) |
|番茄晚疫病菌一般, prob: 0.5776|苹果灰斑病, prob: 0.6683|马铃薯早疫病一般, prob: 0.2478|番茄晚疫病菌一般, prob: 0.4889|
|![image](https://github.com/foamliu/Crop-Disease-Detection/raw/master/images/4_out.png)  | ![image](https://github.com/foamliu/Crop-Disease-Detection/raw/master/images/5_out.png) | ![image](https://github.com/foamliu/Crop-Disease-Detection/raw/master/images/6_out.png)| ![image](https://github.com/foamliu/Crop-Disease-Detection/raw/master/images/7_out.png) |
|苹果黑星病严重, prob: 0.4044|番茄斑枯病一般, prob: 0.5698|辣椒  健康, prob: 0.3866|番茄晚疫病菌一般, prob: 0.04975|
|![image](https://github.com/foamliu/Crop-Disease-Detection/raw/master/images/8_out.png)  | ![image](https://github.com/foamliu/Crop-Disease-Detection/raw/master/images/9_out.png) |![image](https://github.com/foamliu/Crop-Disease-Detection/raw/master/images/10_out.png) | ![image](https://github.com/foamliu/Crop-Disease-Detection/raw/master/images/11_out.png)|
|番茄黄化曲叶病毒病一般, prob: 0.1804|番茄晚疫病菌一般, prob: 0.7724|番茄 花叶病毒病, prob: 0.9591|番茄红蜘蛛损伤严重, prob: 0.391|
|![image](https://github.com/foamliu/Crop-Disease-Detection/raw/master/images/12_out.png)  | ![image](https://github.com/foamliu/Crop-Disease-Detection/raw/master/images/13_out.png) |![image](https://github.com/foamliu/Crop-Disease-Detection/raw/master/images/14_out.png)| ![image](https://github.com/foamliu/Crop-Disease-Detection/raw/master/images/15_out.png)|
|草莓叶枯病严重, prob: 0.3965|番茄疮痂病严重, prob: 0.4543|柑桔健康, prob: 0.1277|番茄晚疫病菌严重, prob: 0.2699|
|![image](https://github.com/foamliu/Crop-Disease-Detection/raw/master/images/16_out.png) | ![image](https://github.com/foamliu/Crop-Disease-Detection/raw/master/images/17_out.png) | ![image](https://github.com/foamliu/Crop-Disease-Detection/raw/master/images/18_out.png) | ![image](https://github.com/foamliu/Crop-Disease-Detection/raw/master/images/19_out.png) |
|马铃薯晚疫病一般, prob: 0.7151|番茄斑点病严重, prob: 0.4571|番茄黄化曲叶病毒病严重, prob: 0.2466|苹果灰斑病, prob: 0.3973|


### 性能评估
```bash
$ python evaluate.py
```
