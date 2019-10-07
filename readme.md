# 项目名称

无人机图像语义分割--天池2019年县域农业大脑AI挑战赛冠军(冲鸭！大黄)解决方案(部分代码)
详细思路介绍见[博客链接](https://blog.csdn.net/qq_21407487/article/details/102326264)。
[最终排行榜](https://tianchi.aliyun.com/competition/entrance/231717/rankingList)
# 依赖

* albumentations==0.3.0
* Keras==2.2.4
* Keras-Applications==1.0.8
* Keras-Preprocessing==1.0.5
* keras-rectified-adam==0.9.0
* numpy==1.15.4
* opencv-python==3.4.1.15
* opencv-python-headless==4.1.0.25
* pandas==0.24.2
* pandocfilters==1.4.2
* Pillow==6.1.0
* pretrainedmodels==0.7.4
* scikit-image==0.14.1
* scipy==1.1.0
* tqdm==4.28.1

或者直接运行
> pip install -r requirement.txt

# 使用
## 训练
### 训练数据裁切
> python3 -W ignore rawdata_crop.py \
--image-num 3 \
--unit 1024 \
--output-dir [输出路径] \
--image_path [训练数据路径] \
--label_path [训练标签路径]
### 训练deeplabv3+
可以参考如下博客：
1. [Deeplab V3+训练自己数据集全过程](https://blog.csdn.net/jairana/article/details/83900226)
2. [deeplab v3+训练自己的数据](https://blog.csdn.net/ncloveqy/article/details/82285106)
3. [DeepLabV3+训练自己的数据](https://blog.csdn.net/PNAN222/article/details/89450711)

[预训练模型权重](http://download.tensorflow.org/models/deeplab_cityscapes_xception71_trainvalfine_2018_09_08.tar.gz)

具体训练参数如下：
```shell
cd $basedir/xception_deeplabv3+_train/deeplabv3+
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
model_name='xception_1'
epoch=5
warn_start_step=1
batch_size=2
base_learning_rate=0.0001
data_set='remote_sensing_data_1024_a81'
data_set_dir="$basedir/xception_deeplabv3+_train/data/tfrecord_1024_a81"

model_variant='xception_65'
num_sample=7175
step=$(($epoch * $num_sample / $batch_size))
warn_start_step=$(($warn_start_step * $num_sample / $batch_size))
echo -e "\033[31m data_set=$data_set \033[0m"
echo -e "\033[31m model_name=$model_name \033[0m"
echo -e "\033[31m step=$step \033[0m"
echo -e "\033[31m warn_start_step=$warn_start_step \033[0m"
echo -e "\033[31m base_learning_rate=$base_learning_rate \033[0m"
export CUDA_VISIBLE_DEVICES=0,1
# train xnception65
python3 -W ignore ./deeplab/train.py \
  --logtostderr \
  --save_summaries_images=True \
  --dataset=$data_set \
  --train_split="trainval"\
  --model_variant=$model_variant \
  --decoder_output_stride=4 \
  --train_crop_size=1025,1025 \
  --num_clones=2 \
  --train_batch_size=$batch_size \
  --training_number_of_steps=$step \
  --initialize_last_layer=False \
  --last_layers_contain_logits_only=True \
  --fine_tune_batch_norm=False \
  --tf_initial_checkpoint="$basedir/xception_deeplabv3+_train/deeplabv3+/pretrain_model/deeplabv3_pascal_trainval_2018_01_04/deeplabv3_pascal_trainval/model.ckpt" \
  --train_logdir="$basedir/xception_deeplabv3+_train/deeplabv3+/log/$model_name" \
  --dataset_dir=$data_set_dir \
  --min_scale_factor=1.0 \
  --max_scale_factor=1.0 \
  --scale_factor_step_size=0.25 \
  --base_learning_rate=$base_learning_rate \
  --learning_policy='poly' \
  --slow_start_step=$warn_start_step \
  --slow_start_learning_rate=0.000005 \
  --last_layer_gradient_multiplier=10.0 \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --save_interval_secs=$(($num_sample / $batch_size)) \
  --weight_decay=0.00004 \
  --save_summaries_secs=30 \
  --learning_power=2.0 \
  --log_steps=500

# export model
cd $basedir/xception_deeplabv3+_train/deeplabv3+
python3 ./deeplab/export_model.py \
  --logtostderr \
  --checkpoint_path="$basedir/xception_deeplabv3+_train/deeplabv3+/log/$model_name/model.ckpt-$step" \
  --export_path="/competition/xception_deeplabv3+/model/xception_1/frozen_inference_graph.pb" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --num_classes=5 \
  --crop_size=2049 \
  --crop_size=2049 \
  --add_flipped_images=True \
  --inference_scales=1.0

```

## 预测

### 裁切测试数据
> python3 creat_test_unit.py \
--image_num=5 \
--data_path=[测试数据路径] \
--output_path=[结果输出路径] \
--unit=2048 \
--pad=512 \
--step=1024

### 创建可视化和mask文件
> python3 creat_vis_and_mask.py \
--image_num=5 \
--data_path=[测试数据路径] \
--output_path=[结果输出路径] \

### 预测

>python3 test_model.py \
--data_dir=[测试数据路径] \
--output_dir=[输出路径] \
--model_path=[pb模型路径] \
--vis_and_mask_dir=[可视化和mask所在路径]
--image_num=5 \
--area_threshold=0 \
--padding_size=512 \
--image_H=20115 \
--image_W=43073 \

### 后处理

> python3 post_process.py \
--model_name=[模型的名称] \
--step=[训练的步数] \
--area_threshold=[后处理的去除连通域的大小] \

model_name和step组成输入数据的文件夹路径
### 投票

> python3 vote.py \
--one_dir=[第一个模型结果路径] \
--two_dir=[第二个模型结果路径] \
--three_dir=[第三个模型结果路径] \
--output_dir=[融合结果输出路径] \
--vis_dir=[可视化和mask所在路径] \
--area_threshold=[后处理的去除连通域的大小]

# 致谢
该项目网络基于tensorflow官方的[deeplabv3+](https://github.com/tensorflow/models/tree/master/research/deeplab)项目
感谢另外两位队友的大力支持
* [大黄有故事](https://tianchi.aliyun.com/home/science/scienceDetail?userId=1095279178743)
* [now more](https://tianchi.aliyun.com/home/science/scienceDetail?userId=1095279428856)

