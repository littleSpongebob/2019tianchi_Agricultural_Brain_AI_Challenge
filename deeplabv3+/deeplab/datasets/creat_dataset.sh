#!/bin/bash
# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:/home/sjw/Desktop/县域农业大脑AI挑战赛/git/models-master/research:/home/sjw/Desktop/县域农业大脑AI挑战赛/git/models-master/research/slim
python3 build_voc2012_data.py \
  --image_folder=/home/sjw/Desktop/jingwei_round_project/data/unit_512/JPEGImages \
  --semantic_segmentation_folder=/home/sjw/Desktop/jingwei_round_project/data/unit_512/SegmentationClass \
  --list_folder=/home/sjw/Desktop/jingwei_round_project/data/unit_512/ImageSets/Segmentation \
  --output_dir=/home/sjw/Desktop/jingwei_round_project/data/tfrecord_unit_512

