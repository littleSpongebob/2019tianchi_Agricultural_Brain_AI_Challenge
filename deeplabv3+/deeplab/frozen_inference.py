import tensorflow as tf
import numpy as np
import cv2
import time
import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_arguments():
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--data_dir", type=str, required=True, help="the root dir of the data")
    parser.add_argument("--output_dir", type=str, required=True, help="the root dir of the output data")
    return parser.parse_args()

def main():
    args = get_arguments()
    rootdir = args.data_dir
    fileslist = os.listdir(rootdir)


    with open("/home/sjw/Desktop/县域农业大脑AI挑战赛/log/deeplab/frozen_inferencd/frozen_inference_graph.pb", "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    sess = tf.Session()
    input_x = sess.graph.get_tensor_by_name('ImageTensor:0')
    output = sess.graph.get_tensor_by_name('SemanticPredictions:0')
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    for file in fileslist:
        read_path = os.path.join(rootdir, file)
        write_path = os.path.join(args.output_dir, file)
        img = cv2.imread(read_path)
        b, g, r = cv2.split(img)
        img_rgb = cv2.merge([r, g, b])
        img = np.expand_dims(img_rgb, axis=0).astype(np.uint8)
        result = sess.run(output, {input_x: img})
        result = result[0]
        # result = np.expand_dims(np.squeeze(result), axis=2)
        # result = np.where(result == 15, 255, 0)
        # cv2.imwrite(write_path, result)
        np.save(write_path,result)

if __name__ == '__main__':
    st = time.time()
    print('start')
    main()
    end = time.time()
    print('done')
    print(end-st)
