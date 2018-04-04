import argparse


def parse_opts():
    parser = argparse.ArgumentParser(description='simple 3d convolution for casme2')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_data', type=int, default=1048)
    parser.add_argument('--epoch', type=int, default=25)
    parser.add_argument('--videos', type=str,
                        default='C:\\Users\\PVer\\Desktop\\output\\',
                        help='directory where videos are stored')
    parser.add_argument('--mapfile', type=str,
                        default='D:\\CASME2_model\\',
                        help='mapfile where you store')
    parser.add_argument('--nclasses', type=int, default=5)
    parser.add_argument('--output', type=str, default="D:\\CASME2_model\\train.tfrecords")
    parser.add_argument('--checkpoint_path', type=str, default="D:\\CASME2_model\\checkpoints\\")
    parser.add_argument('--save_path', type=str, default="D:\\CASME2_model\\model\\model.ckpt")
    parser.add_argument('--color', type=bool, default=True)
    parser.add_argument('--skip', type=bool, default=True)
    parser.add_argument('--img_size', type=int, default=112)
    parser.add_argument('--depth', type=int, default=8)
    args = parser.parse_args()

    return args
