import argparse
import itertools

from wrapper import*

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--dataset_name", type=str, default="horse2zebra", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=32, help="size of image height")
    parser.add_argument("--img_width", type=int, default=32, help="size of image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=500, help="interval between saving generator outputs")
    parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
    parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
    parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")

    opt = parser.parse_args()
    print(opt)

    param = opt.__dict__

    train_list = {'num_epochs': [200,300],
                  'learning_rate': [0.0002, 0.0001],
                  'lr_decay_epochs': [100],
                  'residual_blocks': [9, 5, 3]}

    product_set = itertools.product(train_list['num_epochs'],
                                    train_list['learning_rate'],
                                    train_list['lr_decay_epochs'],
                                    train_list['residual_blocks'])

    for num_epochs, learning_rate, le_decay_epochs, residual_blocks in product_set:
        param['num_epochs'] = num_epochs
        param['learning_rate'] = learning_rate
        param['lr_decay_epochs'] = le_decay_epochs
        param['residual_blocks'] = residual_blocks
        wrapper(param)