import torch

from torchvision.utils import save_image, make_grid
from torch.autograd import Variable

def sample_images(batches_done, dataset_names, save_path, test_data, G_a, G_b, img_a, img_b, Tens):
    """Saves a generated sample from the test set"""
    imgs = next(iter(test_data))
    G_a.eval()
    G_b.eval()
    real_A = Variable(imgs[img_a].type(Tens))
    fake_B = G_a(real_A)
    real_B = Variable(imgs[img_b].type(Tens))
    fake_A = G_b(real_B)
    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, "generated_images/" + save_path + "/%s/%s.png" % (dataset_names, batches_done),
               normalize=False)