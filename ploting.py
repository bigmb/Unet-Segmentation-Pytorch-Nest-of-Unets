import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from visdom import Visdom


def show_images(images, labels):
    """Show image with label
    Args:
        images = input images
        labels = input labels
    Output:
        plt  = concatenated image and label """

    plt.imshow(images.permute(1, 2, 0))
    plt.imshow(labels, alpha=0.7, cmap='gray')
    plt.figure()


def show_training_dataset(training_dataset):
    """Showing the images in training set for dict images and labels
    Args:
        training_dataset = dictionary of images and labels
    Output:
        figure = 3 images shown"""

    if training_dataset:
        print(len(training_dataset))

    for i in range(len(training_dataset)):
        sample = training_dataset[i]

        print(i, sample['images'].shape, sample['labels'].shape)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        show_images(sample['images'],sample['labels'])

        if i == 3:
            plt.show()
            break

class VisdomLinePlotter(object):

    """Plots to Visdom"""

    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')


def input_images(x, y, i, n_iter, k=1):
    """

    :param x: takes input image
    :param y: take input label
    :param i: the epoch number
    :param n_iter:
    :param k: for keeping it in loop
    :return: Returns a image and label
    """
    if k == 1:
        x1 = x
        y1 = y

        x2 = x1.to('cpu')
        y2 = y1.to('cpu')
        x2 = x2.detach().numpy()
        y2 = y2.detach().numpy()

        x3 = x2[1, 1, :, :]
        y3 = y2[1, 0, :, :]

        fig = plt.figure()

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(x3)
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1 = fig.add_subplot(1, 2, 2)
        ax1.imshow(y3)
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        plt.savefig(
            './model/pred/L_' + str(n_iter-1) + '_epoch_'
            + str(i))


def plot_kernels(tensor, n_iter, num_cols=5, cmap="gray"):
    """Plotting the kernals and layers
    Args:
        Tensor :Input layer,
        n_iter : number of interation,
        num_cols : number of columbs required for figure
    Output:
        Gives the figure of the size decided with output layers activation map

    Default : Last layer will be taken into consideration
        """
    if not len(tensor.shape) == 4:
        raise Exception("assumes a 4D tensor")

    fig = plt.figure()
    i = 0
    t = tensor.data.numpy()
    b = 0
    a = 1

    for t1 in t:
        for t2 in t1:
            i += 1

            ax1 = fig.add_subplot(5, num_cols, i)
            ax1.imshow(t2, cmap=cmap)
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])

            if i == 1:
                a = 1
            if a == 10:
                break
            a += 1
        if i % a == 0:
            a = 0
        b += 1
        if b == 20:
            break

    plt.savefig(
        './model/pred/Kernal_' + str(n_iter - 1) + '_epoch_'
        + str(i))


class LayerActivations():
    """Getting the hooks on each layer"""

    features = None

    def __init__(self, layer):
        self.hook = layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()


#to get gradient flow
#From Pytorch-forums
def plot_grad_flow(named_parameters,n_iter):

    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    #plt.savefig('./model/pred/Grad_Flow_' + str(n_iter - 1))
