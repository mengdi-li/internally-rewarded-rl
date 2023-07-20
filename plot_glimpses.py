import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.lines as lines
import os
import wandb

from utils import denormalize, bounding_box


def parse_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument(
        "--plot_dir",
        type=str,
        required=True,
        help="path to directory containing pickle dumps",
    )
    arg.add_argument("--epoch", type=int, required=True, help="epoch of desired plot")
    arg.add_argument(
        "--train_or_eval",
        type=str,
        default="train",
        help="use logs of training or evaluation",
    )
    arg.add_argument("--plot_type", type=str, default="all", help="[figure, video]_[all, one]. if we plot all samples saved in an epoch or only one of them. ")
    arg.add_argument("--i_sample_to_plot", type=int, default=0, help="which sample to plot when plot_type is figure_one or video_one")
    args = vars(arg.parse_args())
    return args

def log_wandb_video(glimpses, locations, labels, predictions, patch_size, train_or_eval):
    assert train_or_eval == "train" or train_or_eval == "eval"
    size = patch_size
    glimpses = np.concatenate(glimpses)
    labels = np.concatenate(labels)
    predictions = np.concatenate(predictions)
    num_anims = len(locations)
    num_glimpses = glimpses.shape[0]
    img_shape = glimpses.shape[1]

    coords = [denormalize(img_shape, l) for l in locations]

    assert num_glimpses == 9
    nrows = 3
    ncols = num_glimpses // nrows
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)

    # plot base image
    for j, ax in enumerate(axs.flat):
        ax.imshow(glimpses[j], cmap="Greys_r")
        xlabel = "label: {} - pred: {}".format(labels[j], predictions[j])
        ax.set_xlabel(xlabel)
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    coords_pre = np.zeros_like(coords[0])

    def get_one_frame(i):
        color_start = "y"
        color_move = "g"
        color_stop = "r"
        linewidth = 1.0
        if i == 0:
            box_color = color_start
        elif i == num_anims - 1:
            box_color = color_stop
        else:
            box_color = color_move
        linestyle_move = "-"
        linestyle_stop = "-"
        co = coords[i]
        for j, ax in enumerate(axs.flat):
            for p in ax.patches:
                p.remove()
            c = co[j]
            if c[0] > img_shape or c[1] > img_shape:
                rect = bounding_box(coords_pre[j][0], coords_pre[j][1], size, linewidth, color_stop, linestyle_stop)
            else:
                rect = bounding_box(c[0], c[1], size, linewidth, box_color, linestyle_move)
                coords_pre[j] = c
            ax.add_patch(rect)

        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data
    
    frames = []
    for i in range(num_anims):
        frame = get_one_frame(i)
        frames.append(frame)
    
    frames = np.transpose(np.array(frames), (0, 3, 1, 2))
    wandb.log({
            '{}/video'.format(train_or_eval):
            wandb.Video(frames, fps=2, format="gif")
        }, commit=False)

def plot_video_all(plot_dir, epoch, train_or_eval):
    # read in pickle files
    assert train_or_eval in ["train", "eval", "test"] 
    glimpses = pickle.load(open(os.path.join(plot_dir, "{}_g_{}.p".format(train_or_eval, epoch)), "rb"))
    locations = pickle.load(open(os.path.join(plot_dir, "{}_l_{}.p".format(train_or_eval, epoch)), "rb"))

    # from ipdb import set_trace
    # set_trace()

    glimpses = np.concatenate(glimpses)

    # grab useful params
    size = int(plot_dir.split("_")[-2][0])
    num_anims = len(locations)
    num_glimpses = glimpses.shape[0]
    img_shape = glimpses.shape[1]

    # denormalize coordinates
    coords = [denormalize(img_shape, l) for l in locations]

    nrows = 3
    ncols = num_glimpses // nrows
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    # fig.set_dpi(100)

    # plot base image
    for j, ax in enumerate(axs.flat):
        ax.imshow(glimpses[j], cmap="Greys_r")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    coords_pre = np.zeros_like(coords[0])
    def updateData(i):
        color_start = "y"
        color_move = "g"
        color_stop = "r"
        linewidth = 1.0
        if i == 0:
            box_color = color_start
        elif i == num_anims - 1:
            box_color = color_stop
        else:
            box_color = color_move
        linestyle_move = "-"
        linestyle_stop = "-"
        co = coords[i]
        for j, ax in enumerate(axs.flat):
            for p in ax.patches:
                p.remove()
            c = co[j]
            if c[0] > img_shape or c[1] > img_shape:
                rect = bounding_box(coords_pre[j][0], coords_pre[j][1], size, linewidth, color_stop, linestyle_stop)
            else:
                rect = bounding_box(c[0], c[1], size, linewidth, box_color, linestyle_move)
                coords_pre[j] = c
            ax.add_patch(rect)

    # animate
    anim = animation.FuncAnimation(
        fig, updateData, frames=num_anims, interval=500, repeat=True
    )

    # save as mp4
    name = os.path.join(plot_dir, "epoch_{}.mp4".format(epoch))
    anim.save(name, extra_args=["-vcodec", "h264", "-pix_fmt", "yuv420p"])


def plot_video_one(plot_dir, epoch, i_sample_to_plot, train_or_eval):
    # read in pickle files
    assert train_or_eval in ["train", "eval", "test"] 
    glimpses = pickle.load(open(os.path.join(plot_dir, "{}_g_{}.p".format(train_or_eval, epoch)), "rb"))
    locations = pickle.load(open(os.path.join(plot_dir, "{}_l_{}.p".format(train_or_eval, epoch)), "rb"))

    # from ipdb import set_trace
    # set_trace()

    glimpses = np.concatenate(glimpses)

    # grab useful params
    size = int(plot_dir.split("_")[-2][0])
    num_anims = len(locations)
    num_glimpses = glimpses.shape[0]
    img_shape = glimpses.shape[1]

    # denormalize coordinates
    coords = [denormalize(img_shape, l) for l in locations]

    nrows = 1
    ncols = 1 # num_glimpses // nrows
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    # fig.set_dpi(100)

    # plot base image
    ax.imshow(glimpses[i_sample_to_plot], cmap="Greys_r")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    coords_pre = np.zeros_like(coords[0])
    def updateData(i):
        color_start = "y"
        color_move = "g"
        color_stop = "r"
        linewidth = 1.0
        if i == 0:
            box_color = color_start
        elif i == num_anims - 1:
            box_color = color_stop
        else:
            box_color = color_move
        linestyle_move = "-"
        linestyle_stop = "-"
        co = coords[i]
        for p in ax.patches:
            p.remove()
        c = co[i_sample_to_plot]
        if c[0] > img_shape or c[1] > img_shape:
            rect = bounding_box(coords_pre[i_sample_to_plot][0], coords_pre[i_sample_to_plot][1], size, linewidth, color_stop, linestyle_stop)
        else:
            rect = bounding_box(c[0], c[1], size, linewidth, box_color, linestyle_move)
            coords_pre[i_sample_to_plot] = c
        ax.add_patch(rect)

    # animate
    anim = animation.FuncAnimation(
        fig, updateData, frames=num_anims, interval=500, repeat=True
    )

    # save as mp4
    name = os.path.join(plot_dir, "epoch_{}_{}.mp4".format(epoch, i_sample_to_plot))
    anim.save(name, extra_args=["-vcodec", "h264", "-pix_fmt", "yuv420p"])

def plot_figure_one(plot_dir, epoch, i_sample_to_plot, train_or_eval):
    # read in pickle files
    assert train_or_eval in ["train", "eval", "test"] 
    glimpses = pickle.load(open(os.path.join(plot_dir, "{}_g_{}.p".format(train_or_eval, epoch)), "rb"))
    locations = pickle.load(open(os.path.join(plot_dir, "{}_l_{}.p".format(train_or_eval, epoch)), "rb"))

    glimpses = np.concatenate(glimpses)

    # grab useful params
    size = int(plot_dir.split("_")[-2][0])
    num_anims = len(locations)
    num_glimpses = glimpses.shape[0]
    img_shape = glimpses.shape[1]

    # denormalize coordinates
    coords = [denormalize(img_shape, l) for l in locations]

    # plot base image
    fig = plt.imshow(glimpses[i_sample_to_plot], cmap="Greys_r")
    ax = fig.axes
    plt.axis('off')

    coords_pre = np.zeros_like(coords[0])
    def updateData(i):
        color_start = "y"
        color_move = "g"
        color_stop = "r"
        linewidth = 4.0
        if i == 0:
            box_color = color_start
        elif i == num_anims - 1:
            box_color = color_stop
        else:
            box_color = color_move
            linewidth = 0
        linestyle_move = "-"
        linestyle_stop = "-"
        co = coords[i]
        c = co[i_sample_to_plot]
        if c[0] > img_shape or c[1] > img_shape:
            rect = bounding_box(coords_pre[i_sample_to_plot][0], coords_pre[i_sample_to_plot][1], size, linewidth, color_stop, linestyle_stop)
        else:
            rect = bounding_box(c[0], c[1], size, linewidth, box_color, linestyle_move)
            if i > 0: 
                # fig.add_artist(lines.Line2D(coords_pre[i_sample_to_plot], c))
                pathline_width = 3
                plt.plot((coords_pre[i_sample_to_plot][0], c[0]), (coords_pre[i_sample_to_plot][1], c[1]), 'lime', lw=pathline_width)
            coords_pre[i_sample_to_plot] = c
        ax.add_patch(rect)
        # add line
        

    for i in range(num_anims):
        updateData(i)

    # save as jpg
    name = os.path.join(plot_dir, "epoch_{}_{}.jpg".format(epoch, i_sample_to_plot))
    plt.savefig(name, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == "__main__":
    args = parse_arguments()
    if args["plot_type"] == "video_one": 
        plot_video_one(args["plot_dir"], args["epoch"], args["i_sample_to_plot"], args["train_or_eval"])
    elif args["plot_type"] == "video_all": 
        plot_video_all(args["plot_dir"], args["epoch"], args["train_or_eval"])
    elif args["plot_type"] == "figure_one": 
        plot_figure_one(args["plot_dir"], args["epoch"], args["i_sample_to_plot"], args["train_or_eval"])
    elif args["plot_type"] == "figure_all": 
        for i in range(9):
            plot_figure_one(args["plot_dir"], args["epoch"], i, args["train_or_eval"])
    else:
        raise