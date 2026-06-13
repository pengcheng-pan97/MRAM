import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
from collections import Counter
import numpy as np
import seaborn as sns

from utils import denormalize, bounding_box

#.\plots\ram_6_8x8_1_0\
#.\HRAM-MNIST\plotsRAM_6_8x8_1\
#.\HRAM-FMNIST\plotsRAM_8_8x8_1\


#labelclass = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

def parse_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument(
        "--plot_dir",
        type=str,
        required=True,
        help="path to directory containing pickle dumps",
    )
    arg.add_argument("--epoch", type=int, required=True, help="epoch of desired plot")
    args = vars(arg.parse_args())
    return args["plot_dir"], args["epoch"]

def group_into_fixations(coords, dist_thresh=6.0):
    """
    coords: (N, 2) array of [x,y] positions
    dist_thresh: threshold below which consecutive points are 'same fixation'
    returns: list of fixations, each fixation is a list (or array) of coords
    """
    fixations = []
    current_fix = [coords[0]]

    for i in range(1, len(coords)):
        dist = np.linalg.norm(coords[i] - coords[i-1])
        if dist < dist_thresh:
            # same fixation
            current_fix.append(coords[i])
        else:
            # new fixation
            fixations.append(np.array(current_fix))
            current_fix = [coords[i]]

    # don’t forget the last one
    if current_fix:
        fixations.append(np.array(current_fix))

    return fixations



def main(plot_dir, epoch):
    # read in pickle files
    #glimpses_RAM = pickle.load(open(plot_dir + "g_8877.p".format(epoch), "rb"))
    locations_RAM = pickle.load(open(plot_dir + "l_0_RAMs2.p".format(epoch), "rb"))
    #predictions_RAM = pickle.load(open(plot_dir + "p_8877.p".format(epoch), "rb"))
    #glimpses_DRAM = pickle.load(open(plot_dir + "g_8987.p".format(epoch), "rb"))
    locations_DRAM = pickle.load(open(plot_dir + "l_0MRAMs2.p".format(epoch), "rb"))
    #predictions_DRAM = pickle.load(open(plot_dir + "p_8987.p".format(epoch), "rb"))
    #glimpses_DRAMnoCNN = pickle.load(open(plot_dir + "g_8820.p".format(epoch), "rb"))
    locations_DRAMnoCNN = pickle.load(open(plot_dir + "l_0Saccader.p".format(epoch), "rb"))
    #predictions_DRAMnoCNN = pickle.load(open(plot_dir + "p_8820.p".format(epoch), "rb"))
    #glimpses_HRAM = pickle.load(open(plot_dir + "g_9193.p".format(epoch), "rb"))
    locations_HRAM = pickle.load(open(plot_dir + "l_0Bravo-small.p".format(epoch), "rb"))
    #predictions_HRAM = pickle.load(open(plot_dir + "p_9193.p".format(epoch), "rb"))

    #from ipdb import set_trace

    #set_trace()

    #glimpses = np.concatenate(glimpses_RAM)

    # grab useful params
    #size = int(plot_dir.split("_")[2].split("x")[0])
    size  = 8
    img_shape =48
    #num_anims = len(locations_HRAM)
    #num_cols = glimpses.shape[0]
    #img_shape = glimpses.shape[-1]
    #gg = torch.tensor(glimpses)

    # denormalize coordinates
    #coords = [denormalize(img_shape, l) for l in locations]
    coords_RAM = [denormalize(img_shape, l) for l in locations_RAM]
    coords_DRAM = [denormalize(img_shape, l) for l in locations_DRAM]
    coords_DRAMnoCNN = [denormalize(img_shape, l) for l in locations_DRAMnoCNN]
    coords_HRAM = [denormalize(img_shape, l) for l in locations_HRAM]

    all_coords_RAM = np.concatenate(coords_RAM, axis=0)
    all_coords_DRAM = np.concatenate(coords_DRAM, axis=0)
    all_coords_DRAMnoCNN = np.concatenate(coords_DRAMnoCNN, axis=0)
    all_coords_HRAM = np.concatenate(coords_HRAM, axis=0)

    dist_thresh = 6.0

    fixations_RAM = group_into_fixations(all_coords_RAM, dist_thresh=dist_thresh)
    durations_RAM = [len(fix_) for fix_ in fixations_RAM]  # number of frames in each fixation
    fixations_DRAM = group_into_fixations(all_coords_DRAM, dist_thresh=dist_thresh)
    durations_DRAM = [len(fix_) for fix_ in fixations_DRAM]  # number of frames in each fixation
    fixations_DRAMnoCNN = group_into_fixations(all_coords_DRAMnoCNN, dist_thresh=dist_thresh)
    durations_DRAMnoCNN = [len(fix_) for fix_ in fixations_DRAMnoCNN]  # number of frames in each fixation
    fixations_HRAM = group_into_fixations(all_coords_HRAM, dist_thresh=dist_thresh)
    durations_HRAM = [len(fix_) for fix_ in fixations_HRAM]  # number of frames in each fixation

    all_data = [durations_RAM, durations_DRAM, durations_DRAMnoCNN, durations_HRAM]
    labels = ["RAM", "MRAM", "saccader", "Bravo"]
    hatches = [".",".x", "o", "x"]
    all_vals = np.concatenate(all_data)
    max_val = int(all_vals.max())
    bins = range(1, max_val + 2)

    plt.figure()
    """
    plt.hist(
        all_data,  # a list of your four arrays
        bins=bins,
        label=labels,
        histtype='barstacked',
        stacked=False,  # ensures side-by-side bars in each bin
        rwidth=8
    )
    """
    #sns.set_style('whitegrid')
    #sns.kdeplot(np.array(all_data), bw=0.5)
    for i in range(len(all_data)):
        sns.kdeplot(np.array(all_data[i]), label=labels[i], shade=True, bw=0.25)
        #plt.plot(all_data[i], label=labels[i])

    #plt.hist(durations_RAM, bins=range(1, max(durations_RAM) + 2), align='left', rwidth=0.8, label='RAM',histtype='bar', stacked=False)
    #plt.hist(durations_DRAM, bins=range(1, max(durations_DRAM) + 2), align='left', rwidth=0.8, label='DRAM',histtype='bar', stacked=False)
    #plt.hist(durations_DRAMnoCNN, bins=range(1, max(durations_DRAMnoCNN) + 2), align='left', rwidth=0.8, label='DRAM w/o Context',histtype='bar', stacked=False)
    #plt.hist(durations_HRAM, bins=range(1, max(durations_HRAM) + 2), align='left', rwidth=0.8, label='HRAM',histtype='bar', stacked=False)
    plt.legend()
    plt.xlim((-2, 20))  # set the xlim to left, right
    plt.xlabel("Fixation Duration (frames)")
    plt.ylabel("Probability Density")
    plt.title("Fixation Durations")
    plt.show()

    """
    counts_RAM = Counter(durations_RAM)
    unique_durs_RAM = sorted(counts_RAM.keys())
    freqs_RAM = [counts_RAM[d] for d in unique_durs_RAM]

    counts_DRAM = Counter(durations_DRAM)
    unique_durs_DRAM = sorted(counts_DRAM.keys())
    freqs_DRAM = [counts_DRAM[d] for d in unique_durs_DRAM]

    counts_DRAMnoCNN = Counter(durations_DRAMnoCNN)
    unique_durs_DRAMnoCNN = sorted(counts_DRAMnoCNN.keys())
    freqs_DRAMnoCNN = [counts_DRAMnoCNN[d] for d in unique_durs_DRAMnoCNN]

    counts_HRAM = Counter(durations_HRAM)
    unique_durs_HRAM = sorted(counts_HRAM.keys())
    freqs_HRAM = [counts_HRAM[d] for d in unique_durs_HRAM]

    all_data = [durations_RAM, durations_DRAM, durations_DRAMnoCNN, durations_HRAM]
    labels = ["RAM", "DRAM", "DRAM w/o Context", "HRAM"]
    all_vals = np.concatenate(all_data)
    max_val = int(all_vals.max())
    bins = range(1, max_val + 2)
    bar_width = 0.2

    plt.figure()
    plt.bar(unique_durs_RAM, freqs_RAM, label='RAM')
    plt.bar(unique_durs_DRAM, freqs_DRAM, label='DRAM')
    plt.bar(unique_durs_DRAMnoCNN, freqs_DRAMnoCNN, label='DRAM w/o Context')
    plt.bar(unique_durs_HRAM, freqs_HRAM, label='HRAM')
    plt.legend()
    plt.xlabel("Fixation duration (frames)")
    plt.ylabel("Frequency")
    plt.title("Bar chart of fixation durations")
    plt.show()
    """
    jump_distances_RAM = []
    for i in range(len(fixations_RAM) - 1):
        # last coordinate of fixation i
        last_of_i = fixations_RAM[i][-1]
        # first coordinate of fixation i+1
        first_of_next = fixations_RAM[i + 1][0]
        jump_dist = np.linalg.norm(first_of_next - last_of_i)
        jump_distances_RAM.append(jump_dist)

    jump_distances_DRAM = []
    for i in range(len(fixations_DRAM) - 1):
        # last coordinate of fixation i
        last_of_i = fixations_DRAM[i][-1]
        # first coordinate of fixation i+1
        first_of_next = fixations_DRAM[i + 1][0]
        jump_dist = np.linalg.norm(first_of_next - last_of_i)
        jump_distances_DRAM.append(jump_dist)

    jump_distances_DRAMnoCNN = []
    for i in range(len(fixations_DRAMnoCNN) - 1):
        # last coordinate of fixation i
        last_of_i = fixations_DRAMnoCNN[i][-1]
        # first coordinate of fixation i+1
        first_of_next = fixations_DRAMnoCNN[i + 1][0]
        jump_dist = np.linalg.norm(first_of_next - last_of_i)
        jump_distances_DRAMnoCNN.append(jump_dist)

    jump_distances_HRAM = []
    for i in range(len(fixations_HRAM) - 1):
        # last coordinate of fixation i
        last_of_i = fixations_HRAM[i][-1]
        # first coordinate of fixation i+1
        first_of_next = fixations_HRAM[i + 1][0]
        jump_dist = np.linalg.norm(first_of_next - last_of_i)
        jump_distances_HRAM.append(jump_dist)

    all_data = [jump_distances_RAM, jump_distances_DRAM, jump_distances_DRAMnoCNN, jump_distances_HRAM]
    labels = ["RAM", "MRAM", "saccader", "Bravo"]
    hatches = [".",  "o", "x"]
    all_vals = np.concatenate(all_data)
    max_val = int(all_vals.max())
    bins = range(1, max_val + 2)
    """
    plt.figure()
    plt.hist(
        all_data,  # a list of your four arrays
        bins=bins,
        label=labels,
        histtype='barstacked',
        stacked=False,  # ensures side-by-side bars in each bin
        rwidth=8
    )
    """
    #sns.set_style('whitegrid')
    #sns.kdeplot(np.array(all_data), bw=0.5)
    for i in range(len(all_data)):
        sns.kdeplot(np.array(all_data[i]), label=labels[i], shade=True,bw=0.25)
        #plt.plot(all_data[i], label=labels[i])

    #plt.figure()
    #plt.hist(jump_distances_RAM, bins=30, label='RAM',histtype='bar', stacked=False)  # for example
    #plt.hist(jump_distances_DRAM, bins=30, label='DRAM',histtype='bar', stacked=False)  # for example
    #plt.hist(jump_distances_DRAMnoCNN, bins=30, label='DRAM w/o Context',histtype='bar', stacked=False)  # for example
    #plt.hist(jump_distances_HRAM, bins=30, label='HRAM',histtype='bar', stacked=False)  # for example
    plt.legend()
    plt.xlabel("Jump Distance")
    plt.ylabel("Probability Density")
    plt.title("Jump Distances between Fixations")
    plt.show()
    all_data

if __name__ == "__main__":
    args = parse_arguments()
    main(*args)
