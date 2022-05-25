import math
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import itertools

# -------------------------------------------------------------
# show_images
# -------------------------------------------------------------
#


def plot_images(x, y=None, indices='all', columns=12, x_size=1, y_size=1,
                colorbar=False, y_pred=None, cm='binary', norm=None, y_padding=0.35, spines_alpha=1,
                fontsize=20, interpolation='lanczos', save_as='auto'):
    """
    Show some images in a grid, with legends
    args:
        x             : images - Shapes must be (-1,lx,ly) (-1,lx,ly,1) or (-1,lx,ly,3)
        y             : real classes or labels or None (None)
        indices       : indices of images to show or 'all' for all ('all')
        columns       : number of columns (12)
        x_size,y_size : figure size (1), (1)
        colorbar      : show colorbar (False)
        y_pred        : predicted classes (None)
        cm            : Matplotlib color map (binary)
        norm          : Matplotlib imshow normalization (None)
        y_padding     : Padding / rows (0.35)
        spines_alpha  : Spines alpha (1.)
        font_size     : Font size in px (20)
        save_as       : Filename to use if save figs is enable ('auto')
    returns: 
        nothing
    """
    if indices == 'all':
        indices = range(len(x))
    if norm and len(norm) == 2:
        norm = matplotlib.colors.Normalize(vmin=norm[0], vmax=norm[1])
    draw_labels = (y is not None)
    draw_pred = (y_pred is not None)
    rows = math.ceil(len(indices)/columns)
    fig = plt.figure(figsize=(columns*x_size, rows*(y_size+y_padding)))
    n = 1
    for i in indices:
        axs = fig.add_subplot(rows, columns, n)
        n += 1
        # ---- Shape is (lx,ly)
        if len(x[i].shape) == 2:
            xx = x[i]
        # ---- Shape is (lx,ly,n)
        if len(x[i].shape) == 3:
            (lx, ly, lz) = x[i].shape
            if lz == 1:
                xx = x[i].reshape(lx, ly)
            else:
                xx = x[i]
        img = axs.imshow(xx,   cmap=cm, norm=norm, interpolation=interpolation)
#         img=axs.imshow(xx,   cmap = cm, interpolation=interpolation)
        axs.spines['right'].set_visible(True)
        axs.spines['left'].set_visible(True)
        axs.spines['top'].set_visible(True)
        axs.spines['bottom'].set_visible(True)
        axs.spines['right'].set_alpha(spines_alpha)
        axs.spines['left'].set_alpha(spines_alpha)
        axs.spines['top'].set_alpha(spines_alpha)
        axs.spines['bottom'].set_alpha(spines_alpha)
        axs.set_yticks([])
        axs.set_xticks([])
        if draw_labels and not draw_pred:
            axs.set_xlabel(y[i], fontsize=fontsize)
        if draw_labels and draw_pred:
            if y[i] != y_pred[i]:
                axs.set_xlabel(f'{y_pred[i]} ({y[i]})', fontsize=fontsize)
                axs.xaxis.label.set_color('red')
            else:
                axs.set_xlabel(y[i], fontsize=fontsize)
        if colorbar:
            fig.colorbar(img, orientation="vertical", shrink=0.65)
    # save_fig(save_as)
    plt.show()


def plot_image(x, cm='binary', figsize=(4, 4), interpolation='lanczos', save_as='auto'):
    """
    Draw a single image.
    Image shape can be (lx,ly), (lx,ly,1) or (lx,ly,n)
    args:
        x       : image as np array
        cm      : color map ('binary')
        figsize : fig size (4,4)
    """
    # ---- Shape is (lx,ly)
    if len(x.shape) == 2:
        xx = x
    # ---- Shape is (lx,ly,n)
    if len(x.shape) == 3:
        (lx, ly, lz) = x.shape
        if lz == 1:
            xx = x.reshape(lx, ly)
        else:
            xx = x
    # ---- Draw it
    plt.figure(figsize=figsize)
    plt.imshow(xx,   cmap=cm, interpolation=interpolation)
    # save_fig(save_as)
    plt.show()

# -------------------------------------------------------------
# show_history
# -------------------------------------------------------------
#


def plot_history(history, figsize=(8, 6),
                 plot={"Accuracy": ['accuracy', 'val_accuracy'],
                       'Loss': ['loss', 'val_loss']},
                 save_as='auto'):
    """
    Show history
    args:
        history: history
        figsize: fig size
        plot: list of data to plot : {<title>:[<metrics>,...], ...}
    """
    fig_id = 0
    for title, curves in plot.items():
        plt.figure(figsize=figsize)
        plt.title(title)
        plt.ylabel(title)
        plt.xlabel('Epoch')
        for c in curves:
            plt.plot(history.history[c])
        plt.legend(curves, loc='upper left')
        if save_as == 'auto':
            figname = 'auto'
        else:
            figname = f'{save_as}_{fig_id}'
            fig_id += 1
        # save_fig(figname)
        plt.show()


def plot_confusion_matrix(y_true, y_pred,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          figsize=(10, 8),
                          digit_format='{:0.2f}',
                          save_as='auto'):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    cm = confusion_matrix(y_true, y_pred, normalize=None, labels=target_names)

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, digit_format.format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
        accuracy, misclass))
    # save_fig(save_as)
    plt.show()
