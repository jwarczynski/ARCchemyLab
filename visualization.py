import numpy as np

from matplotlib import ticker
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.colors as mcolors


# Define a colormap for values from 1 to 9
cmap = mcolors.ListedColormap(['white', 'red', 'orange', 'yellow', 'green', 'blue', 'purple', 'brown', 'pink'])
norm = mcolors.BoundaryNorm(np.arange(1, 11), cmap.N)


def draw_grid(ax: plt.Axes, grid: np.ndarray) -> None:
    """
    Draws a 2D array as a grid on the given Axes object.

    Parameters:
    ax (Axes): The axis where the grid should be drawn.
    grid (np.ndarray): 2D array containing the grid data.
    """
    ax.imshow(grid, cmap=cmap, norm=norm)
    ax.grid(which='major', color='black', linestyle='-', linewidth=2)  # Add major gridlines
    ax.set_xticks(np.arange(-.5, grid.shape[1], 1))
    ax.set_yticks(np.arange(-.5, grid.shape[0], 1))
    ax.set_xticklabels([])  # Hide labels
    ax.set_yticklabels([])
    ax.tick_params(length=0)  # Remove tick marks


def draw_input_output_pair(axs: np.ndarray[plt.Axes], example: dict[str, any], index: int) -> None:
    """
    Draws the input and output grids in the provided Axes.

    Parameters:
    axs (np.ndarray): Array of Axes where the grids will be drawn.
    example (dict[str, any]): A dictionary containing 'input' and 'output' 2D arrays.
    index (int): The index of the current example (used for titles).
    """
    input_grid = np.array(example['input'])
    output_grid = np.array(example['output'])

    # Draw input grid in the first row
    draw_grid(axs[0, index], input_grid)
    axs[0, index].set_title(f'Input {index + 1}')

    # Draw output grid in the second row
    draw_grid(axs[1, index], output_grid)
    axs[1, index].set_title(f'Output {index + 1}')


def plot_challenge_examples(examples: list[dict[str, any]]) -> None:
    """
    Plots a grid of input-output examples from the provided list of examples.

    Parameters:
    examples (list[dict[str, any]]): list of examples where each example contains 'input' and 'output' grids.
    """
    num_examples = len(examples)
    fig, axs = plt.subplots(2, num_examples, figsize=(20, 5))  # 2 rows (input/output), num_examples columns

    for i, example in enumerate(examples):
        draw_input_output_pair(axs, example, i)

    plt.tight_layout()
    plt.show()


def plot_entire_dataset(training_challenges: dict[str, dict[str, list[dict[str, any]]]]) -> None:
    """
    Loops through all the challenges in the dataset and plots them by calling plot_challenge_examples.

    Parameters:
    training_challenges (dict[str, dict[str, list[dict[str, any]]]]): The entire dataset of challenges.
    """
    for identifier, challenge in training_challenges.items():
        examples = challenge['train']
        plot_challenge_examples(examples)
        break

