import numpy as np


def action_space_to_dict(action_space):
    """
    Convert a Box action space to a dictionary representation.

    Args:
        action_space: gym.spaces.Box
            The Box action space to convert.

    Returns:
        dict: A dictionary containing information about the action space.
            - "high": Upper bound values
            - "low": Lower bound values
            - "shape": Shape of the action space
            - "dtype": Data type of the action space
    """
    return {"high": action_space.high,
            "low": action_space.low,
            "shape": action_space.shape,
            "dtype": str(action_space.dtype)
            }


def get_random_order(elements):
    """
    Randomize the order of elements in an array.

    Args:
        elements: array-like
            The elements to be randomly ordered.

    Returns:
        array-like: A random permutation of the input elements.
    """
    return np.random.permutation(elements)


def get_elements_order(b_random_order, elements):
    """
    Get the order of elements either in a random order or in ascending order.

    Args:
        b_random_order: bool
            If True, the elements will be in random order, otherwise in ascending order.
        elements: array-like
            The elements to be ordered.

    Returns:
        array-like: Ordered elements according to the specified order.
    """
    if b_random_order:
        ordered_elements = get_random_order(elements)
    else:
        ordered_elements = np.arange(1, len(elements) + 1)

    return ordered_elements

