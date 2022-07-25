
def to_list(arg) -> list:
    """
    Converts non-iterable object to a list. If `arg` is already a list or tuple the same object is returned.

    Parameters
    ----------
    arg : non-iterable or list or tuple
        Non-iterable, which should be converted to a list.

    Returns
    -------
    arg_list : list or tuple
        List containing `arg` as a value.

    """
    if arg is None:
        arg_list = []
    elif not isinstance(arg, (list, tuple)):
        arg_list = [arg]
    else:
        arg_list = arg

    return arg_list
