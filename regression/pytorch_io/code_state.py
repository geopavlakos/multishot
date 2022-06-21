def get_code_state(args):
    if not args.log_code_state:
        return None, None

    import pathlib
    current = pathlib.Path(__file__).parent.absolute()

    return commit, patch
