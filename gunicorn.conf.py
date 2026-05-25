import gc


def when_ready(server):
    # Freeze all GC-tracked objects loaded during --preload so their reference
    # counts don't dirty copy-on-write pages after workers are forked.
    gc.freeze()
