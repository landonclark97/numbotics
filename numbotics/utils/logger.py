import sys

INFO_HEADER = " NUMBOT INFO: "
WARN_HEADER = " NUMBOT WARNING: "
ERR_HEADER = " NUMBOT ERROR: "


def info(info_string, *args, **kwargs):
    print(INFO_HEADER + info_string, *args, **kwargs)


def warning(warn_string, *args, **kwargs):
    print(WARN_HEADER + warn_string, *args, **kwargs)


def error(err_string, *args, **kwargs):
    print(ERR_HEADER + err_string, *args, **kwargs)
