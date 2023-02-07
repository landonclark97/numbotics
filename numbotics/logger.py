import sys

INFO_HEADER = ' NUMBOT INFO: '
WARN_HEADER = ' NUMBOT WARNING: '
ERR_HEADER = ' NUMBOT ERROR: '

def info(info_string):
    print(INFO_HEADER + info_string)

def warning(warn_string):
    print(WARN_HEADER + warn_string)

def error(err_string):
    print(ERR_HEADER + err_string)
