import sys
import yaml


def open_file(fp, *args, **kargs):
    if sys.version_info[0] >= 3:
        kargs['encoding'] = 'utf8'
        return open(fp, *args, **kargs)
    else:
        return open(fp, *args, **kargs)


with open_file("aaa.yml") as ff:
    strs = yaml.load(ff)

print(strs)

with open("test.yml", "w") as f_w:
    yaml.dump(strs, f_w, allow_unicode = True)
