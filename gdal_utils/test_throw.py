
from itertools import tee, izip

lst = list('abcdefghi')
lst.append(None)
pairs = izip(lst[::2], lst[1::2])

for i in pairs:
    print i