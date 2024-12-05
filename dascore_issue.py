import dascore as dc
import numpy as np

memspool = dc.examples.random_spool()
dir_spool = dc.examples.spool_to_directory(memspool)
spool = dc.spool(dir_spool)
# print(spool[0].coords.get_array('distance'))
distance_coords = spool[0].coords.get_array("distance")
# integer distance array
distance_array = distance_coords[np.arange(0, 298)]
print(distance_array)
start, end = 1, 50
# sub_spool = spool.select(distance=(distance_array))
sub_spool = spool.select(distance=(start, end))
# sub_spool = spool.select(distance=(1,50))
# print(sub_spool)
print(sub_spool[0])
