import numpy as np

import os, sys, warnings, logging
logger = logging.getLogger(os.path.basename(__file__)[:10])
if len(logger.handlers)==0:
    log_formatter = logging.Formatter("%(asctime)s %(name)0.10s %(levelname)0.3s   %(message)s ","%y-%m-%d %H:%M:%S")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)
warnings.filterwarnings("ignore",category=DeprecationWarning)

def add_cols(rec, names, data='zeros', dtypes='floats'):

    if data == 'zeros':
        data = [np.zeros(len(rec)) for x in names]
    if dtypes == 'floats':
        dtypes = [np.float64 for x in names]

    data = np.asarray(data)
    newdtype = np.dtype( rec.dtype.descr + zip(names, dtypes) )
    newrec = np.empty(rec.shape, dtype=newdtype)
    for field in rec.dtype.fields:
        newrec[field] = rec[field]
    for ni, name in enumerate(names):
        newrec[name] = data[ni]
    return newrec

def ensure_cols(rec, names, data='zeros', dtypes='floats'):

    if data == 'zeros':
        data = [np.zeros(len(rec)) for x in names]
    if dtypes == 'floats':
        dtypes = [np.float64 for x in names]

    list_new_names = []
    list_new_dtypes = []
    list_new_data = []
    for ni in range(len(names)):
        if names[ni] not in rec.dtype.names:
            list_new_names.append(names[ni])
            list_new_dtypes.append(dtypes[ni])
            list_new_data.append(data[ni])

    data = np.asarray(data)
    newdtype = np.dtype( rec.dtype.descr + zip(list_new_names, list_new_dtypes) )
    newrec = np.empty(rec.shape, dtype=newdtype)
    for field in rec.dtype.fields:
        newrec[field] = rec[field]
    for ni, name in enumerate(list_new_names):
        newrec[name] = list_new_data[ni]
    return newrec

def arr_to_rec(myarray, dtype):

    import numpy
    newrecarray = numpy.core.records.fromarrays(numpy.array(myarray).transpose(), dtype=dtype)
    return newrecarray


def new_array(n_rows, columns, ints=[], float_dtype=np.float64, int_dtype=np.int64):

    n_columns = len(columns)
    formats = [None]*n_columns
    for ic in range(n_columns):
        if columns[ic] in ints:
                formats[ic] = int_dtype
        else:
            formats[ic] = float_dtype
    newrec = np.zeros(n_rows, dtype=np.dtype( zip(columns, formats)))
    return newrec

def get_dtype(columns, main='f8'):

    list_name = []
    list_dtype = []
    for col in columns:
        if ':' in col:
            name, dtype = col.split(':')
        else:
            name, dtype = col, main
        list_name.append(name)
        list_dtype.append(dtype)

    dtype = np.dtype(zip(list_name, list_dtype))
    return dtype, list_name

def save_hdf(filename, arr):
    import h5py
    f5 = h5py.File(name=filename)
    try:
        f5.clear()
    except:
        for datasetname in f5.keys():
            del f5[datasetname]
    f5.create_dataset(name='data', data=arr)
    f5.close()
    logger.info('saved %s' % filename)

def load_hdf(filename):
    import h5py
    f5 = h5py.File(name=filename)
    data = np.array(f5['data'])
    f5.close()
    logger.info('loaded %s' % filename)
    return data
