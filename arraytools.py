import logging, sys

loaded_tables = {}

default_logger = logging.getLogger("arraytools") 
default_logger.setLevel(logging.INFO)  
log_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s   %(message)s ","%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
default_logger.addHandler(stream_handler)
default_logger.propagate = False

def set_logger(arg_logger):

    # set to default level
    default_logger.setLevel(logging.INFO)

    logging_levels_int = { 0: logging.CRITICAL, 
                           1: logging.ERROR,
                           2: logging.WARNING,
                           3: logging.INFO,
                           4: logging.DEBUG }

    logging_levels_str = { 'critical' : logging.CRITICAL, 
                           'error' : logging.ERROR,
                           'warning' : logging.WARNING,
                           'info' : logging.INFO,
                           'debug' : logging.DEBUG }

    if type(arg_logger) == int:

        try:
            if arg_logger in [logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]:
                default_logger.setLevel(arg_logger)
                logger = default_logger
            else:
                default_logger.setLevel(logging_levels_int[arg_logger])
                logger = default_logger
        except:
            raise Exception('Unknow log level %d' % arg_logger) 
    elif type(arg_logger) == str:
        try:
            default_logger.setLevel(logging_levels_str[arg_logger])
            logger = default_logger
        except:
            raise Exception('Unknow log level %s' % arg_logger) 
    elif type(arg_logger) == type(default_logger):

        logger = arg_logger

    return logger
    
def save(filepath,arr,clobber=False,logger=default_logger):

    logger = set_logger(logger)
    import pyfits
    if filepath.split('.')[-1] == 'fits' or filepath.split('.')[-1] == 'fit' or filepath.split('.')[-2] == 'fits' or filepath.split('.')[-2] == 'fit':
        import pyfits, warnings, os
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            exists = os.path.isfile(filepath)
            pyfits.writeto(filepath,arr,clobber=clobber)
            if exists:
                logger.info('overwriting %s with %d rows',filepath,len(arr))
            else:
                logger.info('saved %s with %d rows',filepath,len(arr))
    elif filepath.split('.')[-1] == 'cpickle':
        import cPickle as pickle
        import os
        if os.path.isfile(filepath):
            if clobber==False:
                raise Exception('file exists %s' % filename)
            else:
                pickle.dump(arr,open(filepath,'w'),protocol=2)
                logger.info('overwrite pickle %s',filepath)
        else:
            pickle.dump(arr,open(filepath,'w'),protocol=2)
            logger.info('wrote new pickle %s',filepath)


def load(filepath,remember=False,dtype=None,hdu=None,logger=default_logger,skiprows=0):

    logger = set_logger(logger)

    if (filepath in loaded_tables) and remember:

        logger.debug('using preloaded array %s' % filepath)
        table = loaded_tables[filepath]
    
    else:

        logger.debug('loading %s' % filepath)
        if filepath.split('.')[-1] == 'pp' or filepath.split('.')[-1] == 'cpickle' or filepath.split('.')[-1] == 'pp2':
                import cPickle as pickle
                file_pickle = open(filepath)
                table = pickle.load(file_pickle)
                file_pickle.close()
                logger.info('loaded pickle %s' % (filepath))

        elif filepath.split('.')[-1] == 'fits' or filepath.split('.')[-1] == 'fit' or filepath.split('.')[-2] == 'fits' or filepath.split('.')[-2] == 'fit':
                import pyfits
                table = pyfits.getdata(filepath,hdu)
                import numpy
                table = numpy.asarray(table)

                # if using FITS_record, get FITS rec
                if isinstance(table,pyfits.FITS_record):
                    table = table.array
                    import numpy
                    table = numpy.asarray(table)
                logger.info('loaded %s, got %d rows' % (filepath,len(table)))
        elif filepath.split('.')[-1] == 'h5' or filepath.split('.')[-1] == 'hdf5':

                raise Exception('h5 hot implemented')

        else:
                import numpy
                table = numpy.loadtxt(filepath,dtype=dtype,skiprows=skiprows)  
                logger.info('loaded %s, got %d rows' % (filepath,len(table)))

    if remember: loaded_tables[filepath] = table

    return table


def ensure_col(rec,name,dtype='f8',arr=None):

    import numpy as np
    if arr==None:
        arr = np.zeros(len(rec))

    if name not in rec.dtype.names:
        rec = add_col(arr=arr,name=name,rec=rec,dtype=dtype)
    else:
        rec[name] = arr
    return rec

def add_col(rec, name, arr='zeros', dtype=None):

    import numpy as np
    if arr=='zeros': arr=np.zeros(len(rec))

    arr = np.asarray(arr)
    if dtype is None:
        dtype = arr.dtype
    newdtype = np.dtype(rec.dtype.descr + [(name, dtype)])
    newrec = np.empty(rec.shape, dtype=newdtype)
    for field in rec.dtype.fields:
        newrec[field] = rec[field]
    newrec[name] = arr
    return newrec

def arr2rec(myarray,dtype):

    import numpy
    newrecarray = numpy.core.records.fromarrays(numpy.array(myarray).transpose(), dtype=dtype)
    return newrecarray