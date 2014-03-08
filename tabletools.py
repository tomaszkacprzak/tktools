import logging
import pyfits
import sys

loaded_tables = {}

# logging.basicConfig(level=logging.INFO,format='%(asctime)s %(name)s %(levelname)s',datefmt="%Y-%m-%d %H:%M:%S")
# default_log = logging.getLogger("tabletools") 

default_log = logging.getLogger("tabletools") 
default_log.setLevel(logging.INFO)  
log_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s   %(message)s ","%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
default_log.addHandler(stream_handler)
default_log.propagate = False

def setLog(log):

    logging_levels_int = { 0: logging.CRITICAL, 
                           1: logging.WARNING,
                           2: logging.INFO,
                           3: logging.DEBUG }

    logging_levels_str = { 'critical' : logging.CRITICAL, 
                           'warning' : logging.WARNING,
                           'info' : logging.INFO,
                           'debug' : logging.DEBUG }

    if type(log) == int:
        try:
            default_log.setLevel(logging_levels_int[log])
        except:
            raise Exception('Unknow log level %d' % log) 
    elif type(log) == str:
        try:
            default_log.setLevel(logging_levels_str[log])
        except:
            raise Exception('Unknow log level %s' % log) 

    log = default_log
    return log


def toMatrix(x):

    return x.view(np.float64).reshape(x.shape + (-1,))



def addTable(table_name,table,log=default_log):

    log = setLog(log)
    
    if not isinstance(table_name,str):
        raise Exception('table_name should be a type string and is %s' % str(type(table_name)))

    
    if table_name in loaded_tables:
        log.warning('replacing %s in loaded tables' % table_name)
        loaded_tables[table_name] = table


def loadTable(filepath,table_name='do_not_store',dtype=None,hdu=1,log=default_log):

    log = setLog(log)

    if (table_name in loaded_tables) and (table_name !='do_not_store'):

        log.debug('using preloaded array %s' % table_name)
        table = loaded_tables[table_name]
    
    else:

        log.debug('loading %s' % filepath)
        if filepath.split('.')[-1] == 'pp':
                import cPickle as pickle
                file_pickle = open(filepath)
                table = pickle.load(file_pickle)
                file_pickle.close()

        elif filepath.split('.')[-1] == 'fits' or filepath.split('.')[-1] == 'fit' or filepath.split('.')[-2] == 'fits' or filepath.split('.')[-2] == 'fit':
                import pyfits
                fits = pyfits.open(filepath)
                table = fits[hdu].data
                import numpy
                table = numpy.asarray(table)

                # if using FITS_record, get FITS rec
                if isinstance(table,pyfits.FITS_record):
                    table = table.array
                    import numpy
                    table = numpy.asarray(table)
        else:
                import numpy
                table = numpy.loadtxt(filepath,dtype=dtype)
        
    
    log.info('loaded %s correctly, got %d rows' % (filepath,len(table)))

    if ( table_name != 'do_not_store' ): loaded_tables[table_name] = table

    return table

def saveTable(filepath,table,log=default_log,append=False):

    log = setLog(log)

    import numpy
    formats = { numpy.dtype('int64') : '% 12d' ,
                numpy.dtype('int32') : '% 12d' ,
                numpy.dtype('float32') : '% .10e' ,
                numpy.dtype('float64') : '% .10e' ,
                numpy.dtype('>i8') : '% 12d',
                numpy.dtype('>f8') : '% .10f',
                numpy.dtype('S1024') : '%s',}

    if filepath.split('.')[-1] == 'pp':

        if append == True:
            log.error('appending a pickle not supported yet')
            raise Exception('appending a pickle not supported yet');

        import cPickle as pickle
        file_pickle = open(filepath,'w')
        pickle.dump(table,file_pickle,protocol=2)
        file_pickle.close()



    elif filepath.split('.')[-1] == 'fits' or filepath.split('.')[-2] == 'fits':
        
        import pyfits
        if append:
            # if type(table) == numpy.ndarray:
            pyfits.append(filepath,table)
            print 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA append' , pyfits.open(filepath) , type(table)
        else:
            if type(table) is pyfits.core.HDUList:
                fits_obj_to_write = table
            else:
                fits_obj_to_write = getFITSTable(table)
            fits_obj_to_write.writeto(filepath,clobber=True)

    else:

        if append:
            log.error('appending a pickle not supported yet')
            raise Exception('appending a pickle not supported yet');

        header = '# ' + ' '.join(table.dtype.names)
        fmt = [formats[table.dtype.fields[f][0]] for f in table.dtype.names]
        float(numpy.__version__[0:3])
        if float(numpy.__version__[0:3]) >= 1.7:
            numpy.savetxt(filepath,table,header=header,fmt=fmt,delimiter='\t')
        else:
            numpy.savetxt(filepath,table,fmt=fmt,delimiter='\t')
            with file(filepath, 'r') as original: data = original.read()
            with file(filepath, 'w') as modified: 
                modified.write(header + '\n' + data)
                modified.close()
            
    if append:
        log.info('appended table %s, %d rows' % (filepath,len(table)))
        # log.info('appended table %s, %d rows, new number of HDUs %d' % (filepath,len(table),n_hdus))
    else:
        log.info('saved table %s, %d rows' % (filepath,len(table)))


def getBinaryTable(numpy_array):

    import numpy
    import pyfits
    numpy_array = numpy.array(numpy_array)
    formats = { numpy.dtype('int64') : 'K' , 
                numpy.dtype('int16') : 'I' , 
                numpy.dtype('int32') : 'J' , 
                numpy.dtype('float32') : 'E' , 
                numpy.dtype('float64') : 'D' ,
                numpy.dtype('>i2') : 'I', 
                numpy.dtype('>i8') : 'K', 
                numpy.dtype('>i4') : 'I', 
                numpy.dtype('>f4') : 'E' , 
                numpy.dtype('>f8') : 'D',
                numpy.dtype('S1024') : '1024A',
                numpy.dtype('S4') : '4A'}
    # formats = { numpy.dtype('int64') : '' , 
    #             numpy.dtype('int16') : 'I' , 
    #             numpy.dtype('int32') : 'J' , 
    #             numpy.dtype('float32') : 'E' , 
    #             numpy.dtype('float64') : 'D' ,
    #             numpy.dtype('>i2') : 'I', 
    #             numpy.dtype('>i8') : 'K', 
    #             numpy.dtype('>i4') : 'I', 
    #             numpy.dtype('>f4') : 'E' , 
    #             numpy.dtype('>f8') : 'D'}

    cols = []

    for i,col_name in enumerate(numpy_array.dtype.names):
        col_type = numpy_array.dtype[i]
        col_fmt = formats[col_type]
        col = pyfits.Column(name=col_name,format=col_fmt,array=numpy_array[col_name])
        cols.append(col)

    tbhdu = pyfits.new_table(pyfits.ColDefs(cols))
    return tbhdu


def getFITSTable(numpy_array):

    tbhdu = getBinaryTable(numpy_array)
    hdu = pyfits.PrimaryHDU()
    hdulist = pyfits.HDUList([hdu, tbhdu])

    return hdulist




def getColNames(table):
    
    import numpy
    import pyfits

    if isinstance(table,pyfits.FITS_record):
        colnames = table.array.dtype.names
    elif isinstance(table,pyfits.FITS_rec):
        colnames = table.dtype.names
    elif isinstance(table,numpy.ndarray):
        colnames = table.dtype.names

    return colnames

def appendColumn(rec, name, arr, dtype=None):
    import numpy
    arr = numpy.asarray(arr)
    if dtype is None:
        dtype = arr.dtype
    newdtype = numpy.dtype(rec.dtype.descr + [(name, dtype)])
    newrec = numpy.empty(rec.shape, dtype=newdtype)
    for field in rec.dtype.fields:
        newrec[field] = rec[field]
    newrec[name] = arr
    return newrec

def array2recarray(myarray,dtype):

    import numpy
    newrecarray = numpy.core.records.fromarrays(myarray.transpose(), dtype=dtype)
    return newrecarray
