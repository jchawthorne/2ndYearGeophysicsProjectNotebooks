import numpy as np
import obspy
import scipy
from scipy import signal
    

def interp_masked(st,tmask=3.):
    """
    :param        st: waveforms or trace
    :param     tmask: time window to mask within some interval or endpoint, 
                         in seconds
    :return      msk: a set of waveforms with masks
    """

    if isinstance(st,obspy.Trace):

        # allowable values
        try:
            ms=np.logical_or(st.data.mask,np.isnan(st.data.data))
            data=st.data.data
        except:
            ms = np.isnan(st.data)
            data=st.data

        # times
        tm=np.arange(0.,data.size)

        # interpolate
        if np.sum(~ms):
            data[ms]=np.interp(tm[ms],tm[~ms],data[~ms])
        else:
            data[:]=0.
        
        # and copy data
        st.data = data    

        # to mask
        nwin = int(tmask/st.stats.delta)
        nwin = np.maximum(nwin,1)
        win = scipy.signal.boxcar(nwin*2+1)
        ms = ms.astype(float)
        ms = scipy.signal.convolve(ms,win,mode='same')

        # also the beginning and end
        if tmask != 0.:
            ms[0:nwin+1]=1.
            ms[-nwin:]=1.

        # place in trace
        ms = np.minimum(ms,1.)
        msk = st.copy()
        msk.data = ms

    elif isinstance(st,obspy.Stream):
        msk = obspy.Stream()

        for tr in st:
            mski = interp_masked(tr,tmask=tmask)
            msk.append(mski)

    return msk



    

def add_mask(st,msk):
    """
    :param        st: waveforms or trace
    :param       msk: a set of waveforms with masks
    """
    
    if isinstance(st,obspy.Trace):
        # the mask
        #ms = msk.data.astype(bool)
        ms = msk.data > 0.1
        try:
            # if there's already a mask, combine them
            st.data.mask=np.logical_or(st.data.mask,ms)
        except:
            st.data=np.ma.masked_array(st.data,mask=ms)

    elif isinstance(st,obspy.Stream):

        for tr in st:
            # select mask
            ms = msk.select(id=tr.id)[0]

            # add filter
            add_mask(tr,ms)

