"""
Module containing key preprocessing routines for neuropixels data.
Many of the tools came from Jennifer Colonell's SpikeGLX tools repo:
https://github.com/jenniferColonell/SpikeGLX_Datafile_Tools/tree/main/Python

Other functions are things I made. 

Created by Thomas Elston in December 2023
"""

import numpy as np
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tkinter import Tk
from tkinter import filedialog


# Parse ini file returning a dictionary whose keys are the metadata
# left-hand-side-tags, and values are string versions of the right-hand-side
# metadata values. We remove any leading '~' characters in the tags to match
# the MATLAB version of readMeta.
#
# The string values are converted to numbers using the "int" and "float"
# functions. Note that python 3 has no size limit for integers.

def get_path_from_dir(base_folder, file_name):
    """
    Search for a file within the specified directory and its subdirectories
    by matching the given file name.

    Args:
    - base_folder (str): The directory path to start the search from.
    - file_name (str): The specific file name or part of the file name to be matched.

    Returns:
    - str or None: The path to the first file found with the given name,
      or None if no file is found.
    """
    base_folder = Path(base_folder)
    
    # Iterate through all files and directories in the base_folder
    for file in base_folder.glob('**/*'):
        if file.is_file() and file_name in file.name:
            return file.resolve()  # Return the path of the first file found with the given name
    
    # Print an error message if no file with the given name is found
    print(f"No '{file_name}' file found in the directory.")
    return None


def readMeta(binFullPath):
    metaName = binFullPath.stem + ".meta"
    metaPath = Path(binFullPath.parent / metaName)
    metaDict = {}
    if metaPath.exists():
        # print("meta file present")
        with metaPath.open() as f:
            mdatList = f.read().splitlines()
            # convert the list entries into key value pairs
            for m in mdatList:
                csList = m.split(sep='=')
                if csList[0][0] == '~':
                    currKey = csList[0][1:len(csList[0])]
                else:
                    currKey = csList[0]
                metaDict.update({currKey: csList[1]})
    else:
        print("no meta file")
    return(metaDict)

def read_meta_from_path(metaPath):
    metaDict = {}
    if metaPath.exists():
        # print("meta file present")
        with metaPath.open() as f:
            mdatList = f.read().splitlines()
            # convert the list entries into key value pairs
            for m in mdatList:
                csList = m.split(sep='=')
                if csList[0][0] == '~':
                    currKey = csList[0][1:len(csList[0])]
                else:
                    currKey = csList[0]
                metaDict.update({currKey: csList[1]})
    else:
        print("no meta file")
    return(metaDict)


# Return sample rate as python float.
# On most systems, this will be implemented as C++ double.
# Use python command sys.float_info to get properties of float on your system.
#
def SampRate(meta):
    if meta['typeThis'] == 'imec':
        srate = float(meta['imSampRate'])
    else:
        srate = float(meta['niSampRate'])
    return(srate)


# Return a multiplicative factor for converting 16-bit file data
# to voltage. This does not take gain into account. The full
# conversion with gain is:
#         dataVolts = dataInt * fI2V / gain
# Note that each channel may have its own gain.
#
def Int2Volts(meta):
    if meta['typeThis'] == 'imec':
        if 'imMaxInt' in meta:
            maxInt = int(meta['imMaxInt'])
        else:
            maxInt = 512
        fI2V = float(meta['imAiRangeMax'])/maxInt
    else:
        fI2V = float(meta['niAiRangeMax'])/32768
    return(fI2V)


# Return array of original channel IDs. As an example, suppose we want the
# imec gain for the ith channel stored in the binary data. A gain array
# can be obtained using ChanGainsIM(), but we need an original channel
# index to do the lookup. Because you can selectively save channels, the
# ith channel in the file isn't necessarily the ith acquired channel.
# Use this function to convert from ith stored to original index.
# Note that the SpikeGLX channels are 0 based.
#
def OriginalChans(meta):
    if meta['snsSaveChanSubset'] == 'all':
        # output = int32, 0 to nSavedChans - 1
        chans = np.arange(0, int(meta['nSavedChans']))
    else:
        # parse the snsSaveChanSubset string
        # split at commas
        chStrList = meta['snsSaveChanSubset'].split(sep=',')
        chans = np.arange(0, 0)  # creates an empty array of int32
        for sL in chStrList:
            currList = sL.split(sep=':')
            if len(currList) > 1:
                # each set of contiguous channels specified by
                # chan1:chan2 inclusive
                newChans = np.arange(int(currList[0]), int(currList[1])+1)
            else:
                newChans = np.arange(int(currList[0]), int(currList[0])+1)
            chans = np.append(chans, newChans)
    return(chans)


# Return counts of each nidq channel type that composes the timepoints
# stored in the binary file.
#
def ChannelCountsNI(meta):
    chanCountList = meta['snsMnMaXaDw'].split(sep=',')
    MN = int(chanCountList[0])
    MA = int(chanCountList[1])
    XA = int(chanCountList[2])
    DW = int(chanCountList[3])
    return(MN, MA, XA, DW)


# Return counts of each imec channel type that composes the timepoints
# stored in the binary files.
#
def ChannelCountsIM(meta):
    chanCountList = meta['snsApLfSy'].split(sep=',')
    AP = int(chanCountList[0])
    LF = int(chanCountList[1])
    SY = int(chanCountList[2])
    return(AP, LF, SY)


# Return gain for ith channel stored in nidq file.
# ichan is a saved channel index, rather than the original (acquired) index.
#
def ChanGainNI(ichan, savedMN, savedMA, meta):
    if ichan < savedMN:
        gain = float(meta['niMNGain'])
    elif ichan < (savedMN + savedMA):
        gain = float(meta['niMAGain'])
    else:
        gain = 1    # non multiplexed channels have no extra gain
    return(gain)


# Return gain for imec channels.
# Index into these with the original (acquired) channel IDs.
#
def ChanGainsIM(meta):
    imroList = meta['imroTbl'].split(sep=')')
    # One entry for each channel plus header entry,
    # plus a final empty entry following the last ')'
    nChan = len(imroList) - 2
    APgain = np.zeros(nChan)        # default type = float
    LFgain = np.zeros(nChan)
    if 'imDatPrb_type' in meta:
        probeType = int(meta['imDatPrb_type'])
    else:
        probeType = 0
    if (probeType == 21) or (probeType == 24):
        # NP 2.0; APGain = 80 for all AP
        # return 0 for LFgain (no LF channels)
        APgain = APgain + 80
    else:
        # 3A, 3B1, 3B2 (NP 1.0)
        for i in range(0, nChan):
            currList = imroList[i+1].split(sep=' ')
            APgain[i] = currList[3]
            LFgain[i] = currList[4]
    return(APgain, LFgain)


# Having accessed a block of raw nidq data using makeMemMapRaw, convert
# values to gain-corrected voltage. The conversion is only applied to the
# saved-channel indices in chanList. Remember, saved-channel indices are
# in the range [0:nSavedChans-1]. The dimensions of dataArray remain
# unchanged. ChanList examples:
# [0:MN-1]  all MN channels (MN from ChannelCountsNI)
# [2,6,20]  just these three channels (zero based, as they appear in SGLX).
#
def GainCorrectNI(dataArray, chanList, meta):
    MN, MA, XA, DW = ChannelCountsNI(meta)
    fI2V = Int2Volts(meta)
    # print statements used for testing...
    # print("NI fI2V: %.3e" % (fI2V))
    # print("NI ChanGainNI: %.3f" % (ChanGainNI(0, MN, MA, meta)))

    # make array of floats to return. dataArray contains only the channels
    # in chanList, so output matches that shape
    convArray = np.zeros(dataArray.shape, dtype=float)
    for i in range(0, len(chanList)):
        j = chanList[i]             # index into timepoint
        conv = fI2V/ChanGainNI(j, MN, MA, meta)
        # dataArray contains only the channels in chanList
        convArray[i, :] = dataArray[i, :] * conv
    return(convArray)


# Having accessed a block of raw imec data using makeMemMapRaw, convert
# values to gain corrected voltages. The conversion is only applied to
# the saved-channel indices in chanList. Remember saved-channel indices
# are in the range [0:nSavedChans-1]. The dimensions of the dataArray
# remain unchanged. ChanList examples:
# [0:AP-1]  all AP channels
# [2,6,20]  just these three channels (zero based)
# Remember that for an lf file, the saved channel indices (fetched by
# OriginalChans) will be in the range 384-767 for a standard 3A or 3B probe.
#
def GainCorrectIM(dataArray, chanList, meta):
    # Look up gain with acquired channel ID
    chans = OriginalChans(meta)
    APgain, LFgain = ChanGainsIM(meta)
    nAP = len(APgain)
    nNu = nAP * 2

    # Common conversion factor
    fI2V = Int2Volts(meta)

    # make array of floats to return. dataArray contains only the channels
    # in chanList, so output matches that shape
    convArray = np.zeros(dataArray.shape, dtype='float')
    for i in range(0, len(chanList)):
        j = chanList[i]     # index into timepoint
        k = chans[j]        # acquisition index
        if k < nAP:
            conv = fI2V / APgain[k]
        elif k < nNu:
            conv = fI2V / LFgain[k - nAP]
        else:
            conv = 1
        # The dataArray contains only the channels in chanList
        convArray[i, :] = dataArray[i, :]*conv
    return(convArray)

# Return memmap for the raw data
# Fortran ordering is used to match the MATLAB version
# of these tools.
#
def makeMemMapRaw(binFullPath, meta):
    nChan = int(meta['nSavedChans'])
    nFileSamp = int(int(meta['fileSizeBytes'])/(2*nChan))
    print("n_channels: %d, n_file_samples: %d" % (nChan, nFileSamp))
    rawData = np.memmap(binFullPath, dtype='int16', mode='r',
                        shape=(nChan, nFileSamp), offset=0, order='F')
    return(rawData)


# Return an array [lines X timepoints] of uint8 values for a
# specified set of digital lines.
#
# - dwReq is the zero-based index into the saved file of the
#    16-bit word that contains the digital lines of interest.
# - dLineList is a zero-based list of one or more lines/bits
#    to scan from word dwReq.
#
def ExtractDigital(rawData, firstSamp, lastSamp, dwReq, dLineList, meta):
    # Get channel index of requested digital word dwReq
    if meta['typeThis'] == 'imec':
        AP, LF, SY = ChannelCountsIM(meta)
        if SY == 0:
            print("No imec sync channel saved.")
            digArray = np.zeros((0), 'uint8')
            return(digArray)
        else:
            digCh = AP + LF + dwReq
    else:
        MN, MA, XA, DW = ChannelCountsNI(meta)
        if dwReq > DW-1:
            print("Maximum digital word in file = %d" % (DW-1))
            digArray = np.zeros((0), 'uint8')
            return(digArray)
        else:
            digCh = MN + MA + XA + dwReq

    selectData = np.ascontiguousarray(rawData[digCh, firstSamp:lastSamp+1], 'int16')
    nSamp = lastSamp-firstSamp + 1

    # unpack bits of selectData; unpack bits works with uint8
    # original data is int16
    bitWiseData = np.unpackbits(selectData.view(dtype='uint8'))
    # output is 1-D array, nSamp*16. Reshape and transpose
    bitWiseData = np.transpose(np.reshape(bitWiseData, (nSamp, 16)))

    nLine = len(dLineList)
    digArray = np.zeros((nLine, nSamp), 'uint8')
    for i in range(0, nLine):
        byteN, bitN = np.divmod(dLineList[i], 8)
        targI = byteN*8 + (7 - bitN)
        digArray[i, :] = bitWiseData[targI, :]
    return(digArray)


# Sample calling program to get a file from the user,
# read metadata fetch sample rate, voltage conversion
# values for this file and channel, and plot a small range
# of voltages from a single channel.
# Note that this code merely demonstrates indexing into the
# data file, without any optimization for efficiency.

def load(folder, filename):

    """
    Loads a numpy file from a folder.

    Inputs:
    -------
    folder : String
        Directory containing the file to load
    filename : String
        Name of the numpy file

    Outputs:
    --------
    data : numpy.ndarray
        File contents

    """

    return np.load(os.path.join(folder, filename))
    

def extract_sync_edges(sync_files):

    print('Extracting sync edges...\n')
    
    for this_file in sync_files:

        this_file = str(this_file)

        # parse whether this file is a probe or nidaq and get the appropriate sync_channel
        if 'imec' in str(this_file):
            sync_channel = [384]
        else:
            sync_channel = [0]

        # read in the associated meta data
        meta = readMeta(Path(this_file))
        samp_rate = SampRate(meta)

        # load in the associated data
        rawData = makeMemMapRaw(Path(this_file), meta)

        # pull out the sync_channel
        sync_data = rawData[sync_channel, ]

        # rescale the data between 0 and 1 to make edge detection easy/uniform
        sync_data = ((sync_data - np.min(sync_data)) / (np.max(sync_data) - np.min(sync_data))).flatten()

        # now find the rising edges of the sync_data
        edges = np.argwhere(np.diff(sync_data) > .5)

        # convert edges to times
        edge_times = 1000*edges/samp_rate

        # sanity check plotting - plot first 5 seconds of data
        # plt.plot(sync_data)
        # plt.plot(edges, (sync_data[edges])+1.1, marker='o', linestyle='none')
        # plt.xlim((0, 5*samp_rate))
        # plt.title(this_file)

        # save the edge times
        file_folder = Path(this_file).parent
        save_name = 'edge_times'

        edge_path = Path(file_folder / save_name)

        # now save the sync edge times in the relevant directory
        np.save(edge_path, edge_times)

    print('\nSync edge times saved in original directory.')

def extract_event_codes(nidq_file):

    print('Extracting event codes from nidq stream...\n')
    # let's get the event codes from the nidaq stream

    # read in the associated meta data
    ni_meta = readMeta(Path(nidq_file))
    ni_samp_rate = SampRate(ni_meta)

    dw = 0    
    # Which lines within the digital word, zero-based
    dLineList = np.arange(0,8)

    # Load in the associated data
    nidq_data_raw = makeMemMapRaw(Path(nidq_file), ni_meta)
    n_channels, n_samples = nidq_data_raw.shape
    first_samp = 0
    last_samp = int(n_samples-1)

    # Get digital data for the selected lines
    digital_lines = ExtractDigital(nidq_data_raw, first_samp, last_samp, dw, dLineList, ni_meta)

    # Convert the digital_lines array into an 8-bit representation
    # Reshape the array to get each 8-bit word in a row
    eight_bit_words = digital_lines.T.reshape(-1, 8)

    # Convert each 8-bit word to a decimal value (MSB to LSB)
    decimal_values = np.packbits(eight_bit_words[:, ::-1], axis=1)

    # Find indices where the 8-bit word changes
    change_points = np.where(decimal_values[:-1] != decimal_values[1:])[0] + 1

    # Find the event codes where the change occured
    event_codes = decimal_values[change_points].flatten()

    # Convert event times to milliseconds
    event_times = 1000*change_points/ni_samp_rate

    # construct the name of the file to save the event codes and times
    event_save_name = Path(Path(nidq_file).parent / 'raw_event_codes')

    # save the data as a single array with two columns: 
    # one for event ID and the other for the time in MS
    np.save(event_save_name, np.array([event_codes, event_times]).T)

    print('\nEvent codes saved in original directory.')


def align_data_streams(sync_files, data_stream_info, reference_stream):

    sync_dirs = [os.path.dirname(s_file) for s_file in sync_files]

    print('Mapping spikes and task events to a common timeline.\n')

    # load the sync edge times (in milliseconds!) associated with the reference stream
    ref_edge_times = load(sync_dirs[reference_stream], 'edge_times.npy')

    print(data_stream_info[reference_stream] + ' set as master timeline.')

    # loop through each data stream, pull out the associated sync edges and data and then interpolate 
    # into the reference data stream
    for ix, this_dir in enumerate(sync_dirs):

        # pull out the sync edge times
        stream_sync_edges = load(this_dir, 'edge_times.npy')
        
        # get the sampling rate for this data stream
        dir_contents = os.listdir(this_dir)

        # find the meta file associated with the data stream
        meta_ix = []
        for fname in dir_contents:

            # is this a probe or the nidaq stream?
            if 'imec' in data_stream_info[ix]:
                meta_ix.append('ap.meta' in fname)

            else: # we're dealing with the nidaq stream
                meta_ix.append('nidq.meta' in fname)

        meta_name = dir_contents[int(np.argwhere(meta_ix))]

        # now load that meta file
        print('aligning: ' + str(this_dir))
        meta_path = Path(str(this_dir) + '/' + meta_name)

        meta = read_meta_from_path(meta_path)
        sampling_rate = SampRate(meta)

        # now that we have the sampling rate, let's pull in the spikes/event codes
        # if we're dealing with a probe stream, grab the spike times
        if 'imec' in data_stream_info[ix]:

            # load the sample indices associated with each spike
            spike_dir = Path(str(this_dir) + '/' + 'ks3_out' + '/' + 'sorter_output')
            spike_samples = load(spike_dir, 'spike_times.npy')

            # convert into milliseconds
            spike_times = 1000*(spike_samples/sampling_rate)

            # map the spike_times into the reference timeline
            # sync_spike_times = np.interp(spike_times, stream_times, ref_times)
            sync_spike_times = np.interp(spike_times.flatten(), stream_sync_edges.flatten(), ref_edge_times.flatten())

            # create path to where to save data
            save_path = Path(Path(spike_dir) / 'sync_spike_times')

            # now save these sync'd spike times as a new numpy array in its original directory
            np.save(save_path, sync_spike_times)

        else: # we are dealing with the nidaq stream

            # load the event codes and times
            event_codes = load(this_dir, 'raw_event_codes.npy')
            event_times = event_codes[:, 1]

            # map the event_times into the reference timeline
            # sync_event_times = np.interp(event_times, stream_times, ref_times)
            sync_event_times = np.interp(event_times.flatten(), stream_sync_edges.flatten(), ref_edge_times.flatten())

            sync_codes = event_codes
            sync_codes[:,1] = sync_event_times

            # create path to where to save data
            save_path = Path(Path(this_dir) / 'sync_event_codes')

            # now save these sync'd spike times as a new numpy array in its original directory
            np.save(save_path, sync_codes)
    
    print('\nAligned spikes and event codes saved in their original directories.')

def align_data_streams2(raw_dirs, data_stream_info, reference_stream):

    print('Mapping spikes and task events to a common timeline.\n')

    # load the sync edge times (in milliseconds!) associated with the reference stream
    ref_edge_times = load(raw_dirs[reference_stream], 'edge_times.npy')

    print(data_stream_info[reference_stream] + ' set as master timeline.')

    # loop through each data stream, pull out the associated sync edges and data and then interpolate 
    # into the reference data stream
    for ix, this_dir in enumerate(raw_dirs):

        # pull out the sync edge times
        stream_sync_edges = load(this_dir, 'edge_times.npy')
        
        # get the sampling rate for this data stream
        dir_contents = os.listdir(this_dir)

        # find the meta file associated with the data stream
        meta_ix = []
        for fname in dir_contents:

            # is this a probe or the nidaq stream?
            if 'probe' in data_stream_info[ix]:
                meta_ix.append('ap.meta' in fname)

            else: # we're dealing with the nidaq stream
                meta_ix.append('nidq.meta' in fname)

        meta_name = dir_contents[int(np.argwhere(meta_ix))]

        # now load that meta file
        print('aligning: ' + this_dir)
        meta_path = Path(this_dir + '/' + meta_name)

        meta = read_meta_from_path(meta_path)
        sampling_rate = SampRate(meta)

        # now that we have the sampling rate, let's pull in the spikes/event codes
        # if we're dealing with a probe stream, grab the spike times
        if 'probe' in data_stream_info[ix]:

            # load the sample indices associated with each spike
            spike_dir = this_dir / 'ks3_out' / 'sorter_output'
            spike_samples = load(spike_dir, 'spike_times.npy')

            # convert into milliseconds
            spike_times = 1000*(spike_samples/sampling_rate)

            # map the spike_times into the reference timeline
            # sync_spike_times = np.interp(spike_times, stream_times, ref_times)
            sync_spike_times = np.interp(spike_times.flatten(), stream_sync_edges.flatten(), ref_edge_times.flatten())

            # create path to where to save data
            save_path = Path(spike_dir / 'sync_spike_times')

            # now save these sync'd spike times as a new numpy array in its original directory
            np.save(save_path, sync_spike_times)

        else: # we are dealing with the nidaq stream

            # load the event codes and times
            event_codes = load(this_dir, 'raw_event_codes.npy')
            event_times = event_codes[:, 1]

            # map the event_times into the reference timeline
            # sync_event_times = np.interp(event_times, stream_times, ref_times)
            sync_event_times = np.interp(event_times.flatten(), stream_sync_edges.flatten(), ref_edge_times.flatten())

            sync_codes = event_codes
            sync_codes[:,1] = sync_event_times

            # create path to where to save data
            save_path = Path(this_dir / 'sync_event_codes')

            # now save these sync'd spike times as a new numpy array in its original directory
            np.save(save_path, sync_codes)
    
    print('\nAligned spikes and event codes saved in their original directories.')