# neuropixel_preprocessing - spike-sorting, event extraction, synchronization

I provide a set of jupyter notebooks which implement spike sorting and basic post-sorting data alignment procedures. 

## Automated spike-sorting and unit curation

The first notenook is `npx_spike_sorting_pipeline.ipynb` - this makes use of [Spike Interface](https://spikeinterface.readthedocs.io/en/latest/) to clean your datastreams, run [Kilosort 3](https://github.com/MouseLand/Kilosort), and compute quality metrics for each of the putative units identified by Kilosort. To use this notebook straight out of the box, you'll need to have Kilosort 3 (and [Matlab](https://www.mathworks.com/products/matlab.html)) installed on your computer. This pipeline is optimized for recordings made with [SpikeGLX](https://billkarsh.github.io/SpikeGLX/) where each probe has its own folder. 

If you've used got everything set up as described above, all you need to do the run the notebook is modify a few path variables in the third code cell. Specifically:
- `base_folder`: path to the folder containing your recording data (the one that has the imec0 and imec1 sub-directories in it). 
- `kilosort3_path`: path to the folder containing Kilosort 3

If you are recording from multiple brain areas, you can also modify the `brain_areas` variable such that the first element is the brain area you lowered the `imec0` probe into and the second element is the brain area you lowered the `imec1` probe into. You can extend this list to as many brain areas as you have probes. 

**<ins>Note:</ins>** This pipeline will make two copies of your original data - the first after preprocessing, the second after spike-sorting. The pipeline will attempt to automatically delete these intermediate files but it's possible that it can't. This won't stop the pipeline from working but you should be sure to manually delete those files in case. A printed message will inform you whether deleting these intermediate files was successful. 

## Extracting event codes and aligning your data onto a common timeline

**<ins>Note:</ins>** This is under active development to be integrated as a step in `npx_spike_sorting_pipeline.ipynb`. 

`npx_preprocessing_pipeline.ipynb` is a jupyter notebook which extracts the event codes and edges of syncrhonization pulses across all of your data streams and places all of your spike/event data onto a common timeline. 

**<ins>Note:</ins>** For the nidaq stream, it assumes that you are using analog channel 0 as your sync_channel - in other words, that you have run a coax cable from the IMEC card on the NI-chassis to analog channel 0 on your nidaq board. 

This set of tools assumes you acquired your neuropixel data with SpikeGLX and you have already run Kilosort. Many functions in the module are either directly taken or modified from [existing tools](https://github.com/jenniferColonell/Neuropixels_evaluation_tools). 

I am actively working to integrate these steps into `npx_spike_sorting_pipeline.ipynb` - stay tuned! 


## some other helpful notes
It regularly seems to happen that KiloSort 3 doesn't save individual spike waveforms, which means that Phy cannot create its Feature Viewer - the critical view where you can see PC clusters and do manual merges and splits of clusters. The workaround is to force Phy to extract the waveforms. First, open the command line and navigate to the directory where your kilosort output is 
- `cd <your/ks/output/path>`
  
Then run the following from the command line:
- `phy extract-waveforms params.py`

Now you can run Phy like usual:
- `phy template-gui params.py`

