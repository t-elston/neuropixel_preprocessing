# neuropixel_preprocessing

I provide a module and a template jupyter notebook showing how you can extract event codes and map your neuropixel data onto a common timeline. 

This set of tools assumes you acquired your neuropixel data with SpikeGLX and you have already run Kilosort. Many functions in the module are either directly taken or modified from existing tools.

These are meant to be very minimal/easy scripts that don't require any fancy big program. Eventually I'll get quality metrics going in an equally easy-to-use manner. 


## some other helpful notes
It regularly seems to happen that KiloSort 3 doesn't save individual spike waveforms, which means that Phy cannot create its Feature Viewer - the critical view where you can see PC clusters and do manual merges and splits of clusters. The workaround is to force Phy to extract the waveforms. First, open the command line and navigate to the directory where your kilosort output is 
- `cd <your/ks/output/path>`
  
Then run the following from the command line:
- `phy extract-waveforms params.py`

Now you can run Phy like usual:
- `phy template-gui params.py`

