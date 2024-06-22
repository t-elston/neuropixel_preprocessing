## neuropixel_preprocessing: spike-sorting, event extraction, synchronization

# STOP - this version is deprecated as of kilosort 4. See [here](https://github.com/t-elston/ks4_pipeline) for the current KS4 sorting pipeline. 

This is a pipeline which spike-sorts neuropixel data aquired with [SpikeGLX](https://billkarsh.github.io/SpikeGLX/) and task control via either [NIMH MonkeyLogic](https://monkeylogic.nimh.nih.gov/index.html) or [PsychToolbox](https://psychtoolbox.org/). 

## Automated spike-sorting and unit curation

`npx_spike_sorting_pipeline.ipynb` makes use of [Spike Interface](https://spikeinterface.readthedocs.io/en/latest/) to clean your datastreams, run [Kilosort 3](https://github.com/MouseLand/Kilosort), and compute quality metrics for each of the putative units identified by Kilosort. To use this notebook straight out of the box, you'll need to have Kilosort 3 (and [Matlab](https://www.mathworks.com/products/matlab.html)) installed on your computer. This pipeline is optimized for recordings made with [SpikeGLX](https://billkarsh.github.io/SpikeGLX/) where each probe has its own folder. 

If you've got everything set up as described above, all you need to do the run the notebook is modify a few path variables in the second code cell. Specifically:
- `base_folder`: path to the folder containing your recording data (the one that has the imec0 and imec1 sub-directories in it). 
- `kilosort3_path`: path to the folder containing Kilosort 3

If you are recording from multiple brain areas, you can also modify the `brain_areas` variable such that the first element is the brain area you lowered the `imec0` probe into and the second element is the brain area you lowered the `imec1` probe into. Extend this list to as many brain areas as you have probes. 

**<ins>Note:</ins>** This pipeline will make two copies of your original data - the first after preprocessing by SpikeInterface, the second after spike-sorting. The pipeline will automatically delete the intermediate file produced by Kilosort but cannot delete the one created by SpikeInterface. This won't stop the pipeline from working but you should be sure to manually delete the "preprocess" subdirectory in each probe folder after each run. A printed message will inform you whether deleting these intermediate files was successful. 

## Extracting event codes and aligning your data onto a common timeline

`npx_spike_sorting_pipeline.ipynb` also extracts the event codes and edges of synchronization pulses across all of your data streams and places all of your spike/event data onto a common timeline. 

**<ins>Note:</ins>** For the nidaq stream, it assumes that you are using analog channel 0 as your sync_channel - in other words, that you have run a coax cable from the IMEC card on the NI-chassis to analog channel 0 on your nidaq board. 
