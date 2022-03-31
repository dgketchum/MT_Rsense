David Ketchum
7 March 2022

PRMS/GSFLOW in the Upper Yellowstone River Basin.


The following instructions use code that depends on two disparate Python environments and projects.

1) gsflow-arcpy, download at https://github.com/dgketchum/gsflow-arcpy
    - this is a branch of the original work in github.com/gsflow/gsflow-arcpy, credit to the authors
    - our branch has an additional batch runnin script, and my work on Upper Yellowstone in

2) MT_Rsense/gsflow_prep, from https://github.com/dgketchum/MT_Rsense

Building the model domain:

All data sources for this project were projected into EPSG 5071 before use in any of the following analysis.

The gsflow-arcpy project is used to build the model domain and depends on an ArcGIS installation. The procedure
follows a lengthy list of commands to convert a DEM and soils rasters into hydrological response units (HRUs), a
stream flow network, sub-basins, etc. The gsflow-arcpy procedures take us from the raw data through the creation of
precipitation data correction surfaces that will be used in PRMS and GSFLOW. These steps are laid out clearly
in the gsflow_arcpy_tutorial.pdf document (the 'tutorial'). The code in MT_Rsense/gsflow_prep directory is
used to procure the raw data before running any gsflow-arcpy, to automate the creation of the precipation zones
before running prism_800m_normals.py and ppt_ratio_parameters.py. In the course of building the precipitation
zones shapefile, weather stations and gages will be automatically selected, the data from their records extracted
and written to the datafile (e.g., 'uy.data'), a prerequisite for subsequent work in PRMS/GSFLOW.

Environments:
    - For the MT_Rsense workflow, use conda to run the following commands to get the needed
      packages (see https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html:

    - conda create -n gsflow python=3.7
    - conda activate gsflow
    - conda config --env --add channels conda-forge
    - conda config --env --set channel_priority strict
    - conda install -c conda-forge gdal pyproj
    - conda install numpy scikit-learn shapely fiona
    - pip install rasterio rtree

Most scripts will need paths edited in each .py file. They can then be run from the conda command line
from a terminal with an active conda environment (for MT_Rsense code), or by calling the ArcGIS python executable
from the command line (for the gsflow-arcpy code; e.g., c:\python27\ArcGIS10.7\python.exe my_script.py). Further,
comment and uncomment function calls as needed in batch_sequence.py, so as to run only the needed functions.

Follow the steps listed below to reproduce my work on the Upper Yellowstone basin study. As steps are completed,
the resulting data's path should be edited in the project .ini file:

1. Produce a basin delineation using StreamStats, https://streamstats.usgs.gov/ss/
    - Delineate a basin by choosing and outlet just above the confluence of the Yellowstone and Shields rivers.
    - Export the basin to shapefile, project to EPSG 5071 (as with all data used in this project).

2. Clip raster data to the basin shapfile, using MT_Rsense/gsflow_prep/raster_prep.py and the statewide rasters
    in our database. Move rasters to directories specified in the project .ini.

3. To prepare climate and streamflow input data, run batch_sequence.py through impervious_parameters().
    Then run climate and stream gage data processing functions in gsflow_prep.py. This will build shapefiles
    with a selection of stream gages and climate stations, build the precipitation zones for the model domain,
    write the record of streamflow and climate data to a PRMS datafile, and calculate monthly min/max temperature
    lapse rates.

4. Run prism_800m_parameters() and ppt_ratio_parameters(). Open the hru_params.shp in a GIS and confirm the PPT_ZONE
    field has been set correctly. NOTE: I had to double the block size parameter (to 400000) in
    gsflow-arcpy/scripts/support_functions.py zone_by_centroid_func(), line #1181 when I found the basin area PPT_ZONE
    attributes were incomplete (zeros within the domain).

5. Run batch_sequence.py through flow_parameters(). Observe initial stream network, and change DEM_ADJ according
    to instructions in the tutorial. Re-run flow_parameters().  Run crt_fill_parameters(), examine. Re-run
    these two steps if necessary. See the gsflow-arcpy readme at github.com/dgketchum/gsflow-arcpy/README.txt.

Fix the .params file:
1. In unix, open each .params file and :set ff=unix then :wq! to remove the carraige return characters.
2. Set nlake and nlake_hrus to their true values.

