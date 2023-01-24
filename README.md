# Soundscape categorization

This notebooks are the compliation of the code use for the publication "Categorizing shallow marine soundscapes using 
unsupervised clustering and explainable machine learning". 

It is necessary to run the jupyter notebooks in the right order, as they will create files that are necessary 
for the next steps. 

0. Download the data directly from MDA. The dataset entry is stored here: https://www.vliz.be/en/imis?module=dataset&dasid=8138
To use the notebook to download the data you first need to register to the Marine Data Archive (MDA) to access the data 
(https://mda.vliz.be/archive.php). This dataset is just the output pypam after processing all the sound files of all 
the deployments specified in data/data_summary.csv

You can also skip this step if you already have manually downloaded the data 
and stored it on the right place data/raw_data (or changed all the paths of the notebooks accordingly).

1. Add environmental data: bpnsdata will add the environmental data to the output of pypam. It will add a file per 
deployment in data/processed, ending with _env.nc. This will be a netCDF file with the pypam output and the environmental 
data as extra data_vars in the xarray dataset. 

2. Prepare the data in one single DataFrame for the process. The output will be stored in data/processed. There will be
a df_complete.pkl, a df_no_artifacts.pkl, umap.pkl, umap_clean.pkl and used_freqticks.npy

3. Dimension reduction, categorization and interpretable machine learning. Output will be saved in form of images or 
plot inline in the jupyter notebook. All the images and files will be saved under the output/ folder. The models
are saved as joblib files. Some plots are also shown in the jupyter notebook. 

Data under geo/ is extracted from: 
* belgium_coast/: Marine Regions (https://www.marineregions.org/)
* bentehic_habitats/: Derous, S., Vincx, M., Degraer, S., Deneudt, K., Deckers, P., Cuvelier, D., Mees, J., Courtens, W., Stienen, E.W. M., Hillewaert, H., Hostens, K., Moulaert, I., Lancker, V. V., and Verfaillie, E. (2007). A biological valuation map for the Belgian Part of the North Sea. Global Change, page 162.https://biblio.ugent.be/publication/4391454
* boundaries/: Marine Regions (https://www.marineregions.org/)
* seabed_habitat: EMODnet Seabed Habitats (https://emodnet.ec.europa.eu/en/seabed-habitats) 
* bathymetry: EMODnet Bathymetry (https://emodnet.ec.europa.eu/en/bathymetry)
