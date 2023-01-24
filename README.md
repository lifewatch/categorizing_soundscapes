# Soundscape categorization

This notebooks are the compliation of the code use for the publication "Categorizing shallow marine soundscapes using 
unsupervised clustering and explainable machine learning" (doi: XXX). 

To use it: you first need to register to the Marine Data Archive (MDA) to access the data 
(https://mda.vliz.be/archive.php). This dataset is just the output pypam after processing all the sound files of all 
the deployments specified in data/data_summary.csv
It is necessary to run the jupyter notebooks in the right order, as they will create files that are necessary 
for the next steps. 

0. Download the data directly from MDA. You can also skip this step if you already have manually downloaded the data 
and stored it on the right place data/raw_data (or changed all the paths of the notebooks accordingly).
1. Add environmental data: bpnsdata will add the environmental data to the output of pypam. It will add a file per 
deployment in data/raw_data, ending with _env.nc. This will be a netCDF file with the pypam output and the environmental 
data as extra data_vars in the xarray dataset. To access the data from Meetnet Vlaamsebanken you need to register. 
Please add your username and password as environment variables as username_bankenand password_banken. You can register
at https://api.meetnetvlaamsebanken.be/ (version 2)
2. Prepare the data in one single DataFrame for the process. The output will be stored in data/processed. There will be
a df_complete.pkl, a df_no_artifacts.pkl, umap.pkl, umap_clean.pkl and used_freqticks.npy
3. Dimension reduction, categorization and interpretable machine learning. Output will be saved  in form of images, or 
plot inline in the jupyter notebook. All the images and files will be saved under the output/ folder. The models
are saved as joblib files. Some plots are also shown in the jupyter notebook. 