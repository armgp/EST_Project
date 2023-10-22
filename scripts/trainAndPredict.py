import os
import sys
import pandas
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from matplotlib.pylab import *
from osgeo import osr, gdal
from osgeo.gdalnumeric import ravel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
import rasterio

# Define functions for extracting spectral values and preparing training data
def extract_spectral_values(dataset, bandNumber):
    # read band/array from Geotiff/JPEG dataset ... return as 1D array
    raster = dataset.GetRasterBand(bandNumber+1).ReadAsArray().flatten()
    return raster

def PrepareTrainingDataFromCSV(training_csv):
    training_dataframe = pandas.read_csv(training_csv, header=0)
    training_dataframe = training_dataframe.reindex(
      np.random.permutation(training_dataframe.index)
    )

    training_spectral_pixel_values_dataframe = training_dataframe.iloc[:,0:23]
    tree_label_dataframe = training_dataframe['VEG']
    return ( training_spectral_pixel_values_dataframe, tree_label_dataframe)

def BuildRandomForestModel(numberOfTrees, spectral_pixel_values_dataframe, treeNontree_dataframe):
    classifier_random_forest = ExtraTreesClassifier(
      n_estimators=numberOfTrees,
      max_depth=None,
      min_samples_split=1.0,
      random_state=0
    )

    classifier_random_forest_fit = classifier_random_forest.fit(
      spectral_pixel_values_dataframe, treeNontree_dataframe )
    
    return classifier_random_forest_fit

# Define a function for reading pixel data into the Random Forest model
def ReadPixelDataIntoRandomForestModel(imageryDict):
    # create empty pandas dataframe to hold spectral pixel value columns
    # and variable names
    fullVariablesDataFrame = pandas.DataFrame()

    # iterate through all Geotiff/JPEG image
    # filenames in our dictionary{}, get corresponding variable name
    # that would correspond to that in the CSV, write to dataframe.
    for each_file in imageryDict.values():
      if each_file.endswith('ndvi.tif'): variable_names = ['NDVI'] # correspond to headers in training data file
      elif each_file.endswith('_B04_10m.jp2'): variable_names = ['R']
      elif each_file.endswith('_B03_10m.jp2'): variable_names = ['G']
      elif each_file.endswith('_B02_10m.jp2'): variable_names = ['B']
      elif each_file.endswith('_B08_10m.jp2'): variable_names = ['NIR']
      elif each_file.endswith('SAVI1.tif'): variable_names = ['SAVI01']
      elif each_file.endswith('SAVI2.tif'): variable_names = ['SAVI02']
      elif each_file.endswith('SAVI3.tif'): variable_names = ['SAVI03']
      elif each_file.endswith('SAVI4.tif'): variable_names = ['SAVI04']
      elif each_file.endswith('SAVI5.tif'): variable_names = ['SAVI05']
      elif each_file.endswith('SAVI6.tif'): variable_names = ['SAVI06']
      elif each_file.endswith('SAVI7.tif'): variable_names = ['SAVI07']
      elif each_file.endswith('SAVI8.tif'): variable_names = ['SAVI08']
      elif each_file.endswith('SAVI9.tif'): variable_names = ['SAVI09']
      elif each_file.endswith('SAVI10.tif'): variable_names = ['SAVI10']
      elif each_file.endswith('blue_blurred.tif'): variable_names = ['Background_Blue']
      elif each_file.endswith('green_blurred.tif'): variable_names = ['Background_Green']
      elif each_file.endswith('ndvi_blurred.tif'): variable_names = ['Background_NDVI']
      elif each_file.endswith('nir_blurred.tif'): variable_names = ['Background_NIR']
      elif each_file.endswith('panchromatic_blurred.tif'): variable_names = ['Background_Pan']
      elif each_file.endswith('red_blurred.tif'): variable_names = ['Background_Red']
      elif each_file.endswith('pan.tif') and 'Background' not in each_file: variable_names = ['Pan']
      else: continue

      # open current Geotiff/JPEG in imagery dataset
      rasterImageDataset = gdal.Open(each_file)

      # get pixel-level spectral values from each input band
      for band,varname in enumerate(variable_names):

        # get pixel values as 1D array from Geotiff/JPEG file for current band
        spectral_pixel_values = extract_spectral_values(rasterImageDataset,band)

        # assign data to output dataframe, with "varname" i.e. SAVI01,NDVI,Blue,Pan,...
        # and a 1D array of all pixel values from image file
        fullVariablesDataFrame[varname] = spectral_pixel_values

      # release resources for the raster dataset
      rasterImageDataset = None
      del rasterImageDataset

    # return variable names Dataframe
    return fullVariablesDataFrame

# # Define a function for writing forest classification results
# def WriteForestClassification(fullVariablesDataFrame, classificationFitRandomForest, imageryDict):
#     # drop/remove any columns from dataframe
#     # that have NoData (NaNs, or np.nan)
#     fullVariablesDataFrame = fullVariablesDataFrame.dropna(axis=1)

#     # pass-in dataframe containing pixel values from imagery
#     # into predict() method from ExtraTreesClassifier
#     classifierPredictRandomForest = classificationFitRandomForest.predict(fullVariablesDataFrame)

#     # open up panchromatic image Geotiff file, read
#     # its projection,geotransform, and array dimensions (for reference)
#     referenceDataset = gdal.Open( imageryDict['pan'] )
#     out_geotransform = referenceDataset.GetGeoTransform()
#     out_projection = referenceDataset.GetProjection()
#     out_nrows = referenceDataset.RasterYSize
#     out_ncols = referenceDataset.RasterXSize

#     # create 2D array of 1s and 0s containing final classification
#     finalClassification = np.reshape(classifierPredictRandomForest,(out_nrows,out_ncols))

#     # begin to write out Geotiff to hold final classification mask
#     driverTiff = gdal.GetDriverByName('GTiff')
#     outname   = 'vegetation_classified.tif'
#     if os.path.isfile(outname): os.remove(outname)

#     # create output GDAL Geotiff driver
#     # for writing Geotiff
#     vegClassDataset = driverTiff.Create(
#       outname,
#       out_ncols,
#       out_nrows,
#       1,
#       gdal.GDT_Byte
#     )

#     # set output projection &amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp; geotransform to hold
#     # final classification Geotiff image of 1s and 0s
#     vegClassDataset.SetGeoTransform(out_geotransform)
#     vegClassDataset.SetProjection(out_projection)
#     vegBand = vegClassDataset.GetRasterBand(1)
#     vegBand.WriteArray(finalClassification)
#     vegBand.FlushCache()
#     del vegClassDataset
#     vegData=None

#     # # save the output classification mask, consisting
#     # # of 1s and 0s, as a PNG
#     plt.title('')
#     matshow(finalClassification)
#     #colorbar(orientation='horizontal')
#     plt.savefig('test_classes_NEWCODE.png',dpi=150,bbox_inches='tight')
#     plt.close()


    # Assuming finalClassification is your binary mask with 1s and 0s
    # Specify the file path for the output JP2 file
    # output_jp2_path = '/home/gp/Documents/Dev/EST_Project/scripts/output_mask.jp2'

    # # Write the binary mask to a JP2 file using rasterio
    # with rasterio.open(
    #     output_jp2_path,
    #     'w',
    #     driver='JP2OpenJPEG',  # Specify the driver for JP2 format
    #     width=finalClassification.shape[1],  # Specify the width of the mask
    #     height=finalClassification.shape[0],  # Specify the height of the mask
    #     count=1,  # Specify the number of bands (1 for binary mask)
    #     dtype=rasterio.uint8,  # Specify the data type of the mask (8-bit unsigned integer)
    # ) as dst:
    #     dst.write(finalClassification.astype(rasterio.uint8), 1)  # Write the binary mask to the JP2 file

    # print(f'Classification mask saved as {output_jp2_path}')

def WriteForestClassification(fullVariablesDataFrame, classificationFitRandomForest, imageryDict):
    # drop/remove any columns from dataframe
    # that have NoData (NaNs, or np.nan)
    print('on')
    fullVariablesDataFrame = fullVariablesDataFrame.dropna(axis=1)

    # pass-in dataframe containing pixel values from imagery
    # into predict() method from RandomForestClassifier
    print('start')
    classifierPredictRandomForest = classificationFitRandomForest.predict(fullVariablesDataFrame)

    # open up panchromatic image Geotiff file, read
    # its projection, geotransform, and array dimensions (for reference)
    print('hey3')
    with rasterio.open(imageryDict['pan']) as referenceDataset:
        print('hey4')
        out_transform = referenceDataset.transform
        out_height = referenceDataset.height
        out_width = referenceDataset.width

        print('hey1')
        # create 2D array of 1s and 0s containing final classification
        finalClassification = np.reshape(classifierPredictRandomForest, (out_height, out_width))
        print('hey')
        # # Write the binary mask to a JP2 file using rasterio
        # with rasterio.open('./out.jp2', 'w', driver='JP2OpenJPEG', 
        #                    height=finalClassification.shape[0], width=finalClassification.shape[1], 
        #                    count=1, dtype=rasterio.uint8, transform=out_transform) as dst:
        #     dst.write(finalClassification.astype(rasterio.uint8), 1)

        # print(f'Classification mask saved as {output_path}')

def main(numberTrees):
    inputImagery = {
        'pan': '/home/gp/Documents/Dev/EST_Project/QGIS_files/S2B_MSIL2A_20231007T072759_N0509_R049_T36LYH_20231007T101850.SAFE/GRANULE/L2A_T36LYH_A034395_20231007T075214/IMG_DATA/R10m/pan.tif', #
        'ndvi': '/home/gp/Documents/Dev/EST_Project/QGIS_files/S2B_MSIL2A_20231007T072759_N0509_R049_T36LYH_20231007T101850.SAFE/GRANULE/L2A_T36LYH_A034395_20231007T075214/IMG_DATA/R10m/ndvi.tif', #
        # 'rgb': 'T34SDH_20180830T093029_RGB.jp2',
        'r': '/home/gp/Documents/Dev/EST_Project/QGIS_files/S2B_MSIL2A_20231007T072759_N0509_R049_T36LYH_20231007T101850.SAFE/GRANULE/L2A_T36LYH_A034395_20231007T075214/IMG_DATA/R10m/T36LYH_20231007T072759_B04_10m.jp2',
        'g': '/home/gp/Documents/Dev/EST_Project/QGIS_files/S2B_MSIL2A_20231007T072759_N0509_R049_T36LYH_20231007T101850.SAFE/GRANULE/L2A_T36LYH_A034395_20231007T075214/IMG_DATA/R10m/T36LYH_20231007T072759_B03_10m.jp2',
        'b': '/home/gp/Documents/Dev/EST_Project/QGIS_files/S2B_MSIL2A_20231007T072759_N0509_R049_T36LYH_20231007T101850.SAFE/GRANULE/L2A_T36LYH_A034395_20231007T075214/IMG_DATA/R10m/T36LYH_20231007T072759_B02_10m.jp2',
        'nir': '/home/gp/Documents/Dev/EST_Project/QGIS_files/S2B_MSIL2A_20231007T072759_N0509_R049_T36LYH_20231007T101850.SAFE/GRANULE/L2A_T36LYH_A034395_20231007T075214/IMG_DATA/R10m/T36LYH_20231007T072759_B08_10m.jp2',#
        'bg_blue': '/home/gp/Documents/Dev/EST_Project/QGIS_files/S2B_MSIL2A_20231007T072759_N0509_R049_T36LYH_20231007T101850.SAFE/GRANULE/L2A_T36LYH_A034395_20231007T075214/IMG_DATA/R10m/blue_blurred.tif',
        'bg_green': '/home/gp/Documents/Dev/EST_Project/QGIS_files/S2B_MSIL2A_20231007T072759_N0509_R049_T36LYH_20231007T101850.SAFE/GRANULE/L2A_T36LYH_A034395_20231007T075214/IMG_DATA/R10m/green_blurred.tif',
        'bg_red': '/home/gp/Documents/Dev/EST_Project/QGIS_files/S2B_MSIL2A_20231007T072759_N0509_R049_T36LYH_20231007T101850.SAFE/GRANULE/L2A_T36LYH_A034395_20231007T075214/IMG_DATA/R10m/red_blurred.tif',
        'bg_ndvi': '/home/gp/Documents/Dev/EST_Project/QGIS_files/S2B_MSIL2A_20231007T072759_N0509_R049_T36LYH_20231007T101850.SAFE/GRANULE/L2A_T36LYH_A034395_20231007T075214/IMG_DATA/R10m/ndvi_blurred.tif',
        'bg_nir': '/home/gp/Documents/Dev/EST_Project/QGIS_files/S2B_MSIL2A_20231007T072759_N0509_R049_T36LYH_20231007T101850.SAFE/GRANULE/L2A_T36LYH_A034395_20231007T075214/IMG_DATA/R10m/nir_blurred.tif',
        'bg_pan': '/home/gp/Documents/Dev/EST_Project/QGIS_files/S2B_MSIL2A_20231007T072759_N0509_R049_T36LYH_20231007T101850.SAFE/GRANULE/L2A_T36LYH_A034395_20231007T075214/IMG_DATA/R10m/panchromatic_blurred.tif',
        'savi01': '/home/gp/Documents/Dev/EST_Project/QGIS_files/S2B_MSIL2A_20231007T072759_N0509_R049_T36LYH_20231007T101850.SAFE/GRANULE/L2A_T36LYH_A034395_20231007T075214/IMG_DATA/R10m/SAVI1.tif',#
        'savi02': '/home/gp/Documents/Dev/EST_Project/QGIS_files/S2B_MSIL2A_20231007T072759_N0509_R049_T36LYH_20231007T101850.SAFE/GRANULE/L2A_T36LYH_A034395_20231007T075214/IMG_DATA/R10m/SAVI2.tif',#
        'savi03': '/home/gp/Documents/Dev/EST_Project/QGIS_files/S2B_MSIL2A_20231007T072759_N0509_R049_T36LYH_20231007T101850.SAFE/GRANULE/L2A_T36LYH_A034395_20231007T075214/IMG_DATA/R10m/SAVI3.tif',#
        'savi04': '/home/gp/Documents/Dev/EST_Project/QGIS_files/S2B_MSIL2A_20231007T072759_N0509_R049_T36LYH_20231007T101850.SAFE/GRANULE/L2A_T36LYH_A034395_20231007T075214/IMG_DATA/R10m/SAVI4.tif',#
        'savi05': '/home/gp/Documents/Dev/EST_Project/QGIS_files/S2B_MSIL2A_20231007T072759_N0509_R049_T36LYH_20231007T101850.SAFE/GRANULE/L2A_T36LYH_A034395_20231007T075214/IMG_DATA/R10m/SAVI5.tif',#
        'savi06': '/home/gp/Documents/Dev/EST_Project/QGIS_files/S2B_MSIL2A_20231007T072759_N0509_R049_T36LYH_20231007T101850.SAFE/GRANULE/L2A_T36LYH_A034395_20231007T075214/IMG_DATA/R10m/SAVI6.tif',#
        'savi07': '/home/gp/Documents/Dev/EST_Project/QGIS_files/S2B_MSIL2A_20231007T072759_N0509_R049_T36LYH_20231007T101850.SAFE/GRANULE/L2A_T36LYH_A034395_20231007T075214/IMG_DATA/R10m/SAVI7.tif',#
        'savi08': '/home/gp/Documents/Dev/EST_Project/QGIS_files/S2B_MSIL2A_20231007T072759_N0509_R049_T36LYH_20231007T101850.SAFE/GRANULE/L2A_T36LYH_A034395_20231007T075214/IMG_DATA/R10m/SAVI8.tif',#
        'savi09': '/home/gp/Documents/Dev/EST_Project/QGIS_files/S2B_MSIL2A_20231007T072759_N0509_R049_T36LYH_20231007T101850.SAFE/GRANULE/L2A_T36LYH_A034395_20231007T075214/IMG_DATA/R10m/SAVI9.tif',#
        'savi10': '/home/gp/Documents/Dev/EST_Project/QGIS_files/S2B_MSIL2A_20231007T072759_N0509_R049_T36LYH_20231007T101850.SAFE/GRANULE/L2A_T36LYH_A034395_20231007T075214/IMG_DATA/R10m/SAVI10.tif'#
    }

    ioff()
    outputDir = os.getcwd()

    trainingCSV = '/home/gp/Documents/Dev/EST_Project/scripts/train.csv'

    (spectral_pixel_values_df, treeNonTree_df) = PrepareTrainingDataFromCSV(trainingCSV)

    classifierRandomForestFit = BuildRandomForestModel(
        numberTrees, spectral_pixel_values_df, treeNonTree_df
    )

    # read all pixel values from imagery into pandas dataframe
    print('helo1')
    all_vars_dataframe = ReadPixelDataIntoRandomForestModel(inputImagery)
    print('helo')
    WriteForestClassification(all_vars_dataframe, classifierRandomForestFit, inputImagery)


if __name__ == '__main__':
    numberTrees = 10  # Replace with your desired number of trees
    main(numberTrees)
