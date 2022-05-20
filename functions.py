def density_plot(predictions, experimental, header = "title", nbins = 1000):
    
    matplotlib.rcParams["axes.titlepad"] = 25
    matplotlib.rcParams["axes.labelpad"] = 10
    
    df = pd.DataFrame({"predictions" : np.ravel(predictions), "test_targets" : np.ravel(experimental)})

    H, xedges, yedges = np.histogram2d(df["predictions"],df["test_targets"], bins=nbins) # df[x] yerine df["predictions"] kullan, predictions -> x yerine
    H = np.rot90(H)
    H = np.flipud(H)
    Hmasked = np.ma.masked_where(H==0,H) 
    cm = plt.cm.get_cmap('RdYlBu')
    
    plt.figure(figsize = (7,10))    
    plt.pcolormesh(xedges, yedges, Hmasked, cmap = cm, norm = LogNorm(1e0,1e1))
    
    plt.xlabel("iRT (measured)", fontsize = 20)  # part is excluded:"\n Epochs : {}\n{}\n{}".format(len(Histdf), report_str_1,report_str_2)
    plt.ylabel("iRT (predicted)",  fontsize = 20)
    plt.title(header, fontsize = 24)
    
    cbar = plt.colorbar( ticks = LogLocator(subs = range(10)))
    cbar.ax.set_ylabel('Counts', fontsize = 20)
    cbar.ax.tick_params(labelsize = 20)
    # cbar.ax.minorticks_on()
    
    a = np.percentile(np.abs(df["predictions"] - df["test_targets"]), 95)
    b = np.mean(np.abs(df["predictions"] - df["test_targets"]))
    
    plt.text(-20,155,"R-squared = {:.3f}\nn = {:3d}\nDelta 95 = {:.3f}\nMean Error = {:.3f}".format(r2_score(df["test_targets"], df["predictions"]) , len(df["test_targets"]), a, b), 
             fontsize = 14, bbox = dict(facecolor = "m", alpha = 0.3, boxstyle = "round"))

    plt.plot([-5, 160], [-5, 160], ls = "--", linewidth = "2", color = "black", label= "Diagonal") 
    plt.legend(loc=4, prop={"size": 14})
    
    plt.savefig("./Ekin_plots/{} Density Plot.png".format(header), bbox_inches="tight", pad_inches = 1, dpi = 300, transparent= None)


    return 


# take the raw data and filter it to be used later on

def raw_data_filter(file_path, new_folder_name = "data_name"): # for original datasets with rt values 
    
    os.mkdir(new_folder_name) # create a new directory for filtered data 
    
    data = pd.read_csv(file_path, delimiter = "\t", usecols=["Modified sequence", "Retention time","Score", "Reverse"])
    data = data[~data["Modified sequence"].str.contains("(ac)")] # filter the modifications
    data = data[~data["Modified sequence"].str.contains("(ox)")] 
    data = data.reset_index(drop = True)  # reset index
    data["Modified sequence"] = data["Modified sequence"].replace({"_":""}, regex = True) # filter "_" from the sequences 
    data = data.rename(columns = {"Modified sequence": "sequence", "Retention time": "rt", "Score" : "_score","Reverse":"reverse"}) # rename columns
    data = data.sort_values(by = ["_score"]) # sort the data frame by score values
    data = data.reset_index().drop_duplicates(subset = ["sequence"], keep = "last").set_index("index") # drop the duplicates 
    data = data.sort_index()
    filtered_data = data.reset_index(drop = True) # at last, call our data frame "filtered_data" before writing it into a .csv file

    filtered_data.to_csv("./{}/filtered_{}.csv".format(new_folder_name,new_folder_name), index = False)     
    
    return



# Visualize the iRT value distribution and sequence length distribution of the dataset 

def distributionVisualization(dataLocation,dataName,header="title"): 
    
    filePath = os.path.join(dataLocation,dataName)
    
    matplotlib.rcParams["axes.titlepad"] = 10

    data = pd.read_csv(filePath, delimiter = ",")
        
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.hist(data["rt"], bins=50, alpha = 0.7, histtype = "stepfilled", color="orange", edgecolor = "none") # irt değil normalde 
    ax1.set(xlabel="rt Values",ylabel="Freq")
    ax1.set_title("rt Freq Histogram of\n {}".format(header))
    
    sequences = data["sequence"]
    s = sequences.str.len()
    
    ax2.hist(s, bins = np.arange(min(s),max(s)+1), alpha = 0.6, histtype = "stepfilled", color="green", edgecolor = "none") 
    ax2.set(xlabel = "Sequence Lengths",ylabel="Freq")
    ax2.set_title("Sequence Lenght Freq\n Histogram of {}".format(header))
    
    fig.tight_layout()
    # fig.savefig("{}/{} Distribution Plots.png".format(dataLocation, header), dpi = 500)
    
    return



# this data preparation should be done with 0.01 FDR data of target project and original prosit data  
# regressionFile2Location -> should be prosit original data (raw)

def indexingDataPreparation(regressionFile1Location, regressionFile2Location):
    
    regression1DataFrame = pd.read_csv(regressionFile1Location) # filtered target project data (like dongxue, yangyang) 
    
    regression2DataFrame = pd.read_csv(regressionFile2Location, delimiter = ",", usecols=["mod_sequence", "irt","score"]) 
    regression2DataFrame = regression2DataFrame.rename(columns = {"mod_sequence": "sequence"})
    regression2DataFrame = regression2DataFrame[~regression2DataFrame["sequence"].str.contains("-OxM-")]
    
    referenceDataFrame = pd.merge(regression1DataFrame, regression2DataFrame, how="inner", on=["sequence"])
    referenceDataFrame = referenceDataFrame.sort_values(by = ["score"])
    referenceDataFrame = referenceDataFrame.reset_index().drop_duplicates(subset = ["sequence"], keep = "last").set_index("index")
    referenceDataFrame = referenceDataFrame.sort_index()
    referenceDataFrame = referenceDataFrame.reset_index(drop = True) 
    
    return referenceDataFrame



def build_regression_model(reference_df):

    # data preparation for regression
    rt_array = reference_df["rt"].to_numpy()
    irt_array = reference_df["irt"].to_numpy()
    rt_train = rt_array[:-100]
    irt_train = irt_array[:-100]
    
    # test arrays can also be used here in order to see the regression status     
    regr = linear_model.LinearRegression()
    regr.fit(rt_train.reshape(-1, 1), irt_train)

    return regr


def indexandsplit(data_to_index_location, regression_model, irt_data_name="IndexedDataName"):

    # data_to_index -> file name to be indexed, should be present in the upper_directory
    # irt_data_name -> The name that the indexed files (original, holdout, training) will have 
    # regression_model -> linear regression model created with the build_regression_model() function beforehand
    
    indexed_data_folder = irt_data_name
    os.mkdir(indexed_data_folder) # create a new directory for indexed data

    data_to_be_indexed = pd.read_csv(data_to_index_location) 
    rt_to_be_indexed = data_to_be_indexed["rt"].to_numpy()
    
    indexed_rt = regression_model.predict(rt_to_be_indexed.reshape(-1, 1))
    indexed_rt_series = pd.Series(indexed_rt)
    
    sequence_series = data_to_be_indexed["sequence"]
    
    irt_data_frame = pd.concat([sequence_series, indexed_rt_series], axis=1)
    irt_data_frame = irt_data_frame.rename(columns = {0: "irt"})
    
    irt_data_frame.to_csv("./{}/{}.csv".format(indexed_data_folder,irt_data_name),  index = False) # save the whole indexed full FDR Data 
   
    data_train, data_test = train_test_split(irt_data_frame, train_size = 0.8, test_size = 0.2)

    data_train.to_csv("./{}/{}_train.csv".format(indexed_data_folder,irt_data_name), index = False) # save the indexed training Data 
    data_test.to_csv("./{}/{}_holdout.csv".format(indexed_data_folder,irt_data_name), index = False) # save the indexed holdout Data 
    
    return 
    
    
 
    
def modelRefine(refinementData, weights_path, new_weights_file_name="new_refinement", learning_rate = 0.00001,
                SEQ_LENGTH=30, epoch_number=5): 
    # refinementData must be a RetentionTimeDataset object
    
    loss_format = "{val_loss:.5f}"
    epoch_format = "{epoch:02d}"
    weights_file = "{}/weight_{}_{}".format(
        new_weights_file_name, epoch_format, loss_format
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(weights_file,monitor='val_loss', 
                                                    save_best_only=True, save_weights_only=True, mode='min')
    decay = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, verbose=1, min_lr=0)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=10)
    callbacks = [checkpoint, early_stop, decay]
    
    model = PrositRetentionTimePredictor(seq_length = SEQ_LENGTH)
    
    model.load_weights(weights_path)
    
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate, decay=1e-7)
    
    model.compile(optimizer=optimizer, 
              loss="mean_squared_error",
              metrics=["mean_absolute_error", TimeDeltaMetric(refinementData.data_mean, refinementData.data_std)]) 
    
    history = model.fit(refinementData.train_data, validation_data=refinementData.val_data, 
                                       epochs=epoch_number, callbacks=callbacks)
    
    # save the history as a data frame in a .csv file for further use
    pd.DataFrame.from_dict(history.history).to_csv("{}_historyDF".format(new_weights_file_name),index=False)
    
    histDF = pd.read_csv("{}_historyDF".format(new_weights_file_name))
    add_to_report = histDF.iloc[[-1]]
    first_three = add_to_report[["loss", "mean_absolute_error","timedelta"]]
    last_three = add_to_report[["val_loss", "val_mean_absolute_error","val_timedelta"]]
    report_str_1 = first_three.to_string(index = False)
    report_str_2 = last_three.to_string(index = False)
    print("\n\nEpochs:",len(histDF),"\n{}\n{}".format(report_str_1,report_str_2))
    
    return



def modelPredict(weights, test_data, denormalize_data): 
    # weights should be the path for the last checkpoint of the pretrained model
    # test_data should be a callable RetentionTimeDataset object 
    # denormalize_data should be a callable RetentionTimeDataset object, 
        # to which the test_data will be denormalized accordinly 
    
    model = PrositRetentionTimePredictor(seq_length = 30)
    
    model.load_weights(weights)
    
    model_predictions = model.predict(test_data.test_data)
    
    model_predictions = model_predictions.ravel()
    
    model_predictions = denormalize_data.denormalize_targets(model_predictions)
    
    test_targets = test_data.get_split_targets(split="test") # test_targets shouldn't be acquired like this
    
    return model_predictions,test_targets



def predictionDataFrame(predictions, test_targets, test_rtdata, modelName="Name of the model"):
    # the test_rtdata must be a dlpro RetentionTimeDataset object
    
    sequences = test_rtdata.sequences
    sequences = pd.Series(sequences.ravel())

    ser1 = pd.Series(predictions)
    ser2 = pd.Series(test_targets)
    
    modelDF = pd.concat([sequences, ser1, ser2], axis=1)
    modelDF = modelDF.rename(columns = {0:"sequence", 1:"iRT (model_pred)", 2:"iRT (exp)" })
    
    modelDF["abs_delta_irt"] = abs(modelDF["iRT (model_pred)"] - modelDF["iRT (exp)"])
    
    modelDF.to_csv("./SVM_DataFrames/{}ModelPredictionDF.csv".format(modelName), index = False)
    
    return


def svmFormatter(TestDataLocation, modelDFLocation, header = "modelName"):
    
    # TestDataLocation -> the filtered and indexed data (with raw_data_filter function) which will be used to get the experimental irt values and scores of the predicted peptides. For example, if the predicted values in the modelDFLocation are belong to 1.0 FDR Dongxue file, the filteredTestDataLocation should take us into the dongxue_1.0_FDR folder and filtered data inside
    # modelDFLocation -> the location of the Model_Predictions file acquired from dlpro framework (prediction files created under the folder "SVM_DataFrames" by the Dongxue_Full_FDR_Script.ipynb)
    
    modelDF = pd.read_csv(modelDFLocation, delimiter = ",")
    originalData = pd.read_csv(originalDataLocation, delimiter = ",")
    
    modelCombine = pd.merge(originalData, modelDF, how="inner", on=["sequence"])
    modelCombine = modelCombine[["sequence","abs_delta_irt","_score","reverse"]]
    modelCombine = modelCombine.rename(columns = {"_score" : "score"})
    scanr = np.arange(len(modelCombine))
    modelCombine["scannr"] = scanr
    modelCombine["Peptide"] = "2"
    modelCombine["Proteins"] = "3"
    modelCombine = modelCombine[["sequence","reverse","scannr","score","abs_delta_irt","Peptide","Proteins"]]
    modelCombine["reverse"] = modelCombine["reverse"].replace({"+":"-1"}) 
    modelCombine["reverse"] = modelCombine["reverse"].fillna(1)
    modelCombine = modelCombine.reset_index(drop = True) 
    modelCombine = modelCombine.rename(columns = {"sequence" : "specid","reverse": "Label","scannr":"ScanNr","score":"feature1name","abs_delta_irt":"feature2name"})
    modelCombine.to_csv("{}ModelSVM.pin".format(header),index=False,sep ="\t")

    return

