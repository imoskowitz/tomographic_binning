from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import os
from pandas import read_csv
from glob import glob
from tqdm import tqdm
from astropy.io import fits
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import model_from_yaml
#from keras.callbacks.callbacks import EarlyStopping
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam # heh
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import Huber




class nn_classifier_individual_bins:

    def __init__(self, bin_num, run_folder = './tpzruns/SpecSpec/', data_name = 'tpzrun', acc_cutoff = 0.07, epochs = 1000, nodecounts = [100, 200, 100, 50, 1], splitnum = 5, activations = ['selu', 'selu', 'selu', 'selu', 'sigmoid'], phot_feature_list = ['i', 'gr', 'ri', 'iz', 'zy', 'gri', 'riz', 'izy'], include_tpz_results = True, correct_pz = False):

        self.nodecounts = nodecounts
        self.activations = activations
        self.epochs = epochs
        self.splitnum = splitnum
        self.include_tpz_results = include_tpz_results
        self.correct_pz = correct_pz
        self.run_folder = run_folder
        self.data_name = data_name
        self.acc_cutoff = acc_cutoff


        if not self.include_tpz_results:
            tpzflag = '_notpz'
        else:
            tpzflag = ''
        if self.correct_pz:
            corr_flag = '_corr'
        else:
            corr_flag = ''

        self.savefolder = self.run_folder + 'nnc_epoch%i_%i%s%s/' % (self.epochs, bin_num, tpzflag, corr_flag)

        self.get_data(phot_feature_list, bin_num)
        self.get_models()



    def get_models(self):

        if os.path.isdir(self.savefolder) and len(glob(self.savefolder + '*.yaml')) == self.splitnum and len(glob(self.savefolder + '*.h5')) == self.splitnum:

            self.nnlist = []

            transformfile = self.savefolder + 'NNC.transform'
            self.feature_avgs, self.feature_vars = np.loadtxt(transformfile)

            for nn_name in ['nn_%02i'%x for x in range(self.splitnum)]:

                thisyamlfile = self.savefolder + nn_name + '.yaml'
                thish5file = self.savefolder + nn_name + '.h5'

                readyaml = open(thisyamlfile, 'r')
                loaded_model_yaml = readyaml.read()
                readyaml.close()
                loaded_model = model_from_yaml(loaded_model_yaml)
                # load weights into new model
                loaded_model.load_weights(thish5file)
                loaded_model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

                self.nnlist.append(loaded_model)


        else:
            if not os.path.isdir(self.savefolder):
                os.makedirs(self.savefolder)

            # Create models

            self.create_models()

            # Fit Models

            # self.train, self.val = self.split_sample(len(self.features))
            self.feature_avgs = np.average(self.features_train, axis = 0)
            self.feature_vars = np.var(self.features_train, axis = 0)

            fig = plt.figure(figsize = (8,8))
            sp = fig.add_subplot(111)
            
            for x, thismodel in enumerate(self.nnlist):

                x_train = self.preprocess(self.features_train)
                y_train = self.is_goodfit_train.reshape(-1,1)
                x_val = self.preprocess(self.features_val)
                y_val = self.is_goodfit_val.reshape(-1,1)

                es = EarlyStopping(patience = 25, restore_best_weights = True)
                history = thismodel.fit(x_train, y_train, batch_size = 1000, epochs = self.epochs, verbose = 2, validation_data = (x_val, y_val), callbacks = [es])

                # history = thismodel.fit(x_train, y_train, batch_size = 1000, epochs = self.epochs, verbose = 2, validation_data = (x_val, y_val))
                this_fit_val = np.squeeze(thismodel.predict(self.preprocess(self.features_val), verbose = 0))

                sp.plot(history.history['loss'], label = 'Training')
                sp.plot(history.history['val_loss'], label = 'Validation')
                sp.set_ylabel('Loss')
                sp.set_xlabel('Epoch')
                sp.legend(loc = 'upper right')
                plt.savefig(self.savefolder + 'loss_%02i.png' % x, bbox_inches = 'tight')
                sp.cla()

                sp.plot(history.history['accuracy'], label = 'Training')
                sp.plot(history.history['val_accuracy'], label = 'Validation')
                sp.set_ylabel('Accuracy')
                sp.set_xlabel('Epoch')
                sp.legend(loc = 'upper left')
                plt.savefig(self.savefolder + 'acc_%02i.png' % x, bbox_inches = 'tight')
                sp.cla()

                predictions = np.squeeze(thismodel.predict(self.preprocess(self.features_val)))
                binary_predictions = (predictions>.5).astype(int)
                true_pos = np.sum((binary_predictions == self.is_goodfit_val) & (binary_predictions == 1))
                false_pos = np.sum((binary_predictions != self.is_goodfit_val) & (binary_predictions == 1))
                true_neg = np.sum((binary_predictions == self.is_goodfit_val) & (binary_predictions == 0))
                false_neg = np.sum((binary_predictions != self.is_goodfit_val) & (binary_predictions == 0))

                sp.plot([0.5, 0.5], [0,1], color = 'k')
                sp.plot([0,1], [0.5, 0.5], color = 'k')
                sp.text(.25,.25, 'FPR = %.5f' % (false_pos/float(true_neg + false_pos)), ha = 'center', va = 'center', fontsize = 30)
                sp.text(.25,.75, 'TNR = %.5f' % (true_neg/float(true_neg + false_pos)), ha = 'center', va = 'center', fontsize = 30)
                sp.text(.75,.25, 'TPR = %.5f' % (true_pos/float(false_neg + true_pos)), ha = 'center', va = 'center', fontsize = 30)
                sp.text(.75,.75, 'FNR = %.5f' % (false_neg/float(false_neg + true_pos)), ha = 'center', va = 'center', fontsize = 30)
                sp.text(0.75, 0.5, 'Acc = %.3f' % ((true_pos + true_neg) / float(len(self.is_goodfit_val))), ha = 'center', va = 'center', fontsize = 20, color = 'r', bbox = dict(facecolor = 'white', ec = 'k'))
                sp.text(0.25, 0.5, 'Misclass = %.3f' % ((false_neg + false_pos) / float(len(self.is_goodfit_val))), ha = 'center', va = 'center', fontsize = 20, color = 'r', bbox = dict(facecolor = 'white', ec = 'k'))
                sp.set_xlim(0,1)
                sp.set_ylim(0,1)
                sp.set_xticklabels([])
                sp.set_yticklabels([])
                sp.text(0.5, 1.05, 'True', ha = 'center', va = 'bottom', fontsize = 20)
                sp.text(-0.05, 0.5, 'Predicted', ha = 'right', va = 'center', rotation = 'vertical', fontsize = 20)
                sp.text(-0.02, 0.25, 'Good Fit', ha = 'right', va = 'center', rotation = 'vertical', fontsize = 20)
                sp.text(-0.02, 0.75, 'Bad Fit', ha = 'right', va = 'center', rotation = 'vertical', fontsize = 20)
                sp.text(0.25, 1.02, 'Bad Fit', ha = 'center', va = 'bottom', fontsize = 20)
                sp.text(0.75, 1.02, 'Good Fit', ha = 'center', va = 'bottom', fontsize = 20)
                plt.savefig(self.savefolder + 'confusion_%02i.png' % x, bbox_inches = 'tight')
                sp.cla()

                
                sorted_predictions, sorted_is_goodfit = np.array(list(zip(*sorted(zip(predictions, self.is_goodfit_val)))))
                sorted_is_goodfit = sorted_is_goodfit.astype(bool)

                tpr = (np.cumsum(sorted_is_goodfit[::-1])/float(np.sum(sorted_is_goodfit)))[::-1]
                fpr = (np.cumsum(~sorted_is_goodfit[::-1])/float(np.sum(~sorted_is_goodfit)))[::-1]
                auc = -np.trapz(tpr, fpr)

                sp.plot(fpr, tpr)
                sp.text(0.98, 0.02, '$AUC = %.2f$' % auc, fontsize = 20, ha = 'right', va = 'bottom')
                sp.set_xlabel('False Positive Rate')
                sp.set_ylabel('True Positive Rate')
                sp.set_xlim(0,1)
                sp.set_ylim(0,1)
                for cutoff in np.arange(0.1,1,.1):
                    thisx = np.interp(cutoff, sorted_predictions, fpr)
                    thisy = np.interp(cutoff, sorted_predictions, tpr)
                    sp.annotate('%.1f\n(%.2f, %.2f)'%(cutoff, thisx, thisy), (thisx, thisy), (thisx+.05, thisy- 0.05), arrowprops = dict(arrowstyle = '-', ec = 'k'))
                plt.savefig(self.savefolder + 'roc_%02i.png' % x, bbox_inches = 'tight')

                sp.cla()


                precision = (np.cumsum(sorted_is_goodfit[::-1])/(np.arange(len(sorted_is_goodfit[::-1]), dtype = float) + 1))[::-1]
                recall = tpr

                # Delete Later if nothing breaks
                # for y in tqdm(range(len(sorted_is_goodfit))):

                #     predict_true = sorted_is_goodfit[y:]

                #     precision.append(np.sum(predict_true)/float(len(predict_true)))
                #     # recall.append(np.sum(predict_true)/float(np.sum(sorted_is_goodfit)))

                # precision = np.array(precision)

                auc = -np.trapz(precision, recall)
                sp.text(0.98, 0.02, '$AUC = %.2f$' % auc, fontsize = 20, ha = 'right', va = 'bottom')
                sp.plot(recall, precision)
                sp.set_xlabel('Recall')
                sp.set_ylabel('Precision')
                sp.set_xlim(0,1)
                sp.set_ylim(0,1)
                for cutoff in np.arange(0.1,1,.1):
                    thisx = np.interp(cutoff, sorted_predictions, recall)
                    thisy = np.interp(cutoff, sorted_predictions, precision)
                    sp.annotate('%.1f\n(%.2f, %.2f)'%(cutoff, thisx, thisy), (thisx, thisy), (thisx-.05, thisy- 0.05), arrowprops = dict(arrowstyle = '-', ec = 'k'))
                try:
                    plt.savefig(self.savefolder + 'precision_recall_%02i.png' % x, bbox_inches = 'tight')
                except:
                    print('three')
                sp.cla()

            plt.close()

            # Write Models

            if not os.path.isdir(self.savefolder):
                os.makedirs(self.savefolder)

            for nn_number, thismodel in enumerate(self.nnlist):

                nn_name = 'nn_%02i' % nn_number

                model_yaml = thismodel.to_yaml()
                thisyamlfile = self.savefolder + nn_name + '.yaml'
                thish5file = self.savefolder + nn_name + '.h5'
                thistransformfile = self.savefolder + nn_name + '.transform'

                with open(thisyamlfile, "w") as yamlwrite:
                    yamlwrite.write(model_yaml)
                # serialize weights to HDF5
                thismodel.save_weights(thish5file)


            transformfile = self.savefolder + 'NNC.transform'
            np.savetxt(transformfile, np.vstack((self.feature_avgs, self.feature_vars)))



            savekeys = ['acc_cutoff', 'feature_names', 'nodecounts', 'activations', 'epochs', 'run_folder']

            with open(self.savefolder + 'info.txt', 'w') as writefile:
                for thiskey in savekeys:
                    writefile.write(thiskey + ': ' + repr(getattr(self, thiskey)) + '\n')


            np.savetxt(self.savefolder + 'results_train.dat', np.average(self.fit_features(self.features_train), axis = 1))
            np.savetxt(self.savefolder + 'results_validate.dat', np.average(self.fit_features(self.features_val), axis = 1))

            stub = ''.join(('r_epoch'.join(self.savefolder.split('c_epoch'))).split('_corr'))

            if os.path.isfile(self.run_folder + self.data_name + '.nnc_test') and os.path.isfile(stub + 'results_test.dat'):
                np.savetxt(self.savefolder + 'results_test.dat', self.fit_file(self.run_folder + self.data_name + '.nnc_test', stub + 'results_test.dat', self.run_folder + self.data_name + '.nnc_test_inds'))







    def get_data(self, phot_feature_list, bin_num):

        keep_feature_list = phot_feature_list
        if self.include_tpz_results:
            keep_feature_list = keep_feature_list + ['zphot', 'zconf', 'zerr']

        trainfile = read_csv(self.run_folder + self.data_name  + '.nnc_train_bin_{}'.format(bin_num), delimiter = '\s+', comment = '#')
        valfile = read_csv(self.run_folder + self.data_name  + '.nnc_validate_bin_{}'.format(bin_num), delimiter = '\s+', comment = '#')
        self.features_train = trainfile[keep_feature_list].values
        self.features_val = valfile[keep_feature_list].values

        self.feature_names = np.array(keep_feature_list)

        pz_train = np.squeeze(self.features_train.T[self.feature_names == 'zphot'])
        sz_train = trainfile['specz'].to_numpy()
        pz_err_train = np.squeeze(self.features_train.T[self.feature_names == 'zerr'])
        misclass_train = trainfile['misclass'].to_numpy()

        pz_val = np.squeeze(self.features_val.T[self.feature_names == 'zphot'])
        sz_val = valfile['specz'].to_numpy()
        pz_err_val = np.squeeze(self.features_val.T[self.feature_names == 'zerr'])
        misclass_val = valfile['misclass'].to_numpy()

        train_trans_inds = np.loadtxt(self.run_folder + self.data_name + '.nnc_train_inds', unpack = True, dtype = int)[1]
        val_trans_inds = np.loadtxt(self.run_folder + self.data_name + '.nnc_validate_inds', unpack = True, dtype = int)[1]

        if self.correct_pz:
            # Add corrected photometric redshifts to features
            stub = ''.join(('r_epoch'.join(self.savefolder.split('c_epoch'))).split('_corr'))
            regressed_err_test = np.loadtxt(stub + 'results_test.dat')

            self.feature_names = np.append(self.feature_names, ['zphot_NNR'])
            self.features_train = np.hstack((self.features_train, (pz_train - regressed_err_test[train_trans_inds]).reshape(-1,1)))
            self.features_val = np.hstack((self.features_val, (pz_val - regressed_err_test[val_trans_inds]).reshape(-1,1)))

        #self.is_goodfit_train = (np.abs(pz_train - sz_train) < (1. + sz_train)*self.acc_cutoff)
        #self.is_goodfit_val = (np.abs(pz_val - sz_val) < (1. + sz_val)*self.acc_cutoff)
        self.is_goodfit_train = misclass_train.astype(bool)
        self.is_goodfit_val = misclass_val.astype(bool)



    def create_models(self):

        # Initializes untrained neural networks

        self.nnlist = []

        for x in range(self.splitnum):

            model = Sequential()
            model.add(Dense(self.nodecounts[0], activation = self.activations[0], input_shape = (self.features_train.shape[1],)))
            # model.add(Dropout(0.2))

            for thisnodenum, thisactivation in zip(self.nodecounts[1:], self.activations[1:]):

                if thisactivation == 'selu':
                    kernel_initializer = 'lecun_normal'
                else:
                    kernel_initializer = 'glorot_uniform'
                model.add(Dense(thisnodenum, activation = thisactivation, kernel_initializer = kernel_initializer))

            initial_learning_rate = 0.005
            lr_schedule = ExponentialDecay(
                initial_learning_rate,
                decay_steps=3500,
                decay_rate=0.1,
                staircase=False)

            model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate = lr_schedule), metrics = ['accuracy'])

            self.nnlist.append(model)


    def preprocess(self, X):

        return (X - self.feature_avgs)/np.sqrt(self.feature_vars)


    def fit_file(self, fname, nnr_results_fname = None, index_file = None, average_results = True):

        feature_file = read_csv(fname, comment = '#', delimiter = '\s+')
        
        if (not self.correct_pz) and index_file != None:
            translate_inds = np.loadtxt(index_file, unpack = True, dtype = int)[1]
            feature_file['zphot_NNR'] = feature_file['zphot'] - np.loadtxt(nnr_results_fname)[translate_inds]
            features_fit = feature_file[self.feature_names].to_numpy()
        elif self.correct_pz:
            feature_file['zphot_NNR'] = feature_file['zphot'] - np.loadtxt(nnr_results_fname)

        features_fit = feature_file[self.feature_names].to_numpy()

        if average_results:
            return np.average((self.fit_features(features_fit)), axis = 1)
        else:
            return self.fit_features(features_fit)


    def fit_features(self, features):

        processed_features = self.preprocess(features)

        predictions = []

        for thismodel in self.nnlist:

            predictions.append(thismodel.predict(processed_features, verbose = 1))

        return np.squeeze(np.array(predictions).T)


    def fit_app(self, bin_num):

        if self.correct_pz:
            stub = ''.join(('r_epoch'.join(self.savefolder.split('c_epoch'))).split('_corr'))
            np.savetxt(self.savefolder + 'results_application.dat', self.fit_file(self.run_folder + self.data_name + '.nn_app_bin_{}'.format(bin_num), stub + 'results_application.dat'))
        else:
            np.savetxt(self.savefolder + 'results_application.dat', self.fit_file(self.run_folder + self.data_name + '.nn_app_bin_{}'.format(bin_num)))




###NNC that estimates confidence of being sorted into correct bin###






class nn_classifier_misclass:

    def __init__(self, run_folder = './tpzruns/SpecSpec/', data_name = 'tpzrun', acc_cutoff = 0.07, epochs = 1000, nodecounts = [100, 200, 100, 50, 1], splitnum = 5, activations = ['selu', 'selu', 'selu', 'selu', 'sigmoid'], phot_feature_list = ['i', 'gr', 'ri', 'iz', 'zy', 'gri', 'riz', 'izy'], include_tpz_results = True, correct_pz = False):

        self.nodecounts = nodecounts
        self.activations = activations
        self.epochs = epochs
        self.splitnum = splitnum
        self.include_tpz_results = include_tpz_results
        self.correct_pz = correct_pz
        self.run_folder = run_folder
        self.data_name = data_name
        self.acc_cutoff = acc_cutoff


        if not self.include_tpz_results:
            tpzflag = '_notpz'
        else:
            tpzflag = ''
        if self.correct_pz:
            corr_flag = '_corr'
        else:
            corr_flag = ''

        self.savefolder = self.run_folder + 'nnc_epoch%i%s%s/' % (self.epochs, tpzflag, corr_flag)

        self.get_data(phot_feature_list)
        self.get_models()



    def get_models(self):

        if os.path.isdir(self.savefolder) and len(glob(self.savefolder + '*.yaml')) == self.splitnum and len(glob(self.savefolder + '*.h5')) == self.splitnum:

            self.nnlist = []

            transformfile = self.savefolder + 'NNC.transform'
            self.feature_avgs, self.feature_vars = np.loadtxt(transformfile)

            for nn_name in ['nn_%02i'%x for x in range(self.splitnum)]:

                thisyamlfile = self.savefolder + nn_name + '.yaml'
                thish5file = self.savefolder + nn_name + '.h5'

                readyaml = open(thisyamlfile, 'r')
                loaded_model_yaml = readyaml.read()
                readyaml.close()
                loaded_model = model_from_yaml(loaded_model_yaml)
                # load weights into new model
                loaded_model.load_weights(thish5file)
                loaded_model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

                self.nnlist.append(loaded_model)


        else:
            if not os.path.isdir(self.savefolder):
                os.makedirs(self.savefolder)

            # Create models

            self.create_models()

            # Fit Models

            # self.train, self.val = self.split_sample(len(self.features))
            self.feature_avgs = np.average(self.features_train, axis = 0)
            self.feature_vars = np.var(self.features_train, axis = 0)

            fig = plt.figure(figsize = (8,8))
            sp = fig.add_subplot(111)
            
            for x, thismodel in enumerate(self.nnlist):

                x_train = self.preprocess(self.features_train)
                y_train = self.is_goodfit_train.reshape(-1,1)
                x_val = self.preprocess(self.features_val)
                y_val = self.is_goodfit_val.reshape(-1,1)

                es = EarlyStopping(patience = 25, restore_best_weights = True)
                history = thismodel.fit(x_train, y_train, batch_size = 1000, epochs = self.epochs, verbose = 2, validation_data = (x_val, y_val), callbacks = [es])

                # history = thismodel.fit(x_train, y_train, batch_size = 1000, epochs = self.epochs, verbose = 2, validation_data = (x_val, y_val))
                this_fit_val = np.squeeze(thismodel.predict(self.preprocess(self.features_val), verbose = 0))

                sp.plot(history.history['loss'], label = 'Training')
                sp.plot(history.history['val_loss'], label = 'Validation')
                sp.set_ylabel('Loss')
                sp.set_xlabel('Epoch')
                sp.legend(loc = 'upper right')
                plt.savefig(self.savefolder + 'loss_%02i.png' % x, bbox_inches = 'tight')
                sp.cla()

                sp.plot(history.history['accuracy'], label = 'Training')
                sp.plot(history.history['val_accuracy'], label = 'Validation')
                sp.set_ylabel('Accuracy')
                sp.set_xlabel('Epoch')
                sp.legend(loc = 'upper left')
                plt.savefig(self.savefolder + 'acc_%02i.png' % x, bbox_inches = 'tight')
                sp.cla()

                predictions = np.squeeze(thismodel.predict(self.preprocess(self.features_val)))
                binary_predictions = (predictions>.5).astype(int)
                true_pos = np.sum((binary_predictions == self.is_goodfit_val) & (binary_predictions == 1))
                false_pos = np.sum((binary_predictions != self.is_goodfit_val) & (binary_predictions == 1))
                true_neg = np.sum((binary_predictions == self.is_goodfit_val) & (binary_predictions == 0))
                false_neg = np.sum((binary_predictions != self.is_goodfit_val) & (binary_predictions == 0))

                sp.plot([0.5, 0.5], [0,1], color = 'k')
                sp.plot([0,1], [0.5, 0.5], color = 'k')
                sp.text(.25,.25, 'FPR = %.5f' % (false_pos/float(true_neg + false_pos)), ha = 'center', va = 'center', fontsize = 30)
                sp.text(.25,.75, 'TNR = %.5f' % (true_neg/float(true_neg + false_pos)), ha = 'center', va = 'center', fontsize = 30)
                sp.text(.75,.25, 'TPR = %.5f' % (true_pos/float(false_neg + true_pos)), ha = 'center', va = 'center', fontsize = 30)
                sp.text(.75,.75, 'FNR = %.5f' % (false_neg/float(false_neg + true_pos)), ha = 'center', va = 'center', fontsize = 30)
                sp.text(0.75, 0.5, 'Acc = %.3f' % ((true_pos + true_neg) / float(len(self.is_goodfit_val))), ha = 'center', va = 'center', fontsize = 20, color = 'r', bbox = dict(facecolor = 'white', ec = 'k'))
                sp.text(0.25, 0.5, 'Misclass = %.3f' % ((false_neg + false_pos) / float(len(self.is_goodfit_val))), ha = 'center', va = 'center', fontsize = 20, color = 'r', bbox = dict(facecolor = 'white', ec = 'k'))
                sp.set_xlim(0,1)
                sp.set_ylim(0,1)
                sp.set_xticklabels([])
                sp.set_yticklabels([])
                sp.text(0.5, 1.05, 'True', ha = 'center', va = 'bottom', fontsize = 20)
                sp.text(-0.05, 0.5, 'Predicted', ha = 'right', va = 'center', rotation = 'vertical', fontsize = 20)
                sp.text(-0.02, 0.25, 'Good Fit', ha = 'right', va = 'center', rotation = 'vertical', fontsize = 20)
                sp.text(-0.02, 0.75, 'Bad Fit', ha = 'right', va = 'center', rotation = 'vertical', fontsize = 20)
                sp.text(0.25, 1.02, 'Bad Fit', ha = 'center', va = 'bottom', fontsize = 20)
                sp.text(0.75, 1.02, 'Good Fit', ha = 'center', va = 'bottom', fontsize = 20)
                plt.savefig(self.savefolder + 'confusion_%02i.png' % x, bbox_inches = 'tight')
                sp.cla()

                
                sorted_predictions, sorted_is_goodfit = np.array(list(zip(*sorted(zip(predictions, self.is_goodfit_val)))))
                sorted_is_goodfit = sorted_is_goodfit.astype(bool)

                tpr = (np.cumsum(sorted_is_goodfit[::-1])/float(np.sum(sorted_is_goodfit)))[::-1]
                fpr = (np.cumsum(~sorted_is_goodfit[::-1])/float(np.sum(~sorted_is_goodfit)))[::-1]
                auc = -np.trapz(tpr, fpr)

                sp.plot(fpr, tpr)
                sp.text(0.98, 0.02, '$AUC = %.2f$' % auc, fontsize = 20, ha = 'right', va = 'bottom')
                sp.set_xlabel('False Positive Rate')
                sp.set_ylabel('True Positive Rate')
                sp.set_xlim(0,1)
                sp.set_ylim(0,1)
                for cutoff in np.arange(0.1,1,.1):
                    thisx = np.interp(cutoff, sorted_predictions, fpr)
                    thisy = np.interp(cutoff, sorted_predictions, tpr)
                    sp.annotate('%.1f\n(%.2f, %.2f)'%(cutoff, thisx, thisy), (thisx, thisy), (thisx+.05, thisy- 0.05), arrowprops = dict(arrowstyle = '-', ec = 'k'))
                plt.savefig(self.savefolder + 'roc_%02i.png' % x, bbox_inches = 'tight')

                sp.cla()


                precision = (np.cumsum(sorted_is_goodfit[::-1])/(np.arange(len(sorted_is_goodfit[::-1]), dtype = float) + 1))[::-1]
                recall = tpr

                # Delete Later if nothing breaks
                # for y in tqdm(range(len(sorted_is_goodfit))):

                #     predict_true = sorted_is_goodfit[y:]

                #     precision.append(np.sum(predict_true)/float(len(predict_true)))
                #     # recall.append(np.sum(predict_true)/float(np.sum(sorted_is_goodfit)))

                # precision = np.array(precision)

                auc = -np.trapz(precision, recall)
                sp.text(0.98, 0.02, '$AUC = %.2f$' % auc, fontsize = 20, ha = 'right', va = 'bottom')
                sp.plot(recall, precision)
                sp.set_xlabel('Recall')
                sp.set_ylabel('Precision')
                sp.set_xlim(0,1)
                sp.set_ylim(0,1)
                for cutoff in np.arange(0.1,1,.1):
                    thisx = np.interp(cutoff, sorted_predictions, recall)
                    thisy = np.interp(cutoff, sorted_predictions, precision)
                    sp.annotate('%.1f\n(%.2f, %.2f)'%(cutoff, thisx, thisy), (thisx, thisy), (thisx-.05, thisy- 0.05), arrowprops = dict(arrowstyle = '-', ec = 'k'))
                try:
                    plt.savefig(self.savefolder + 'precision_recall_%02i.png' % x, bbox_inches = 'tight')
                except:
                    print('three')
                sp.cla()

            plt.close()

            # Write Models

            if not os.path.isdir(self.savefolder):
                os.makedirs(self.savefolder)

            for nn_number, thismodel in enumerate(self.nnlist):

                nn_name = 'nn_%02i' % nn_number

                model_yaml = thismodel.to_yaml()
                thisyamlfile = self.savefolder + nn_name + '.yaml'
                thish5file = self.savefolder + nn_name + '.h5'
                thistransformfile = self.savefolder + nn_name + '.transform'

                with open(thisyamlfile, "w") as yamlwrite:
                    yamlwrite.write(model_yaml)
                # serialize weights to HDF5
                thismodel.save_weights(thish5file)


            transformfile = self.savefolder + 'NNC.transform'
            np.savetxt(transformfile, np.vstack((self.feature_avgs, self.feature_vars)))



            savekeys = ['acc_cutoff', 'feature_names', 'nodecounts', 'activations', 'epochs', 'run_folder']

            with open(self.savefolder + 'info.txt', 'w') as writefile:
                for thiskey in savekeys:
                    writefile.write(thiskey + ': ' + repr(getattr(self, thiskey)) + '\n')


            np.savetxt(self.savefolder + 'results_train.dat', np.average(self.fit_features(self.features_train), axis = 1))
            np.savetxt(self.savefolder + 'results_validate.dat', np.average(self.fit_features(self.features_val), axis = 1))

            stub = ''.join(('r_epoch'.join(self.savefolder.split('c_epoch'))).split('_corr'))

            if os.path.isfile(self.run_folder + self.data_name + '.nnc_test') and os.path.isfile(stub + 'results_test.dat'):
                np.savetxt(self.savefolder + 'results_test.dat', self.fit_file(self.run_folder + self.data_name + '.nnc_test', stub + 'results_test.dat', self.run_folder + self.data_name + '.nnc_test_inds'))







    def get_data(self, phot_feature_list):

        keep_feature_list = phot_feature_list
        if self.include_tpz_results:
            keep_feature_list = keep_feature_list + ['zphot', 'zconf', 'zerr']

        trainfile = read_csv(self.run_folder + self.data_name  + '.nnc_train', delimiter = '\s+', comment = '#')
        valfile = read_csv(self.run_folder + self.data_name  + '.nnc_validate', delimiter = '\s+', comment = '#')
        self.features_train = trainfile[keep_feature_list].values
        self.features_val = valfile[keep_feature_list].values

        self.feature_names = np.array(keep_feature_list)

        pz_train = np.squeeze(self.features_train.T[self.feature_names == 'zphot'])
        sz_train = trainfile['specz'].to_numpy()
        pz_err_train = np.squeeze(self.features_train.T[self.feature_names == 'zerr'])
        misclass_train = trainfile['misclass'].to_numpy()

        pz_val = np.squeeze(self.features_val.T[self.feature_names == 'zphot'])
        sz_val = valfile['specz'].to_numpy()
        pz_err_val = np.squeeze(self.features_val.T[self.feature_names == 'zerr'])
        misclass_val = valfile['misclass'].to_numpy()

        train_trans_inds = np.loadtxt(self.run_folder + self.data_name + '.nnc_train_inds', unpack = True, dtype = int)[1]
        val_trans_inds = np.loadtxt(self.run_folder + self.data_name + '.nnc_validate_inds', unpack = True, dtype = int)[1]

        if self.correct_pz:
            # Add corrected photometric redshifts to features
            stub = ''.join(('r_epoch'.join(self.savefolder.split('c_epoch'))).split('_corr'))
            regressed_err_test = np.loadtxt(stub + 'results_test.dat')

            self.feature_names = np.append(self.feature_names, ['zphot_NNR'])
            self.features_train = np.hstack((self.features_train, (pz_train - regressed_err_test[train_trans_inds]).reshape(-1,1)))
            self.features_val = np.hstack((self.features_val, (pz_val - regressed_err_test[val_trans_inds]).reshape(-1,1)))

        #self.is_goodfit_train = (np.abs(pz_train - sz_train) < (1. + sz_train)*self.acc_cutoff)
        #self.is_goodfit_val = (np.abs(pz_val - sz_val) < (1. + sz_val)*self.acc_cutoff)
        self.is_goodfit_train = misclass_train.astype(bool)
        self.is_goodfit_val = misclass_val.astype(bool)



    def create_models(self):

        # Initializes untrained neural networks

        self.nnlist = []

        for x in range(self.splitnum):

            model = Sequential()
            model.add(Dense(self.nodecounts[0], activation = self.activations[0], input_shape = (self.features_train.shape[1],)))
            # model.add(Dropout(0.2))

            for thisnodenum, thisactivation in zip(self.nodecounts[1:], self.activations[1:]):

                if thisactivation == 'selu':
                    kernel_initializer = 'lecun_normal'
                else:
                    kernel_initializer = 'glorot_uniform'
                model.add(Dense(thisnodenum, activation = thisactivation, kernel_initializer = kernel_initializer))

            initial_learning_rate = 0.005
            lr_schedule = ExponentialDecay(
                initial_learning_rate,
                decay_steps=3500,
                decay_rate=0.1,
                staircase=False)

            model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate = lr_schedule), metrics = ['accuracy'])

            self.nnlist.append(model)


    def preprocess(self, X):

        return (X - self.feature_avgs)/np.sqrt(self.feature_vars)


    def fit_file(self, fname, nnr_results_fname = None, index_file = None, average_results = True):

        feature_file = read_csv(fname, comment = '#', delimiter = '\s+')
        
        if (not self.correct_pz) and index_file != None:
            translate_inds = np.loadtxt(index_file, unpack = True, dtype = int)[1]
            feature_file['zphot_NNR'] = feature_file['zphot'] - np.loadtxt(nnr_results_fname)[translate_inds]
            features_fit = feature_file[self.feature_names].to_numpy()
        elif self.correct_pz:
            feature_file['zphot_NNR'] = feature_file['zphot'] - np.loadtxt(nnr_results_fname)

        features_fit = feature_file[self.feature_names].to_numpy()

        if average_results:
            return np.average((self.fit_features(features_fit)), axis = 1)
        else:
            return self.fit_features(features_fit)


    def fit_features(self, features):

        processed_features = self.preprocess(features)

        predictions = []

        for thismodel in self.nnlist:

            predictions.append(thismodel.predict(processed_features, verbose = 1))

        return np.squeeze(np.array(predictions).T)


    def fit_app(self):

        if self.correct_pz:
            stub = ''.join(('r_epoch'.join(self.savefolder.split('c_epoch'))).split('_corr'))
            np.savetxt(self.savefolder + 'results_application.dat', self.fit_file(self.run_folder + self.data_name + '.nn_app', stub + 'results_application.dat'))
        else:
            np.savetxt(self.savefolder + 'results_application.dat', self.fit_file(self.run_folder + self.data_name + '.nn_app'))





#=============
# Description
#=============
# This is the OG neural network (NN) object, which requires a .nnc_train and .nnc_val file to run and outputs a single value from 0 to 1 with its confidence that each galaxy is a good fit.  
# A .nn_app file is optional (or can be provided later).
# When you initialize the NN object, it trains the NN if it doesn't exist or reads it in if it does.
# Will save the NN's and all results in <run_folder>nnc2_epoch<epochs>_corr/.
#======== 
# INPUTS
#========
# bool include_tpz_results: If true, photo-z, uncertainties, and confidences are included in the fit (should be left alone unless you specifically mean to turn it off).
# float acc_cutoff: The boundary delta z/(1+z) that the NN will use for good/bad fit classifications
# int epochs: The maximum number of epochs the NN will train for (it will never reach 1000 because of early stopping and decaying learning rate, but we have to set it as something)
# int splitnum: The number of NNs to train.  By training more than one, we can use the set of estimates to have an understanding of uncertainties.
# str run_folder: The folder where the data will be (and where the NN will be saved)
# str data_name: The name of the data files to be read in.  A minimum of 4 files with the following suffixes are needed to train - <data_name>.nnc_train, .nnc_validate, .nnc_test, .nnc_train_inds, .nnc_validate_inds.  See nn_classifier.get_data() for more details.
# list nodecounts: The last element of this list is the number of output neurons while the other entries are numbers of neurons in each hidden layer of the NN.
# list activations: Activation functions to use in each layer defined in nodecounts
# list phot_feature_list: The names of the photometric features to use for training.  Bands are letters, colors are pairs of letters, band triplets are triplets of letters; this is not a complete list of features used for training.
#============
# Attributes
#============
# bool include_tpz_results: Input parameter include_tpz_results
# float acc_cutoff: Input parameter acc_cutoff
# int epochs: Input parameter epochs - note it may not train all epochs because of Early Stopping
# int splitnum: Input parameter splitnum
# str data_name: Input parameter data_name
# str run_folder: Input parameter run_folder
# str savefolder: The specific folder where the NN is saved
# list activations: Input parameter activations
# list nnlist: Each element is a keras neural network object.  The length of this list is equal to splitnum.
# list nodecounts: Input parameter nodecounts
# numpy.ndarray feature_avgs: The average value of each feature for the whole sample.  This is used to whiten the data before training (or fitting).
# numpy.ndarray feature_names: An array of all of the feature column names used for fitting.
# numpy.ndarray feature_vars: The variance of each feature for the whole sample.  This is used to whiten the data before training (or fitting).
# numpy.ndarray nnr_goodfit_train: An array of boolean values indicating which training set galaxies are truly "good fits" using NNR redshift, i.e., delta z/(1+z) < acc_cutoff.
# numpy.ndarray nnr_goodfit_val: An array of boolean values indicating which validation set galaxies are truly "good fits" using NNR redshift, i.e., delta z/(1+z) < acc_cutoff.
# numpy.ndarray pz_goodfit_train: An array of boolean values indicating which training set galaxies are truly "good fits" using TPZ redshift, i.e., delta z/(1+z) < acc_cutoff.
# numpy.ndarray pz_goodfit_val: An array of boolean values indicating which validation set galaxies are truly "good fits" using TPZ redshift, i.e., delta z/(1+z) < acc_cutoff.
# numpy.ndarray target_train: An Nx2 array where the first value is a boolean indicating a TPZ photo-z good fit (or not) and the second is an NNR photo-z good fit (or not) for the training set.
# numpy.ndarray target_val: An Nx2 array where the first value is a boolean indicating a TPZ photo-z good fit (or not) and the second is an NNR photo-z good fit (or not) for the validation set.
#=========
# Methods
#=========
# self.get_data: Loads the training and validation data sets for the NN.  Must have column names in the first row that correspond to feature names in phot_feature_list, along with 'zphot', 'zconf', and 'zerr' (photo-z, some accuracy parameter, and photo-z uncertainty)
# self.get_models: Initializes the NN models either by training them or reading in a previously trained model if it exists
# self.create_models: Creatues new NN models that are ready for training
# self.preprocess: Whitens data in preparation for training or fitting.
# self.fit_file: Used to read in a file for fitting purposes (similar formatting to training input files).
# self.fit_features: Used to fit a numpy.ndarray of features (should not be whitened when passed in).
# self.fit_app: Looks for a file called <data_name>.nn_app in <run_folder>.  Fits the features and saves the results in <savefolder>.


class nn_classifier:

    def __init__(self, run_folder = './tpzruns/SpecSpec/', data_name = 'tpzrun', acc_cutoff = 0.07, epochs = 1000, nodecounts = [100, 200, 100, 50, 1], splitnum = 5, activations = ['selu', 'selu', 'selu', 'selu', 'sigmoid'], phot_feature_list = ['i', 'gr', 'ri', 'iz', 'zy', 'gri', 'riz', 'izy'], include_tpz_results = True, correct_pz = False):

        self.nodecounts = nodecounts
        self.activations = activations
        self.epochs = epochs
        self.splitnum = splitnum
        self.include_tpz_results = include_tpz_results
        self.correct_pz = correct_pz
        self.run_folder = run_folder
        self.data_name = data_name
        self.acc_cutoff = acc_cutoff


        if not self.include_tpz_results:
            tpzflag = '_notpz'
        else:
            tpzflag = ''
        if self.correct_pz:
            corr_flag = '_corr'
        else:
            corr_flag = ''

        self.savefolder = self.run_folder + 'nnc_epoch%i%s%s/' % (self.epochs, tpzflag, corr_flag)

        self.get_data(phot_feature_list)
        self.get_models()



    def get_models(self):

        if os.path.isdir(self.savefolder) and len(glob(self.savefolder + '*.yaml')) == self.splitnum and len(glob(self.savefolder + '*.h5')) == self.splitnum:

            self.nnlist = []

            transformfile = self.savefolder + 'NNC.transform'
            self.feature_avgs, self.feature_vars = np.loadtxt(transformfile)

            for nn_name in ['nn_%02i'%x for x in range(self.splitnum)]:

                thisyamlfile = self.savefolder + nn_name + '.yaml'
                thish5file = self.savefolder + nn_name + '.h5'

                readyaml = open(thisyamlfile, 'r')
                loaded_model_yaml = readyaml.read()
                readyaml.close()
                loaded_model = model_from_yaml(loaded_model_yaml)
                # load weights into new model
                loaded_model.load_weights(thish5file)
                loaded_model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

                self.nnlist.append(loaded_model)


        else:
            if not os.path.isdir(self.savefolder):
                os.makedirs(self.savefolder)

            # Create models

            self.create_models()

            # Fit Models

            # self.train, self.val = self.split_sample(len(self.features))
            self.feature_avgs = np.average(self.features_train, axis = 0)
            self.feature_vars = np.var(self.features_train, axis = 0)

            fig = plt.figure(figsize = (8,8))
            sp = fig.add_subplot(111)
            
            for x, thismodel in enumerate(self.nnlist):

                x_train = self.preprocess(self.features_train)
                y_train = self.is_goodfit_train.reshape(-1,1)
                x_val = self.preprocess(self.features_val)
                y_val = self.is_goodfit_val.reshape(-1,1)

                es = EarlyStopping(patience = 25, restore_best_weights = True)
                history = thismodel.fit(x_train, y_train, batch_size = 1000, epochs = self.epochs, verbose = 2, validation_data = (x_val, y_val), callbacks = [es])

                # history = thismodel.fit(x_train, y_train, batch_size = 1000, epochs = self.epochs, verbose = 2, validation_data = (x_val, y_val))
                this_fit_val = np.squeeze(thismodel.predict(self.preprocess(self.features_val), verbose = 0))

                sp.plot(history.history['loss'], label = 'Training')
                sp.plot(history.history['val_loss'], label = 'Validation')
                sp.set_ylabel('Loss')
                sp.set_xlabel('Epoch')
                sp.legend(loc = 'upper right')
                plt.savefig(self.savefolder + 'loss_%02i.png' % x, bbox_inches = 'tight')
                sp.cla()

                sp.plot(history.history['accuracy'], label = 'Training')
                sp.plot(history.history['val_accuracy'], label = 'Validation')
                sp.set_ylabel('Accuracy')
                sp.set_xlabel('Epoch')
                sp.legend(loc = 'upper left')
                plt.savefig(self.savefolder + 'acc_%02i.png' % x, bbox_inches = 'tight')
                sp.cla()

                predictions = np.squeeze(thismodel.predict(self.preprocess(self.features_val)))
                binary_predictions = (predictions>.5).astype(int)
                true_pos = np.sum((binary_predictions == self.is_goodfit_val) & (binary_predictions == 1))
                false_pos = np.sum((binary_predictions != self.is_goodfit_val) & (binary_predictions == 1))
                true_neg = np.sum((binary_predictions == self.is_goodfit_val) & (binary_predictions == 0))
                false_neg = np.sum((binary_predictions != self.is_goodfit_val) & (binary_predictions == 0))

                sp.plot([0.5, 0.5], [0,1], color = 'k')
                sp.plot([0,1], [0.5, 0.5], color = 'k')
                sp.text(.25,.25, 'FPR = %.5f' % (false_pos/float(true_neg + false_pos)), ha = 'center', va = 'center', fontsize = 30)
                sp.text(.25,.75, 'TNR = %.5f' % (true_neg/float(true_neg + false_pos)), ha = 'center', va = 'center', fontsize = 30)
                sp.text(.75,.25, 'TPR = %.5f' % (true_pos/float(false_neg + true_pos)), ha = 'center', va = 'center', fontsize = 30)
                sp.text(.75,.75, 'FNR = %.5f' % (false_neg/float(false_neg + true_pos)), ha = 'center', va = 'center', fontsize = 30)
                sp.text(0.75, 0.5, 'Acc = %.3f' % ((true_pos + true_neg) / float(len(self.is_goodfit_val))), ha = 'center', va = 'center', fontsize = 20, color = 'r', bbox = dict(facecolor = 'white', ec = 'k'))
                sp.text(0.25, 0.5, 'Misclass = %.3f' % ((false_neg + false_pos) / float(len(self.is_goodfit_val))), ha = 'center', va = 'center', fontsize = 20, color = 'r', bbox = dict(facecolor = 'white', ec = 'k'))
                sp.set_xlim(0,1)
                sp.set_ylim(0,1)
                sp.set_xticklabels([])
                sp.set_yticklabels([])
                sp.text(0.5, 1.05, 'True', ha = 'center', va = 'bottom', fontsize = 20)
                sp.text(-0.05, 0.5, 'Predicted', ha = 'right', va = 'center', rotation = 'vertical', fontsize = 20)
                sp.text(-0.02, 0.25, 'Good Fit', ha = 'right', va = 'center', rotation = 'vertical', fontsize = 20)
                sp.text(-0.02, 0.75, 'Bad Fit', ha = 'right', va = 'center', rotation = 'vertical', fontsize = 20)
                sp.text(0.25, 1.02, 'Bad Fit', ha = 'center', va = 'bottom', fontsize = 20)
                sp.text(0.75, 1.02, 'Good Fit', ha = 'center', va = 'bottom', fontsize = 20)
                plt.savefig(self.savefolder + 'confusion_%02i.png' % x, bbox_inches = 'tight')
                sp.cla()

                
                sorted_predictions, sorted_is_goodfit = np.array(list(zip(*sorted(zip(predictions, self.is_goodfit_val)))))
                sorted_is_goodfit = sorted_is_goodfit.astype(bool)

                tpr = (np.cumsum(sorted_is_goodfit[::-1])/float(np.sum(sorted_is_goodfit)))[::-1]
                fpr = (np.cumsum(~sorted_is_goodfit[::-1])/float(np.sum(~sorted_is_goodfit)))[::-1]
                auc = -np.trapz(tpr, fpr)

                sp.plot(fpr, tpr)
                sp.text(0.98, 0.02, '$AUC = %.2f$' % auc, fontsize = 20, ha = 'right', va = 'bottom')
                sp.set_xlabel('False Positive Rate')
                sp.set_ylabel('True Positive Rate')
                sp.set_xlim(0,1)
                sp.set_ylim(0,1)
                for cutoff in np.arange(0.1,1,.1):
                    thisx = np.interp(cutoff, sorted_predictions, fpr)
                    thisy = np.interp(cutoff, sorted_predictions, tpr)
                    sp.annotate('%.1f\n(%.2f, %.2f)'%(cutoff, thisx, thisy), (thisx, thisy), (thisx+.05, thisy- 0.05), arrowprops = dict(arrowstyle = '-', ec = 'k'))
                plt.savefig(self.savefolder + 'roc_%02i.png' % x, bbox_inches = 'tight')

                sp.cla()


                precision = (np.cumsum(sorted_is_goodfit[::-1])/(np.arange(len(sorted_is_goodfit[::-1]), dtype = float) + 1))[::-1]
                recall = tpr

                # Delete Later if nothing breaks
                # for y in tqdm(range(len(sorted_is_goodfit))):

                #     predict_true = sorted_is_goodfit[y:]

                #     precision.append(np.sum(predict_true)/float(len(predict_true)))
                #     # recall.append(np.sum(predict_true)/float(np.sum(sorted_is_goodfit)))

                # precision = np.array(precision)

                auc = -np.trapz(precision, recall)
                sp.text(0.98, 0.02, '$AUC = %.2f$' % auc, fontsize = 20, ha = 'right', va = 'bottom')
                sp.plot(recall, precision)
                sp.set_xlabel('Recall')
                sp.set_ylabel('Precision')
                sp.set_xlim(0,1)
                sp.set_ylim(0,1)
                for cutoff in np.arange(0.1,1,.1):
                    thisx = np.interp(cutoff, sorted_predictions, recall)
                    thisy = np.interp(cutoff, sorted_predictions, precision)
                    sp.annotate('%.1f\n(%.2f, %.2f)'%(cutoff, thisx, thisy), (thisx, thisy), (thisx-.05, thisy- 0.05), arrowprops = dict(arrowstyle = '-', ec = 'k'))
                try:
                    plt.savefig(self.savefolder + 'precision_recall_%02i.png' % x, bbox_inches = 'tight')
                except:
                    print('three')
                sp.cla()

            plt.close()

            # Write Models

            if not os.path.isdir(self.savefolder):
                os.makedirs(self.savefolder)

            for nn_number, thismodel in enumerate(self.nnlist):

                nn_name = 'nn_%02i' % nn_number

                model_yaml = thismodel.to_yaml()
                thisyamlfile = self.savefolder + nn_name + '.yaml'
                thish5file = self.savefolder + nn_name + '.h5'
                thistransformfile = self.savefolder + nn_name + '.transform'

                with open(thisyamlfile, "w") as yamlwrite:
                    yamlwrite.write(model_yaml)
                # serialize weights to HDF5
                thismodel.save_weights(thish5file)


            transformfile = self.savefolder + 'NNC.transform'
            np.savetxt(transformfile, np.vstack((self.feature_avgs, self.feature_vars)))



            savekeys = ['acc_cutoff', 'feature_names', 'nodecounts', 'activations', 'epochs', 'run_folder']

            with open(self.savefolder + 'info.txt', 'w') as writefile:
                for thiskey in savekeys:
                    writefile.write(thiskey + ': ' + repr(getattr(self, thiskey)) + '\n')


            np.savetxt(self.savefolder + 'results_train.dat', np.average(self.fit_features(self.features_train), axis = 1))
            np.savetxt(self.savefolder + 'results_validate.dat', np.average(self.fit_features(self.features_val), axis = 1))

            stub = ''.join(('r_epoch'.join(self.savefolder.split('c_epoch'))).split('_corr'))

            if os.path.isfile(self.run_folder + self.data_name + '.nnc_test') and os.path.isfile(stub + 'results_test.dat'):
                np.savetxt(self.savefolder + 'results_test.dat', self.fit_file(self.run_folder + self.data_name + '.nnc_test', stub + 'results_test.dat', self.run_folder + self.data_name + '.nnc_test_inds'))







    def get_data(self, phot_feature_list):

        keep_feature_list = phot_feature_list
        if self.include_tpz_results:
            keep_feature_list = keep_feature_list + ['zphot', 'zconf', 'zerr']

        trainfile = read_csv(self.run_folder + self.data_name  + '.nnc_train', delimiter = '\s+', comment = '#')
        valfile = read_csv(self.run_folder + self.data_name  + '.nnc_validate', delimiter = '\s+', comment = '#')
        self.features_train = trainfile[keep_feature_list].values
        self.features_val = valfile[keep_feature_list].values

        self.feature_names = np.array(keep_feature_list)

        pz_train = np.squeeze(self.features_train.T[self.feature_names == 'zphot'])
        sz_train = trainfile['specz'].to_numpy()
        pz_err_train = np.squeeze(self.features_train.T[self.feature_names == 'zerr'])

        pz_val = np.squeeze(self.features_val.T[self.feature_names == 'zphot'])
        sz_val = valfile['specz'].to_numpy()
        pz_err_val = np.squeeze(self.features_val.T[self.feature_names == 'zerr'])

        train_trans_inds = np.loadtxt(self.run_folder + self.data_name + '.nnc_train_inds', unpack = True, dtype = int)[1]
        val_trans_inds = np.loadtxt(self.run_folder + self.data_name + '.nnc_validate_inds', unpack = True, dtype = int)[1]

        if self.correct_pz:
            # Add corrected photometric redshifts to features
            stub = ''.join(('r_epoch'.join(self.savefolder.split('c_epoch'))).split('_corr'))
            regressed_err_test = np.loadtxt(stub + 'results_test.dat')

            self.feature_names = np.append(self.feature_names, ['zphot_NNR'])
            self.features_train = np.hstack((self.features_train, (pz_train - regressed_err_test[train_trans_inds]).reshape(-1,1)))
            self.features_val = np.hstack((self.features_val, (pz_val - regressed_err_test[val_trans_inds]).reshape(-1,1)))

        self.is_goodfit_train = (np.abs(pz_train - sz_train) < (1. + sz_train)*self.acc_cutoff)
        self.is_goodfit_val = (np.abs(pz_val - sz_val) < (1. + sz_val)*self.acc_cutoff)



    def create_models(self):

        # Initializes untrained neural networks

        self.nnlist = []

        for x in range(self.splitnum):

            model = Sequential()
            model.add(Dense(self.nodecounts[0], activation = self.activations[0], input_shape = (self.features_train.shape[1],)))
            # model.add(Dropout(0.2))

            for thisnodenum, thisactivation in zip(self.nodecounts[1:], self.activations[1:]):

                if thisactivation == 'selu':
                    kernel_initializer = 'lecun_normal'
                else:
                    kernel_initializer = 'glorot_uniform'
                model.add(Dense(thisnodenum, activation = thisactivation, kernel_initializer = kernel_initializer))

            initial_learning_rate = 0.005
            lr_schedule = ExponentialDecay(
                initial_learning_rate,
                decay_steps=3500,
                decay_rate=0.1,
                staircase=False)

            model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate = lr_schedule), metrics = ['accuracy'])

            self.nnlist.append(model)


    def preprocess(self, X):

        return (X - self.feature_avgs)/np.sqrt(self.feature_vars)


    def fit_file(self, fname, nnr_results_fname = None, index_file = None, average_results = True):

        feature_file = read_csv(fname, comment = '#', delimiter = '\s+')
        
        if (not self.correct_pz) and index_file != None:
            translate_inds = np.loadtxt(index_file, unpack = True, dtype = int)[1]
            feature_file['zphot_NNR'] = feature_file['zphot'] - np.loadtxt(nnr_results_fname)[translate_inds]
            features_fit = feature_file[self.feature_names].to_numpy()
        elif self.correct_pz:
            feature_file['zphot_NNR'] = feature_file['zphot'] - np.loadtxt(nnr_results_fname)

        features_fit = feature_file[self.feature_names].to_numpy()

        if average_results:
            return np.average((self.fit_features(features_fit)), axis = 1)
        else:
            return self.fit_features(features_fit)


    def fit_features(self, features):

        processed_features = self.preprocess(features)

        predictions = []

        for thismodel in self.nnlist:

            predictions.append(thismodel.predict(processed_features, verbose = 1))

        return np.squeeze(np.array(predictions).T)


    def fit_app(self):

        if self.correct_pz:
            stub = ''.join(('r_epoch'.join(self.savefolder.split('c_epoch'))).split('_corr'))
            np.savetxt(self.savefolder + 'results_application.dat', self.fit_file(self.run_folder + self.data_name + '.nn_app', stub + 'results_application.dat'))
        else:
            np.savetxt(self.savefolder + 'results_application.dat', self.fit_file(self.run_folder + self.data_name + '.nn_app'))








#=============
# Description
#=============
# This is the OG neural network (NN) object, which requires a .nnc_train and .nnc_val file to run and outputs a single value from 0 to 1 with its confidence that each galaxy is a good fit.  
# A .nn_app file is optional (or can be provided later).
# When you initialize the NN object, it trains the NN if it doesn't exist or reads it in if it does.
# Will save the NN's and all results in <run_folder>nnc2_epoch<epochs>_corr/.
#======== 
# INPUTS
#========
# bool include_tpz_results: If true, photo-z, uncertainties, and confidences are included in the fit (should be left alone unless you specifically mean to turn it off).
# float acc_cutoff: The boundary delta z/(1+z) that the NN will use for good/bad fit classifications
# int epochs: The maximum number of epochs the NN will train for (it will never reach 1000 because of early stopping and decaying learning rate, but we have to set it as something)
# int splitnum: The number of NNs to train.  By training more than one, we can use the set of estimates to have an understanding of uncertainties.
# str run_folder: The folder where the data will be (and where the NN will be saved)
# str data_name: The name of the data files to be read in.  A minimum of 4 files with the following suffixes are needed to train - <data_name>.nnc_train, .nnc_validate, .nnc_test, .nnc_train_inds, .nnc_validate_inds.  See nn_classifier.get_data() for more details.
# list nodecounts: The last element of this list is the number of output neurons while the other entries are numbers of neurons in each hidden layer of the NN.
# list activations: Activation functions to use in each layer defined in nodecounts
# list phot_feature_list: The names of the photometric features to use for training.  Bands are letters, colors are pairs of letters, band triplets are triplets of letters; this is not a complete list of features used for training.
#============
# Attributes
#============
# bool include_tpz_results: Input parameter include_tpz_results
# float acc_cutoff: Input parameter acc_cutoff
# int epochs: Input parameter epochs
# int splitnum: Input parameter splitnum
# str data_name: Input parameter data_name
# str run_folder: Input parameter run_folder
# str savefolder: The specific folder where the NN is saved
# list activations: Input parameter activations
# list nnlist: Each element is a keras neural network object.  The length of this list is equal to splitnum.
# list nodecounts: Input parameter nodecounts
# numpy.ndarray feature_avgs: The average value of each feature for the whole sample.  This is used to whiten the data before training (or fitting).
# numpy.ndarray feature_names: An array of all of the feature column names used for fitting.
# numpy.ndarray feature_vars: The variance of each feature for the whole sample.  This is used to whiten the data before training (or fitting).
# numpy.ndarray nnr_goodfit_train: An array of boolean values indicating which training set galaxies are truly "good fits" using NNR redshift, i.e., delta z/(1+z) < acc_cutoff.
# numpy.ndarray nnr_goodfit_val: An array of boolean values indicating which validation set galaxies are truly "good fits" using NNR redshift, i.e., delta z/(1+z) < acc_cutoff.
# numpy.ndarray pz_goodfit_train: An array of boolean values indicating which training set galaxies are truly "good fits" using TPZ redshift, i.e., delta z/(1+z) < acc_cutoff.
# numpy.ndarray pz_goodfit_val: An array of boolean values indicating which validation set galaxies are truly "good fits" using TPZ redshift, i.e., delta z/(1+z) < acc_cutoff.
# numpy.ndarray target_train: An Nx2 array where the first value is a boolean indicating a TPZ photo-z good fit (or not) and the second is an NNR photo-z good fit (or not) for the training set.
# numpy.ndarray target_val: An Nx2 array where the first value is a boolean indicating a TPZ photo-z good fit (or not) and the second is an NNR photo-z good fit (or not) for the validation set.
#=========
# Methods
#=========
# self.get_data: Loads the training and validation data sets for the NN
# self.get_models: Initializes the NN models either by training them or reading in a previously trained model if it exists
# self.create_models: Creatues new NN models that are ready for training
# self.preprocess: Whitens data in preparation for training or fitting.
# self.fit_file: Used to read in a file for fitting purposes (similar formatting to training input files).
# self.fit_features: Used to fit a numpy.ndarray of features (should not be whitened when passed in).
# self.fit_app: Looks for a file called <data_name>.nn_app in <run_folder>.  Fits the features and saves the results in <savefolder>.


class nn_classifier2:

    def __init__(self, run_folder = './tpzruns/default_run/', data_name = 'default', acc_cutoff = 0.02, epochs = 1000, nodecounts = [100, 200, 100, 50, 2], splitnum = 5, activations = ['selu', 'selu', 'selu', 'selu', 'sigmoid'], phot_feature_list = ['i', 'gr', 'ri', 'iz', 'zy', 'gri', 'riz', 'izy']):

        self.nodecounts = nodecounts
        self.activations = activations
        self.epochs = epochs
        self.splitnum = splitnum
        self.run_folder = run_folder
        self.data_name = data_name
        self.acc_cutoff = acc_cutoff

        self.savefolder = self.run_folder + 'nnc2_epoch%i/' % self.epochs

        self.get_data(phot_feature_list)
        self.get_models()



    def get_models(self):

        if os.path.isdir(self.savefolder) and len(glob(self.savefolder + '*.yaml')) == self.splitnum and len(glob(self.savefolder + '*.h5')) == self.splitnum:

            self.nnlist = []

            transformfile = self.savefolder + 'NNC2.transform'
            self.feature_avgs, self.feature_vars = np.loadtxt(transformfile)

            for nn_name in ['nn_%02i'%x for x in range(self.splitnum)]:

                thisyamlfile = self.savefolder + nn_name + '.yaml'
                thish5file = self.savefolder + nn_name + '.h5'

                readyaml = open(thisyamlfile, 'r')
                loaded_model_yaml = readyaml.read()
                readyaml.close()
                loaded_model = model_from_yaml(loaded_model_yaml)
                # load weights into new model
                loaded_model.load_weights(thish5file)
                loaded_model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

                self.nnlist.append(loaded_model)


        else:
            if not os.path.isdir(self.savefolder):
                os.makedirs(self.savefolder)

            # Create models

            self.create_models()

            # Fit Models

            # self.train, self.val = self.split_sample(len(self.features))
            self.feature_avgs = np.average(self.features_train, axis = 0)
            self.feature_vars = np.var(self.features_train, axis = 0)

            
            for x, thismodel in enumerate(self.nnlist):

                x_train = self.preprocess(self.features_train)
                y_train = self.target_train
                x_val = self.preprocess(self.features_val)
                y_val = self.target_val

                es = EarlyStopping(patience = 25, restore_best_weights = True)
                history = thismodel.fit(x_train, y_train, batch_size = 1000, epochs = self.epochs, verbose = 2, validation_data = (x_val, y_val), callbacks = [es])

                # history = thismodel.fit(x_train, y_train, batch_size = 1000, epochs = self.epochs, verbose = 2, validation_data = (x_val, y_val))
                this_fit_val = thismodel.predict(self.preprocess(self.features_val), verbose = 0)

                
            # Write Models

            if not os.path.isdir(self.savefolder):
                os.makedirs(self.savefolder)

            for nn_number, thismodel in enumerate(self.nnlist):

                nn_name = 'nn_%02i' % nn_number

                model_yaml = thismodel.to_yaml()
                thisyamlfile = self.savefolder + nn_name + '.yaml'
                thish5file = self.savefolder + nn_name + '.h5'
                thistransformfile = self.savefolder + nn_name + '.transform'

                with open(thisyamlfile, "w") as yamlwrite:
                    yamlwrite.write(model_yaml)
                # serialize weights to HDF5
                thismodel.save_weights(thish5file)


            transformfile = self.savefolder + 'NNC2.transform'
            np.savetxt(transformfile, np.vstack((self.feature_avgs, self.feature_vars)))

            savekeys = ['acc_cutoff', 'feature_names', 'nodecounts', 'activations', 'epochs', 'run_folder']

            with open(self.savefolder + 'info.txt', 'w') as writefile:
                for thiskey in savekeys:
                    writefile.write(thiskey + ': ' + repr(getattr(self, thiskey)) + '\n')


            np.savetxt(self.savefolder + 'results_train.dat', np.average(self.fit_features(self.features_train), axis = -1))
            np.savetxt(self.savefolder + 'results_validate.dat', np.average(self.fit_features(self.features_val), axis = -1))

            stub = ''.join(('r_epoch'.join(self.savefolder.split('c_epoch'))))

            if os.path.isfile(self.run_folder + self.data_name + '.nnc_test') and os.path.isfile(stub + 'results_test.dat'):
                np.savetxt(self.savefolder + 'results_test.dat', self.fit_file(self.run_folder + self.data_name + '.nnc_test', stub + 'results_test.dat', self.run_folder + self.data_name + '.nnc_test_inds'))







    def get_data(self, phot_feature_list):

        keep_feature_list = phot_feature_list + ['zphot', 'zconf', 'zerr']

        trainfile = read_csv(self.run_folder + self.data_name  + '.nnc_train', delimiter = '\s+', comment = '#')
        valfile = read_csv(self.run_folder + self.data_name  + '.nnc_validate', delimiter = '\s+', comment = '#')
        self.features_train = trainfile[keep_feature_list].values
        self.features_val = valfile[keep_feature_list].values

        self.feature_names = np.array(keep_feature_list)

        pz_train = np.squeeze(self.features_train.T[self.feature_names == 'zphot'])
        sz_train = trainfile['specz'].to_numpy()
        pz_err_train = np.squeeze(self.features_train.T[self.feature_names == 'zerr'])

        pz_val = np.squeeze(self.features_val.T[self.feature_names == 'zphot'])
        sz_val = valfile['specz'].to_numpy()
        pz_err_val = np.squeeze(self.features_val.T[self.feature_names == 'zerr'])

        train_trans_inds = np.loadtxt(self.run_folder + self.data_name + '.nnc_train_inds', unpack = True, dtype = int)[1]
        val_trans_inds = np.loadtxt(self.run_folder + self.data_name + '.nnc_validate_inds', unpack = True, dtype = int)[1]

        # Add corrected photometric redshifts to features
        stub = ''.join(('r_epoch'.join(self.savefolder.split('c2_epoch'))))
        regressed_err_test = np.loadtxt(stub + 'results_test.dat')

        self.feature_names = np.append(self.feature_names, ['zphot_NNR'])
        self.features_train = np.hstack((self.features_train, (pz_train - regressed_err_test[train_trans_inds]).reshape(-1,1)))
        self.features_val = np.hstack((self.features_val, (pz_val - regressed_err_test[val_trans_inds]).reshape(-1,1)))

        # self.nnr_betterfit_train = (pz_train - regressed_err_test[train_trans_inds] - sz_train) < (pz_train - sz_train)
        # self.nnr_betterfit_val = (pz_val - regressed_err_test[val_trans_inds] - sz_val) < (pz_val - sz_val)

        # self.is_goodfit_train = (np.abs(pz_train - (self.nnr_betterfit_train * regressed_err_test[train_trans_inds]) - sz_train) < (1. + sz_train)*self.acc_cutoff)
        # self.is_goodfit_val = (np.abs(pz_val - (self.nnr_betterfit_val * regressed_err_test[val_trans_inds]) - sz_val) < (1. + sz_val)*self.acc_cutoff)

        self.nnr_goodfit_train = (np.abs(pz_train - regressed_err_test[train_trans_inds] - sz_train) < (1. + sz_train)*self.acc_cutoff)
        self.nnr_goodfit_val = (np.abs(pz_val - regressed_err_test[val_trans_inds] - sz_val) < (1. + sz_val)*self.acc_cutoff)
        self.pz_goodfit_train = (np.abs(pz_train - sz_train) < (1. + sz_train)*self.acc_cutoff)
        self.pz_goodfit_val = (np.abs(pz_val - sz_val) < (1. + sz_val)*self.acc_cutoff)

        self.target_train = np.vstack((self.pz_goodfit_train, self.nnr_goodfit_train)).T
        self.target_val = np.vstack((self.pz_goodfit_val, self.nnr_goodfit_val)).T


    def create_models(self):

        # Initializes untrained neural networks

        self.nnlist = []

        for x in range(self.splitnum):

            model = Sequential()
            model.add(Dense(self.nodecounts[0], activation = self.activations[0], input_shape = (self.features_train.shape[1],)))
            # model.add(Dropout(0.2))

            for thisnodenum, thisactivation in zip(self.nodecounts[1:], self.activations[1:]):

                if thisactivation == 'selu':
                    kernel_initializer = 'lecun_normal'
                else:
                    kernel_initializer = 'glorot_uniform'
                model.add(Dense(thisnodenum, activation = thisactivation, kernel_initializer = kernel_initializer))

            initial_learning_rate = 0.005
            lr_schedule = ExponentialDecay(
                initial_learning_rate,
                decay_steps=3500,
                decay_rate=0.1,
                staircase=False)

            model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate = lr_schedule), metrics = ['accuracy'])

            self.nnlist.append(model)


    def preprocess(self, X):

        return (X - self.feature_avgs)/np.sqrt(self.feature_vars)


    def fit_file(self, fname, nnr_results_fname, index_file = None, average_results = True):

        # The nnr_results file should be created using something like np.savetxt(nnr_results_fname, nn_regressor.fit_file(nn_regressor.run_folder + fitfilename))
        # The index_file can be used to reorder or sub-select objects in the nnr_results file to match them up to the attributes in the fit file fname
        # Because there are <splitnum> neural networks, average_results will average their predictions to a single value (setting to false returns all 5 fit values).

        feature_file = read_csv(fname, comment = '#', delimiter = '\s+')
        
        if index_file != None:
            translate_inds = np.loadtxt(index_file, unpack = True, dtype = int)[1]
            feature_file['zphot_NNR'] = feature_file['zphot'] - np.loadtxt(nnr_results_fname)[translate_inds]
            features_fit = feature_file[self.feature_names].to_numpy()
        else:
            feature_file['zphot_NNR'] = feature_file['zphot'] - np.loadtxt(nnr_results_fname)

        features_fit = feature_file[self.feature_names].to_numpy()

        if average_results:
            return np.average((self.fit_features(features_fit)), axis = -1).T
        else:
            return self.fit_features(features_fit)


    def fit_features(self, features):

        processed_features = self.preprocess(features)

        predictions = []

        for thismodel in self.nnlist:

            predictions.append(thismodel.predict(processed_features, verbose = 1))

        return np.squeeze(np.array(predictions).T)


    def fit_app(self):

        stub = ''.join(('r_epoch'.join(self.savefolder.split('c2_epoch'))))
        np.savetxt(self.savefolder + 'results_application.dat', self.fit_file(self.run_folder + self.data_name + '.nn_app', stub + 'results_application.dat'))





class nn_classifier4:

    def __init__(self, run_folder = './tpzruns/default_run/', data_name = 'default', acc_cutoff = 0.02, epochs = 1000, nodecounts = [100, 200, 100, 50, 4], splitnum = 5, activations = ['selu', 'selu', 'selu', 'selu', 'sigmoid'], phot_feature_list = ['i', 'gr', 'ri', 'iz', 'zy', 'gri', 'riz', 'izy'], include_tpz_results = True, correct_pz = True):

        self.nodecounts = nodecounts
        self.activations = activations
        self.epochs = epochs
        self.splitnum = splitnum
        self.include_tpz_results = include_tpz_results
        self.correct_pz = correct_pz
        self.run_folder = run_folder
        self.data_name = data_name
        self.acc_cutoff = acc_cutoff


        if not self.include_tpz_results:
            tpzflag = '_notpz'
        else:
            tpzflag = ''
        if self.correct_pz:
            corr_flag = '_corr'
        else:
            corr_flag = ''

        self.savefolder = self.run_folder + 'nnc4_epoch%i%s%s/' % (self.epochs, tpzflag, corr_flag)

        self.get_data(phot_feature_list)
        self.get_models()



    def get_models(self):

        if os.path.isdir(self.savefolder) and len(glob(self.savefolder + '*.yaml')) == self.splitnum and len(glob(self.savefolder + '*.h5')) == self.splitnum:

            self.nnlist = []

            transformfile = self.savefolder + 'NNC4.transform'
            self.feature_avgs, self.feature_vars = np.loadtxt(transformfile)

            for nn_name in ['nn_%02i'%x for x in range(self.splitnum)]:

                thisyamlfile = self.savefolder + nn_name + '.yaml'
                thish5file = self.savefolder + nn_name + '.h5'

                readyaml = open(thisyamlfile, 'r')
                loaded_model_yaml = readyaml.read()
                readyaml.close()
                loaded_model = model_from_yaml(loaded_model_yaml)
                # load weights into new model
                loaded_model.load_weights(thish5file)
                loaded_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

                self.nnlist.append(loaded_model)


        else:
            if not os.path.isdir(self.savefolder):
                os.makedirs(self.savefolder)

            # Create models

            self.create_models()

            # Fit Models

            # self.train, self.val = self.split_sample(len(self.features))
            self.feature_avgs = np.average(self.features_train, axis = 0)
            self.feature_vars = np.var(self.features_train, axis = 0)

            # if self.include_tpz_results:
            #     self.feature_avgs[self.feature_names == 'pz'] = 0.
            #     self.feature_vars[self.]
            
            for x, thismodel in enumerate(self.nnlist):

                x_train = self.preprocess(self.features_train)
                y_train = self.target_train
                x_val = self.preprocess(self.features_val)
                y_val = self.target_val

                es = EarlyStopping(patience = 25, restore_best_weights = True)
                history = thismodel.fit(x_train, y_train, batch_size = 1000, epochs = self.epochs, verbose = 2, validation_data = (x_val, y_val), callbacks = [es])

                # history = thismodel.fit(x_train, y_train, batch_size = 1000, epochs = self.epochs, verbose = 2, validation_data = (x_val, y_val))
                this_fit_val = thismodel.predict(self.preprocess(self.features_val), verbose = 0)

                # bool include_tpz_results: Input parameter include_tpz_results
            # Write Models

            if not os.path.isdir(self.savefolder):
                os.makedirs(self.savefolder)

            for nn_number, thismodel in enumerate(self.nnlist):

                nn_name = 'nn_%02i' % nn_number

                model_yaml = thismodel.to_yaml()
                thisyamlfile = self.savefolder + nn_name + '.yaml'
                thish5file = self.savefolder + nn_name + '.h5'
                thistransformfile = self.savefolder + nn_name + '.transform'

                with open(thisyamlfile, "w") as yamlwrite:
                    yamlwrite.write(model_yaml)
                # serialize weights to HDF5
                thismodel.save_weights(thish5file)


            transformfile = self.savefolder + 'NNC4.transform'
            np.savetxt(transformfile, np.vstack((self.feature_avgs, self.feature_vars)))

            savekeys = ['acc_cutoff', 'feature_names', 'nodecounts', 'activations', 'epochs', 'run_folder']

            with open(self.savefolder + 'info.txt', 'w') as writefile:
                for thiskey in savekeys:
                    writefile.write(thiskey + ': ' + repr(getattr(self, thiskey)) + '\n')


            np.savetxt(self.savefolder + 'results_train.dat', np.average(self.fit_features(self.features_train), axis = -1))
            np.savetxt(self.savefolder + 'results_validate.dat', np.average(self.fit_features(self.features_val), axis = -1))

            stub = ''.join(('r_epoch'.join(self.savefolder.split('c_epoch'))).split('_corr'))

            if os.path.isfile(self.run_folder + self.data_name + '.nnc_test') and os.path.isfile(stub + 'results_test.dat'):
                np.savetxt(self.savefolder + 'results_test.dat', self.fit_file(self.run_folder + self.data_name + '.nnc_test', stub + 'results_test.dat', self.run_folder + self.data_name + '.nnc_test_inds'))







    def get_data(self, phot_feature_list):

        keep_feature_list = phot_feature_list
        if self.include_tpz_results:
            keep_feature_list = keep_feature_list + ['zphot', 'zconf', 'zerr']

        trainfile = read_csv(self.run_folder + self.data_name  + '.nnc_train', delimiter = '\s+', comment = '#')
        valfile = read_csv(self.run_folder + self.data_name  + '.nnc_validate', delimiter = '\s+', comment = '#')
        self.features_train = trainfile[keep_feature_list].values
        self.features_val = valfile[keep_feature_list].values

        self.feature_names = np.array(keep_feature_list)

        pz_train = np.squeeze(self.features_train.T[self.feature_names == 'zphot'])
        sz_train = trainfile['specz'].to_numpy()
        pz_err_train = np.squeeze(self.features_train.T[self.feature_names == 'zerr'])

        pz_val = np.squeeze(self.features_val.T[self.feature_names == 'zphot'])
        sz_val = valfile['specz'].to_numpy()
        pz_err_val = np.squeeze(self.features_val.T[self.feature_names == 'zerr'])

        train_trans_inds = np.loadtxt(self.run_folder + self.data_name + '.nnc_train_inds', unpack = True, dtype = int)[1]
        val_trans_inds = np.loadtxt(self.run_folder + self.data_name + '.nnc_validate_inds', unpack = True, dtype = int)[1]

        # Add corrected photometric redshifts to features
        stub = ''.join(('r_epoch'.join(self.savefolder.split('c4_epoch'))).split('_corr'))
        regressed_err_test = np.loadtxt(stub + 'results_test.dat')

        self.feature_names = np.append(self.feature_names, ['zphot_NNR'])
        self.features_train = np.hstack((self.features_train, (pz_train - regressed_err_test[train_trans_inds]).reshape(-1,1)))
        self.features_val = np.hstack((self.features_val, (pz_val - regressed_err_test[val_trans_inds]).reshape(-1,1)))

        self.nnr_betterfit_train = (pz_train - regressed_err_test[train_trans_inds] - sz_train) < (pz_train - sz_train)
        self.nnr_betterfit_val = (pz_val - regressed_err_test[val_trans_inds] - sz_val) < (pz_val - sz_val)

        self.is_goodfit_train = (np.abs(pz_train - (self.nnr_betterfit_train * regressed_err_test[train_trans_inds]) - sz_train) < (1. + sz_train)*self.acc_cutoff)
        self.is_goodfit_val = (np.abs(pz_val - (self.nnr_betterfit_val * regressed_err_test[val_trans_inds]) - sz_val) < (1. + sz_val)*self.acc_cutoff)

        self.target_train = np.vstack((~(self.nnr_betterfit_train & self.is_goodfit_train), (~self.nnr_betterfit_train & self.is_goodfit_train), (self.nnr_betterfit_train & ~self.is_goodfit_train), (self.nnr_betterfit_train & self.is_goodfit_train))).T
        self.target_val = np.vstack((~(self.nnr_betterfit_val & self.is_goodfit_val), (~self.nnr_betterfit_val & self.is_goodfit_val), (self.nnr_betterfit_val & ~self.is_goodfit_val), (self.nnr_betterfit_val & self.is_goodfit_val))).T
        # 0: TPZ is better photoz, is not good fit
        # 1: TPZ is better photoz, is good fit
        # 2: NNR is better photoz, is not good fit
        # 3: NNR is better photoz, is good fit


    def create_models(self):

        # Initializes untrained neural networks

        self.nnlist = []

        for x in range(self.splitnum):

            model = Sequential()
            model.add(Dense(self.nodecounts[0], activation = self.activations[0], input_shape = (self.features_train.shape[1],)))
            # model.add(Dropout(0.2))

            for thisnodenum, thisactivation in zip(self.nodecounts[1:], self.activations[1:]):

                if thisactivation == 'selu':
                    kernel_initializer = 'lecun_normal'
                else:
                    kernel_initializer = 'glorot_uniform'
                model.add(Dense(thisnodenum, activation = thisactivation, kernel_initializer = kernel_initializer))

            initial_learning_rate = 0.005
            lr_schedule = ExponentialDecay(
                initial_learning_rate,
                decay_steps=3500,
                decay_rate=0.1,
                staircase=False)

            model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate = lr_schedule), metrics = ['accuracy'])

            self.nnlist.append(model)


    def preprocess(self, X):

        return (X - self.feature_avgs)/np.sqrt(self.feature_vars)


    def fit_file(self, fname, nnr_results_fname, index_file = None, average_results = True):

        feature_file = read_csv(fname, comment = '#', delimiter = '\s+')
        
        if index_file != None:
            translate_inds = np.loadtxt(index_file, unpack = True, dtype = int)[1]
            feature_file['zphot_NNR'] = feature_file['zphot'] - np.loadtxt(nnr_results_fname)[translate_inds]
            features_fit = feature_file[self.feature_names].to_numpy()
        else:
            feature_file['zphot_NNR'] = feature_file['zphot'] - np.loadtxt(nnr_results_fname)

        features_fit = feature_file[self.feature_names].to_numpy()

        if average_results:
            return np.average((self.fit_features(features_fit)), axis = -1).T
        else:
            return self.fit_features(features_fit)


    def fit_features(self, features):

        processed_features = self.preprocess(features)

        predictions = []

        for thismodel in self.nnlist:

            predictions.append(thismodel.predict(processed_features, verbose = 1))

        return np.squeeze(np.array(predictions).T)


    def fit_app(self):

        stub = ''.join(('r_epoch'.join(self.savefolder.split('c4_epoch'))).split('_corr'))
        np.savetxt(self.savefolder + 'results_application.dat', self.fit_file(self.run_folder + self.data_name + '.nn_app', stub + 'results_application.dat'))





class nn_regressor:

    #def __init__(self, run_folder = './tpzruns/default_run/', data_name = 'default', savefolder = None, epochs = 1000, nodecounts = [100, 200, 100, 50, 1], activations = ['selu', 'selu', 'selu', 'selu', 'linear'], splitnum = 5, phot_feature_list = ['i', 'gr', 'ri', 'iz', 'zy', 'gri', 'riz', 'izy'], include_tpz_results = True):
    def __init__(self, run_folder = '/home/DESC/tomo_challenge', data_name = 'training', savefolder = None, epochs = 1000, nodecounts = [100, 200, 100, 50, 1], activations = ['selu', 'selu', 'selu', 'selu', 'linear'], splitnum = 5, phot_feature_list = ['i', 'gr', 'ri', 'iz', 'zy', 'gri', 'riz', 'izy'], include_tpz_results = True):

        self.nodecounts = nodecounts
        self.activations = activations
        self.epochs = epochs
        self.splitnum = splitnum
        self.include_tpz_results = include_tpz_results
        self.run_folder = run_folder
        self.data_name = data_name


        if savefolder == None:
            if not self.include_tpz_results:
                tpzflag = '_notpz'
            else:
                tpzflag = ''

            self.savefolder = self.run_folder + 'nnr_epoch%i%s/' % (self.epochs, tpzflag)

        else:
            self.savefolder = savefolder

        self.get_data(phot_feature_list)
        self.get_models()



    def get_models(self):

        if os.path.isdir(self.savefolder) and len(glob(self.savefolder + '*.yaml')) == self.splitnum and len(glob(self.savefolder + '*.h5')) == self.splitnum:

            self.nnlist = []

            transformfile = self.savefolder + 'NNR.transform'
            self.feature_avgs, self.feature_vars = np.loadtxt(transformfile)

            for nn_name in ['nn_%02i'%x for x in range(self.splitnum)]:

                thisyamlfile = self.savefolder + nn_name + '.yaml'
                thish5file = self.savefolder + nn_name + '.h5'
                

                readyaml = open(thisyamlfile, 'r')
                loaded_model_yaml = readyaml.read()
                readyaml.close()
                loaded_model = model_from_yaml(loaded_model_yaml)
                # load weights into new model
                loaded_model.load_weights(thish5file)

                initial_learning_rate = 0.001
                lr_schedule = ExponentialDecay(
                    initial_learning_rate,
                    decay_steps=10000,
                    decay_rate=0.1,
                    staircase=False)
                
                # huber = Huber(delta = self.huber_delta, name = 'huber_loss')

                loaded_model.compile(loss = 'mse', optimizer = Adam(learning_rate = lr_schedule), metrics = ['accuracy'])

                self.nnlist.append(loaded_model)


        else:
            if not os.path.isdir(self.savefolder):
                os.makedirs(self.savefolder)

            # Create models

            self.create_models()

            # Fit Models

            self.feature_avgs = np.average(self.features_train, axis = 0)
            self.feature_vars = np.var(self.features_train, axis = 0)

            # if self.include_tpz_results:
            #     self.feature_avgs[self.feature_names == 'pz'] = 0.
            #     self.feature_vars[self.]

            fig1 = plt.figure(figsize = (8,8))
            sp1 = fig1.add_subplot(111)
            fig2 = plt.figure(figsize = (8,8)) # Matplotlib doesn't properly size the figure after clearing the axes if you used imshow to plot on it
            sp2 = fig2.add_subplot(111)
            

            for x, thismodel in enumerate(self.nnlist):

                x_train = self.preprocess(self.features_train)
                y_train = self.bias_train.reshape(-1,1)
                x_val = self.preprocess(self.features_val)
                y_val = self.bias_val.reshape(-1,1)

                es = EarlyStopping(patience = 25, restore_best_weights = True)
                history = thismodel.fit(x_train, y_train, batch_size = 1000, epochs = self.epochs, verbose = 2, validation_data = (x_val, y_val), callbacks = [es])

                # history = thismodel.fit(x_train, y_train, batch_size = 1000, epochs = self.epochs, verbose = 2, validation_data = (x_val, y_val))

                sp1.plot(history.history['loss'], label = 'Training')
                sp1.plot(history.history['val_loss'], label = 'Validation')

                sp1.set_ylim(-.1*max(history.history['val_loss']), 1.2*max(history.history['val_loss']))

                sp1.set_ylabel('Loss')
                sp1.set_xlabel('Epoch')
                sp1.legend(loc = 'upper right')
                fig1.savefig(self.savefolder + 'loss_%02i.png' % x, bbox_inches = 'tight')
                sp1.cla()

                predictions = np.squeeze(thismodel.predict(self.preprocess(self.features_val)))

                # h = np.histogram2d(self.bias, predictions, bins = 100, range = [[0,2],[-2,2]])[0]
                # sp2.imshow(np.log10(h).T, origin = 'lower', cmap = 'plasma', extent = (0,2,-2,2), aspect = 'auto')
                # sp2.set_xlabel('Accuracy')
                # sp2.set_ylabel('Prediction')
                # plt.savefig(self.savefolder + 'predictions_%02i.png' % x, bbox_inches = 'tight')
                sp2.scatter(self.bias_val, predictions, marker = '.')
                fig2.savefig(self.savefolder + 'predictions_%02i.png' % x, bbox_inches = 'tight')
                sp2.set_xlabel('Accuracy')
                sp2.set_ylabel('Prediction')
                sp2.cla()

            plt.close()
            plt.close()

            # Write Models

            if not os.path.isdir(self.savefolder):
                os.makedirs(self.savefolder)

            for nn_number, thismodel in enumerate(self.nnlist):

                nn_name = 'nn_%02i' % nn_number

                model_yaml = thismodel.to_yaml()
                thisyamlfile = self.savefolder + nn_name + '.yaml'
                thish5file = self.savefolder + nn_name + '.h5'

                with open(thisyamlfile, "w") as yamlwrite:
                    yamlwrite.write(model_yaml)
                # serialize weights to HDF5
                thismodel.save_weights(thish5file)

            tranformfile = self.savefolder + 'NNR.transform'
            np.savetxt(tranformfile, np.vstack((self.feature_avgs, self.feature_vars)))

            savekeys = ['feature_names', 'nodecounts', 'activations', 'epochs', 'include_tpz_results', 'run_folder']

            with open(self.savefolder + 'info.txt', 'w') as writefile:
                for thiskey in savekeys:
                    writefile.write(thiskey + ': ' + repr(getattr(self, thiskey)) + '\n')

            np.savetxt(self.savefolder + 'results_train.dat', np.average(self.fit_features(self.features_train), axis = 1))
            np.savetxt(self.savefolder + 'results_validate.dat', np.average(self.fit_features(self.features_val), axis = 1))

            if os.path.isfile(self.run_folder + self.data_name + '.nnr_test'):
                np.savetxt(self.savefolder + 'results_test.dat', self.fit_file(self.run_folder + self.data_name + '.nnr_test'))




    def get_data(self, phot_feature_list):

        keep_feature_list = phot_feature_list
        if self.include_tpz_results:
            keep_feature_list = keep_feature_list + ['zphot', 'zconf', 'zerr']

        trainfile = read_csv(self.run_folder + self.data_name  + '.nnr_train', comment = '#', delimiter = '\s+')
        valfile = read_csv(self.run_folder + self.data_name  + '.nnr_validate', comment = '#', delimiter = '\s+')
        self.features_train = trainfile[keep_feature_list].to_numpy()
        self.features_val = valfile[keep_feature_list].to_numpy()

        self.feature_names = np.array(keep_feature_list)

        pz_train = np.squeeze(self.features_train.T[self.feature_names == 'zphot'])
        sz_train = trainfile['specz'].to_numpy()
        pz_err_train = np.squeeze(self.features_train.T[self.feature_names == 'zerr'])

        pz_val = np.squeeze(self.features_val.T[self.feature_names == 'zphot'])
        sz_val = valfile['specz'].to_numpy()
        pz_err_val = np.squeeze(self.features_val.T[self.feature_names == 'zerr'])

        self.bias_train = pz_train - sz_train
        self.bias_val = pz_val - sz_val



    def create_models(self):

        # Initializes untrained neural networks

        self.nnlist = []

        for x in range(self.splitnum):

            model = Sequential()
            model.add(Dense(self.nodecounts[0], activation = self.activations[0], input_shape = (self.features_train.shape[1],)))
            # model.add(Dropout(0.2))

            for thisnodenum, thisactivation in zip(self.nodecounts[1:], self.activations[1:]):

                if thisactivation == 'selu':
                    kernel_initializer = 'lecun_normal'
                else:
                    kernel_initializer = 'glorot_uniform'
                model.add(Dense(thisnodenum, activation = thisactivation, kernel_initializer = kernel_initializer))

            initial_learning_rate = 0.005
            lr_schedule = ExponentialDecay(
                initial_learning_rate,
                decay_steps=7000,
                decay_rate=0.1,
                staircase=False)

            # huber = Huber(delta = self.huber_delta, name = 'huber_loss')

            model.compile(loss = 'mse', optimizer = Adam(learning_rate = lr_schedule))

            self.nnlist.append(model)



    def preprocess(self, X):

        return (X - self.feature_avgs)/np.sqrt(self.feature_vars)



    def fit_file(self, fname, average_results = True):

        # with open(fname, 'r') as readfile:
        #     column_names = [thisname for thisname in readfile.readline()[:-1].split(' ') if (thisname != '#') and (thisname != '')]

        feature_file = read_csv(fname, comment = '#', delimiter = '\s+')
        features_fit = feature_file[self.feature_names].to_numpy()

        if average_results:
            return np.average((self.fit_features(features_fit)), axis = 1)
        else:
            return self.fit_features(features_fit)


    def fit_features(self, features):

        processed_features = self.preprocess(features)

        predictions = []

        for thismodel in self.nnlist:

            predictions.append(thismodel.predict(processed_features, verbose = 1))

        return np.squeeze(np.array(predictions).T)


    def fit_app(self):

        np.savetxt(self.savefolder + 'results_application.dat', self.fit_file(self.run_folder + self.data_name + '.nn_app'))






def plot_spectra_class(num_plots = 10, random = True, with_spectra = False, run_folder = './tpzruns/tpzrun_split10_eelfrac0.00_te_color_trip/', savefolder = None, randseed = 4123):

    np.random.seed(randseed)

    nn = nn_classifier(run_folder = run_folder)

    filter_lo = np.array([4110.0, 5476.8, 6980.8, 8540.0, 9380.0])
    filter_hi = np.array([5450.0, 6976.5, 8550.6, 9280.0, 10120.0])
    filter_centers = np.array([4814.203411956398, 6234.948738008778, 7760.555676214155, 8895.780451699658, 9796.179213282396])

    if savefolder == None:
        savefolder = run_folder + 'classification_plots/'

    if not os.path.isdir(savefolder):
        os.makedirs(savefolder)

    classifications = np.loadtxt(nn.savefolder + 'results.dat', dtype = float)

    indices = np.loadtxt(run_folder + 'results.dat', usecols = [0], dtype = int)
    specz, fitz, fitz_err = np.loadtxt(run_folder + 'results.dat', usecols = [2,3,4], dtype = float, unpack = True)

    mags = np.loadtxt(run_folder + 'mags.dat', usecols = [0,1,2,3,4], dtype = float)
    photometry = 10.**(-.4 * (mags + 48.6))
    
    g_mag ,r_mag ,i_mag ,z_mag ,y_mag = mags.T
    is_lrg = (z_mag > 20.41) & (r_mag - z_mag > (z_mag - 17.18)/2.) & (r_mag - z_mag > 0.9) & ((r_mag - z_mag > 1.15) | (g_mag - r_mag > 1.65))

    is_goodfit = nn.is_goodfit

    if num_plots == None:
        num_plots = len(indices)

    zcosmos_id, hsc_ind = np.loadtxt('./zCOSMOS/correlation.txt', usecols = [0,2], unpack = True, dtype = int)
    
    if with_spectra:
        select_indices = [x for x in range(len(indices)) if indices[x] in hsc_ind]

        indices = indices[select_indices]
        specz = specz[select_indices]
        fitz = fitz[select_indices]
        fitz_err = fitz_err[select_indices]
        photometry = photometry[select_indices]
        is_goodfit = is_goodfit[select_indices]
        is_lrg = is_lrg[select_indices]
        classifications = classifications[select_indices]

    if random:
        select_indices = sorted(np.random.choice(np.arange(len(indices)), size = num_plots, replace = False))
        indices = indices[select_indices]
        specz = specz[select_indices]
        fitz = fitz[select_indices]
        fitz_err = fitz_err[select_indices]
        photometry = photometry[select_indices]
        is_goodfit = is_goodfit[select_indices]
        is_lrg = is_lrg[select_indices]
        classifications = classifications[select_indices]
    else:
        indices = indices[:num_plots]
        specz = specz[:num_plots]
        fitz = fitz[:num_plots]
        fitz_err = fitz_err[:num_plots]
        photometry = photometry[:num_plots]
        is_goodfit = is_goodfit[:num_plots]
        is_lrg = is_lrg[:num_plots]
        classifications = classifications[:num_plots]

    hsc_data = fits.open('./HSC/HSC_wide_clean_pdr2.fits')[1].data
    # photometry = np.vstack((hsc_data['g_cmodel_flux'], hsc_data['r_cmodel_flux'], hsc_data['i_cmodel_flux'], hsc_data['z_cmodel_flux'], hsc_data['y_cmodel_flux'])).T[hsc_ind].astype(float) * 10**-23 * 10**-9 # Convert nJy to cgs
    phot_errs = np.vstack((hsc_data['g_cmodel_fluxsigma'], hsc_data['r_cmodel_fluxsigma'], hsc_data['i_cmodel_fluxsigma'], hsc_data['z_cmodel_fluxsigma'], hsc_data['y_cmodel_fluxsigma'])).T[indices].astype(float) * 10**-23 * 10**-9 # Convert nJy to cgs
    # specz = fits.open('./HSC/HSC_wide_clean_pdr2.fits')[1].data['specz_redshift']
    # photometry = photometry * 3every18/(filter_centers**2)

    fig = plt.figure(figsize = (8,8))
    sp = fig.add_subplot(111)

    for thisind, thisspecz, thisfitz, thisfitz_err, thisphot, thisphot_err, this_is_lrg, this_is_goodfit, thisclass in tqdm(zip(indices, specz, fitz, fitz_err, photometry, phot_errs, is_lrg, is_goodfit, classifications), total = len(indices)):

        spec_exists = thisind in hsc_ind

        if spec_exists:

            thisid = zcosmos_id[hsc_ind == thisind]

            if not hasattr(thisid, '__iter__'):
                wave, spec, spec_err = fits.open(glob('./zCOSMOS/*%09i*.fits' % thisid)[0])[1].data[0][0:3]
                spec = spec * wave**2 / 3e18
                sp.plot(wave, spec, color = 'C1', zorder = 0)

        if thisclass > 0.5 and this_is_goodfit:
            thiscolor = 'b'
        elif thisclass > 0.5 and not this_is_goodfit:
            thiscolor = 'r'
        elif thisclass < 0.5 and this_is_goodfit:
            thiscolor = 'g'
        else:
            thiscolor = 'k'

        sp.errorbar(filter_centers, thisphot, xerr = (filter_hi - filter_lo)/2., yerr = thisphot_err, fmt = 'none', ecolor = thiscolor, zorder = 1)

        sp.scatter(filter_centers, thisphot, s = 75, facecolors = 'None', edgecolors = thiscolor, linewidth = 2, zorder = 2)

        class_string = ''
        if this_is_lrg:
            class_string = 'LRG'
            if this_is_goodfit:
                class_string += ', GF'
        elif this_is_goodfit:
            class_string = 'GF'

        sp.text(0.02, 0.98, '$NN = %.4f$' % thisclass, ha = 'left', va = 'top', fontsize = 20, transform = sp.transAxes)
        sp.text(0.02, 0.90, r'$\frac{\sigma_{z_{fit}}}{1+z_{spec}} = ' + '%.3f$' % (thisfitz_err/(1.+thisspecz)), ha = 'left', va = 'top', fontsize = 20, transform = sp.transAxes)
        sp.text(0.02, 0.82, r'$\frac{\Delta z}{1 + z_{spec}} = ' + '%.3f$' % ((thisfitz - thisspecz)/(1.+thisspecz)), ha = 'left', va = 'top', fontsize = 20, transform = sp.transAxes)
        sp.text(0.02, 0.74, class_string, ha = 'left', va = 'top', fontsize = 20, transform = sp.transAxes)

        sp.text(0.98, 0.02, '$z_{TPZ} = %.2f$\n$z_{spec}=%.2f$' % (thisfitz, thisspecz), fontsize = 30, ha = 'right', va = 'bottom', transform = sp.transAxes)

        sp.set_yscale('log')
        sp.set_xscale('log')

        sp.set_ylim(np.median(thisphot)/100., np.median(thisphot)*100.)

        sp.set_xlabel('Wavelength (Angstrom)')
        sp.set_ylabel(r'Flux Density ($F_\nu$/cgs)')

        plt.savefig(savefolder + '%07i.png' % thisind, bbox_inches = 'tight')

        sp.cla()

    plt.close()




def plot_zdist(run_folder = './tpzruns/tpzrun_split10_eelfrac0.00_te_color_trip/'):

    nn = nn_classifier(run_folder = run_folder)

    classifications = np.loadtxt(nn.savefolder + 'results.dat', dtype = float)
    indices = np.loadtxt(run_folder + 'results.dat', usecols = [0], dtype = int)
    specz, fitz, fitz_err = np.loadtxt(run_folder + 'results.dat', usecols = [2,3,4], dtype = float, unpack = True)

    fig = plt.figure(figsize = (8,8))
    sp = fig.add_subplot(111)

    sp.hist(specz, range = (0,2), bins = 25, histtype = 'step', normed = False, color = 'k', linewidth = 2, label = 'All')
    sp.hist(specz[nn.is_goodfit], range = (0,2), bins = 25, histtype = 'step', normed = False, alpha = 0.5, color = 'C0', linewidth = 2, label = 'True Good Fit')
    sp.hist(specz[~nn.is_goodfit], range = (0,2), bins = 25, histtype = 'step', normed = False, alpha = 0.5, color = 'C1', linewidth = 2, label = 'True Non-Good Fit')

    sp.hist(specz[classifications > 0.5], range = (0,2), bins = 25, histtype = 'step', normed = False, color = 'C0', linewidth = 2, label = 'NN Good Fit')
    sp.hist(specz[classifications < 0.5], range = (0,2), bins = 25, histtype = 'step', normed = False, color = 'C1', linewidth = 2, label = 'NN Non-Good Fit')

    sp.set_xlabel('Redshift(z)')
    sp.set_ylabel('Normalized Frequency')

    sp.legend(loc = 'upper right')





def plot_acc_prec_hist(val = 'acc', run_folder = './tpzruns/tpzrun_split10_eelfrac0.00_te_color_trip/'):

    nn = nn_classifier(run_folder = run_folder)

    classifications = np.loadtxt(nn.savefolder + 'results.dat', dtype = float)
    indices = np.loadtxt(run_folder + 'results.dat', usecols = [0], dtype = int)
    specz, fitz, fitz_err = np.loadtxt(run_folder + 'results.dat', usecols = [2,3,4], dtype = float, unpack = True)

    if val == 'acc':
        histval = (fitz - specz)/(1. + specz)
        xlabel = r'$\frac{\Delta z}{1+z_{spec}}$'
        plotrange = (-.5,.5)
    elif val == 'absacc':
        histval = np.log10(np.abs(fitz - specz)/(1. + specz))
        xlabel = r'$\log_{10}\left(\frac{|\Delta z|}{1+z_{spec}}\right)$'
        plotrange = (-5,2)
    elif val == 'prec':
        histval = np.log10(fitz_err / (1. + specz))
        xlabel = r'$\log_{10}\left(\frac{\sigma z}{1+z_{spec}}\right)$'
        plotrange = (-5,2)

    fig = plt.figure(figsize = (8,8))
    sp = fig.add_subplot(111)

    sp.hist(histval, range = plotrange, bins = 50, histtype = 'step', normed = False, color = 'k', linewidth = 2, label = 'All')
    sp.hist(histval[nn.is_goodfit], range = plotrange, bins = 50, histtype = 'step', normed = False, alpha = 0.5, color = 'C0', linewidth = 2, label = 'True Good Fit')
    sp.hist(histval[~nn.is_goodfit], range = plotrange, bins = 50, histtype = 'step', normed = False, alpha = 0.5, color = 'C1', linewidth = 2, label = 'True Non-Good Fit')

    sp.hist(histval[classifications > 0.5], range = plotrange, bins = 50, histtype = 'step', normed = False, color = 'C0', linewidth = 2, label = 'NN Good Fit')
    sp.hist(histval[classifications < 0.5], range = plotrange, bins = 50, histtype = 'step', normed = False, color = 'C1', linewidth = 2, label = 'NN Non-Good Fit')

    sp.set_xlabel(xlabel)
    sp.set_ylabel('Normalized Frequency')

    sp.legend(loc = 'upper right')





def plot_prec_acc(nn, num_points = 1000, nn_boundary = 0.5, run_folder = './tpzruns/tpzrun_split10_eelfrac0.00_te_color_trip/', randseed = 6384):

    indices = np.loadtxt(nn.run_folder + 'results.dat', usecols = [0], dtype = int)
    specz, fitz, fitz_err = np.loadtxt(nn.run_folder + 'results.dat', usecols = [2,3,4], dtype = float, unpack = True)

    classifications = np.loadtxt(nn.savefolder + 'results.dat', dtype = float)

    is_goodfit = nn.is_goodfit

    if nn.include_tpz_results and nn.prec_cutoff == 0.02:
        lim_inds = fitz_err / (1.+specz) < 0.02
        indices = indices[lim_inds]
        specz = specz[lim_inds]
        fitz = fitz[lim_inds]
        fitz_err = fitz_err[lim_inds]


    if num_points == None:
        num_points = len(indices)
    select_indices = sorted(np.random.choice(np.arange(len(indices)), size = num_points, replace = False))
    indices = indices[select_indices]
    specz = specz[select_indices]
    fitz = fitz[select_indices]
    fitz_err = fitz_err[select_indices]
    is_goodfit = is_goodfit[select_indices]
    classifications = classifications[select_indices]

    fig = plt.figure(figsize = (8,8))
    sp = fig.add_subplot(111)

    for thisspecz, thisfitz, thisfitz_err, this_is_goodfit, thisclass in tqdm(zip(specz, fitz, fitz_err, is_goodfit, classifications), total = len(specz)):

        x = np.abs(thisfitz - thisspecz)/(1. + thisspecz)
        y = (thisfitz_err)/(1. + thisspecz)

        if thisclass > nn_boundary and this_is_goodfit:
            thiscolor = 'b'
        elif thisclass > nn_boundary and not this_is_goodfit:
            thiscolor = 'r'
        elif thisclass < nn_boundary and this_is_goodfit:
            thiscolor = 'g'
        else:
            thiscolor = 'k'

        sp.scatter(x, y, marker = '.', c = thiscolor)

    sp.plot([0,nn.acc_cutoff], [nn.prec_cutoff,]*2, color = 'k')
    sp.plot([nn.acc_cutoff,]*2, [0,nn.prec_cutoff], color = 'k')

    sp.text(0.02, 0.98, '%.2f' % nn_boundary, fontsize = 30, ha = 'left', va = 'top', transform = sp.transAxes)

    sp.set_xscale('log')
    sp.set_yscale('log')

    sp.set_xlim(10**-5, 10**0.5)
    sp.set_ylim(10**-3, 10**0)

    sp.set_xlabel(r'$\frac{\Delta z}{1+z_{spec}}$')
    sp.set_ylabel(r'$\frac{\sigma_{TPZ}}{1+z_{spec}}$')







def plot_feature_weights(nn, norm = 'l2'):

    feature_weights = [[] for x in range(len(nn.nnlist))]

    fig = plt.figure(figsize = (9,9))
    sp = fig.add_subplot(111)

    for x, thisnn in enumerate(tqdm(nn.nnlist)):

        for thisfeatureweight in thisnn.weights[0].numpy():

            if norm == 'l2':
                feature_weights[x].append(np.sum(thisfeatureweight**2))
            elif norm == 'l1':
                feature_weights[x].append(np.sum(np.abs(thisfeatureweight)))
        
    feature_weights = np.array(feature_weights)

    for thisnn_weights in feature_weights:

        sp.bar(range(len(thisnn_weights)), thisnn_weights, 1, color = 'None', edgecolor = 'k', alpha = 0.3)

    sp.bar(range(len(nn.feature_names)), np.average(feature_weights, axis = 0), 1, color = 'None', edgecolor = 'white', linewidth = 3)
    sp.bar(range(len(nn.feature_names)), np.average(feature_weights, axis = 0), 1, tick_label = nn.feature_names, color = 'None', edgecolor = 'k', linewidth = 2)

    sp.set_xlabel('Features')
    sp.set_ylabel('Vector %s Norm' % norm.upper())




