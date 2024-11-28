#code credits: https://github.com/MesserLab/GeneDriveForSuppressionOfInvasiveRodents

import pandas as pd
import numpy as np
import torch
import gpytorch
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood, FixedNoiseGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda import is_available as cuda_available, empty_cache
from SALib.sample import sobol as sobolSample
from SALib.analyze import sobol
from copy import deepcopy

#import warnings
#warnings.filterwarnings("ignore")


def getColumnIndex(x): #Dictionary to obtain column index from data
    #assumed data format for conditional simulations: 
     
    #column 0: simRound
    #column 1: simID
    #column 2: alphaRest
    #column 3: alphaAmp
    #column 4: alphaShift
    #column 5: infTicksCount
    #column 6: avgVisitsCount
    #column 7: pVisits
    #column 8: propSocialVisits
    #column 9: locPerSGCount
    #column 10: maxIncidence
    #column 11: epidemicSize
    #column 12: duration (log transformed)
    #column 13: sd_maxIncidence
    #column 14: sd_epidemicSize
    #column 15: sd_duration (log transformed)

    #assumed data format for epidemic probability: 
    #column 0: simRound
    #column 1: simID
    #column 2: alphaRest
    #column 3: alphaAmp
    #column 4: alphaShift
    #column 5: infTicksCount
    #column 6: avgVisitsCount
    #column 7: pVisits
    #column 8: propSocialVisits
    #column 9: locPerSGCount
    #column 10: establishment

    cases = {
        "simRound" : 0,
        "simID": 1,
        "firstInputParam": 2,
        "lastInputParam" : 9,
        "maxIncidence" : 10,
        "epidemicSize": 11,
        "duration" : 12,
        "establishment": 10,
        "sd_maxIncidence": 13,
        "sd_epidemicSize": 14,
        "sd_duration": 15
    }
    return cases.get(x,0)

class GPRegressionModel(ExactGP):
    """
    The gpytorch model underlying the SIR_GP class.
    """
    def __init__(self, train_x, train_y, likelihood):
        """
        Constructor that creates objects necessary for evaluating GP.
        """
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        # The Mattern Kernal is particularly well suited to models with abrupt transitions between success and failure.
        x_from = getColumnIndex("firstInputParam")
        x_to = getColumnIndex("lastInputParam")
        xCount = x_to - x_from + 1
        self.covar_module = ScaleKernel(MaternKernel(nu=0.5, ard_num_dims=(xCount)))
    def forward(self, x):
        """
        Takes in nxd data x and returns a MultivariateNormal with the prior mean and covariance evaluated at x.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return MultivariateNormal(mean_x, covar_x)


class SIR_GP():
    """
    Class for the SIR GP model.
    """
    def __init__(self, training_data, model_type): #model types: maxIncidence, epidemicSize, duration, establishment
        data = pd.read_csv(training_data, sep='\t')
        self.model_type  = model_type

        data = data.to_numpy(dtype="float")
        x_from = getColumnIndex("firstInputParam")
        x_to = getColumnIndex("lastInputParam")

        self.train_x = torch.from_numpy(data[:, x_from:(x_to + 1)]).float().contiguous() #The model input parameters in the training set.

        if self.model_type not in ["maxIncidence", "epidemicSize", "duration", "establishment"]:
            raise ValueError("Specify a model_type of \"maxIncidence\", \"epidemicSize\", \"duration\", or \"establishment\".")
        
        y_id = getColumnIndex(model_type) 
        self.train_y = torch.from_numpy(data[:,y_id:(y_id+1)]).float().contiguous().flatten()

        if self.model_type in ["maxIncidence", "epidemicSize", "duration"]:
            noise_id = getColumnIndex(f"sd_{model_type}")
            self.y_noise = torch.from_numpy(data[:,noise_id:(noise_id+1)]).float().contiguous().flatten()
            self.y_noise = self.y_noise * self.y_noise #FixedNoiseGaussianLikelihood() takes the variance as input 
            if cuda_available():
                self.y_noise = self.y_noise.cuda()
            self.likelihood = FixedNoiseGaussianLikelihood(self.y_noise, learn_additional_noise=False)
        else:
            self.likelihood = GaussianLikelihood()
        
        self.model = GPRegressionModel(self.train_x, self.train_y, self.likelihood)

        if cuda_available():
            self.train_x, self.train_y, self.likelihood, self.model = self.train_x.cuda(), self.train_y.cuda(), self.likelihood.cuda(), self.model.cuda()

        self.default_params = {
                'alphaRest': 0.01,
                'alphaAmp': 0.5,
                'alphaShift': 0,
                'infTicksCount' : 5,
                'avgVisitsCount' : 1,
                'pVisits' : 0.5,
                'propSocialVisits' : 0.5,
                'locPerSGCount' : 5
        }

        self.param_ranges = {
                'alphaRest': (0, 0.03),
                'alphaAmp': (0, 1),
                'alphaShift': (0, 1),
                'infTicksCount' : (4, 6),
                'avgVisitsCount' : (1, 5),
                'pVisits' : (0.05, 0.95),
                'propSocialVisits' : (0, 1),
                'locPerSGCount' : (1, 20)
        }

        x_from = getColumnIndex("firstInputParam")
        x_to = getColumnIndex("lastInputParam")
        num_params = x_to - x_from + 1
        self.sa_params_dict = {
                "num_vars": num_params,
                "names": [k for k, v in self.param_ranges.items()][:(num_params)],
                "bounds": [v for k, v in self.param_ranges.items()][:(num_params)]
        }

    def save(self, filename, cpu=False):
        """
        Saves the trained GP model.
        """
        model_to_save = self.model
        if cpu:
            model_to_save = self.model.cpu()
        torch.save(model_to_save.state_dict(), f"{filename}.pth")
        if cpu:
            if cuda_available():
                self.model.cuda()
        print("Model saved.")

    def load(self, filename):
        """
        Loads a pre-trained GP model.
        """
        try:
            self.model.load_state_dict(torch.load(f"{filename}.pth"))
        except FileNotFoundError:
            try:
                self.model.load_state_dict(torch.load(filename))
            except FileNotFoundError:
                raise FileNotFoundError(f"{filename} not found.")
        trainingLoss = self.train(1)
        print(f"Model loaded. Loss: {trainingLoss}")

    def train(self, num_iterations, learning_rate = 0.01):
        """
        Train the model.
        """
        self.model.train() # Set the model to training mode.
        self.likelihood.train()

        optimizer = torch.optim.Adam([{'params': self.model.parameters()},], lr=learning_rate) # Using the adam optimizer
        mll = ExactMarginalLogLikelihood(self.likelihood, self.model) # "Loss" for GPs: the marginal log likelihood
        if num_iterations >= 100:
            print(f"Training for {num_iterations} iterations:")
        
        # Training loop:
        with gpytorch.settings.max_cg_iterations(16000):
            for i in range(num_iterations):
                optimizer.zero_grad() # Zero gradients from previous iteration
                output = self.model(self.train_x) # Output from model
                loss = -mll(output, self.train_y) # Calc loss and backprop gradients
                loss.backward()
                if (i+1) % 100 == 0:
                    print(f"Iter {i + 1}/{num_iterations} - Loss: {loss.item()}")
                if ((i+1) == num_iterations):
                    trainingLoss = loss.item()
                optimizer.step()
                if cuda_available():
                    empty_cache()

        self.model.eval() # Set the model to evaluation mode.
        self.likelihood.eval()
        return trainingLoss
    
    def predict(self, test_data):
        """
        Predicts y values, lower, and upper confidence for a data set.
        """
        data = pd.read_csv(test_data, sep='\t')
        data = data.to_numpy(dtype="float")
        x_from = getColumnIndex("firstInputParam")
        x_to = getColumnIndex("lastInputParam")
        x = torch.from_numpy(data[:,x_from:(x_to + 1)]).float().contiguous()
        
        if self.model_type == "establishment":
            return self.predict_ys(x)
        else:
            noise_id = getColumnIndex(f"sd_{self.model_type}")
            y_noise = torch.from_numpy(data[:,noise_id:(noise_id+1)]).float().contiguous().flatten()
            y_noise = y_noise * y_noise #FixedNoiseGaussianLikelihood() takes the variance as input 
        return self.predict_ys(x, y_noise)

    def predict_ys(self, parsed_data, y_noise = torch.tensor([-1.0])):
        """
        Predicts y values from X values.
        Takes parsed data as a contiguous torch tensor.
        """
        no_noise = torch.tensor([-1.0])
        if cuda_available():
            parsed_data = parsed_data.cuda()
            y_noise = y_noise.cuda()
            no_noise = no_noise.cuda()

        if torch.equal(y_noise, no_noise):
            loader = DataLoader(TensorDataset(parsed_data), batch_size=1024,shuffle=False)
        else:
            loader = DataLoader(TensorDataset(parsed_data, y_noise), batch_size=1024, shuffle=False)

        mean, lower, upper = torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.])
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for batch in loader:
                if torch.equal(y_noise, no_noise):
                    observed_pred = self.likelihood(self.model(batch[0]))
                else:
                    observed_pred = self.likelihood(self.model(batch[0]), noise = batch[1])
                cur_mean = observed_pred.mean
                if cuda_available():
                    mean = torch.cat([mean, cur_mean.cpu()])
                    cur_lower, cur_upper = observed_pred.confidence_region()
                    lower = torch.cat([lower, cur_lower.cpu()])
                    upper = torch.cat([upper, cur_upper.cpu()])
                else:
                    mean = torch.cat([mean, cur_mean])
                    cur_lower, cur_upper = observed_pred.confidence_region()
                    lower = torch.cat([lower, cur_lower])
                    upper = torch.cat([upper, cur_upper])
        return mean[1:], lower[1:], upper[1:]

    def get_rmse(self, test_data): #calculate RMSE
        if test_data == None:
            raise ValueError("Specify a test set to check.")
        
        predicted_mean, _, _ = self.predict(test_data) #obtain predicted value 
        
        data = pd.read_csv(test_data, sep='\t') #obtain observed value 
        data = data.to_numpy(dtype="float")
        y_id = getColumnIndex(self.model_type)
        target = torch.from_numpy(data[:,y_id:(y_id+1)]).float().contiguous().flatten()
        rmse= np.sqrt( 1/len(target) * sum( (target.numpy() - predicted_mean.numpy())**2 ) )
        return rmse
    
    def samplePoints(self, candidates, N, p): #sample additional data points
        """
        Samples N additional training data points from a numpy array 
        
        candidates = candidate points 
        N = sample size 
        p = proportion of weights that will be scaled by the predicted output  
        """

        x = torch.from_numpy(candidates).float().contiguous() #obtain predictions 
        if cuda_available():
            x = x.cuda()
        pred_mean, pred_lower, pred_upper = self.predict_ys(x)
        pred_mean = pred_mean.numpy()
        pred_width = pred_upper - pred_lower 
        pred_width = pred_width.numpy()
        
        NP = int(np.floor(N * p)) #number of samples scaled by the predicted output 
        NnoP = N - NP #number of samples not scaled by predicted output
        sampleID = [] #book-keeping 

        if NnoP > 0 : 
            w = np.array(pred_width) / np.sum(pred_width) #calculate probabilities 
            tempSampleID = np.random.choice(np.arange(np.shape(candidates)[0]), replace=False, size=NnoP, p=w) #sample 
            sampleID = np.append(sampleID, tempSampleID)

        if NP > 0:
            if self.model_type == "duration":
                weights = np.clip(pred_mean, 0, 3) #clip predictions
                weights = (weights * (3 - weights)) + 1/len(weights) #calculate weights
            else:
                weights = np.clip(pred_mean, 0, 1) #clip out-of-range predictions 
                weights = (weights * (1 - weights)) + 1/len(weights) #calculate weights 

            w = np.array( (pred_width * weights) / np.sum(pred_width * weights) ) #scale to probabilities 

            counter = 0
            while counter < NP:
                tempSampleID = np.random.choice(np.arange(np.shape(candidates)[0]), size = 1, p=w) #sample 
                if np.isin(element=tempSampleID, test_elements=sampleID):
                    continue
                sampleID = np.append(sampleID, tempSampleID)
                counter = counter + 1
                
        sampleID = sampleID.astype(np.int32)
        return sampleID

    def sensitivity_analysis(self, pow2sampleSize=10, param_ranges=None, verbose=False): 
        """
        Perform a sensitivity analysis. Print the analysis if verbose=True.
        Returns a list of 3 pandas dataframes, where
        the first entry in the list is total effects, the second entry is first order, and the third entry is second order effects.
        """
        sa_params = deepcopy(self.sa_params_dict)

        if param_ranges:
            for key in param_ranges:
                if key not in self.default_params:
                    print(f"\"{key}\" not a valid parameter name. Ignoring.")
                if key in sa_params["names"]:
                    sa_params["bounds"][sa_params["names"].index(key)] = param_ranges[key]

        for i in range(len(sa_params["bounds"])):
            if type(sa_params["bounds"][i]) is not list and type(sa_params["bounds"][i]) is not tuple:
                sa_params["bounds"][i] = (sa_params["bounds"][i], sa_params["bounds"][i] + 0.00000001)
        

        param_values = sobolSample.sample(problem = sa_params, N=2**pow2sampleSize)
        print(f"Points to be evaluated: {len(param_values)}")

        x_from = getColumnIndex("firstInputParam")
        x_to = getColumnIndex("lastInputParam")
        x_count = x_to - x_from + 1
        x = np.zeros((len(param_values), (x_count) )) # Evaluate the model at sampled points
        for i in range(len(param_values)):
            x[i] = param_values[i]

        x = torch.from_numpy(x).float().contiguous()
        if cuda_available():
            x = x.cuda()
        y, _, _ = self.predict_ys(x)
        y = y.numpy()
        print("Fished predictions. Starting sensitivity analysis.")
        # Perform the sensitivity analysis:
        sa = sobol.analyze(sa_params, y, print_to_console=verbose)
        sa_df = sa.to_df()
        sa_df[0].columns = [c.replace('ST', 'Total Effects') for c in sa_df[0].columns]
        sa_df[1].columns = [c.replace('S1', 'First Order') for c in sa_df[1].columns]
        sa_df[2].columns = [c.replace('S2', 'Second Order') for c in sa_df[2].columns]
        sa_df.append(f"{self.model_type}")
        return sa_df
