import numpy as np
from sklearn.metrics import confusion_matrix


class RunningConfusionMatrix():
    """Running Confusion Matrix class that enables computation of confusion matrix
    on the go and has methods to compute such accuracy metrics as Mean Intersection over
    Union MIOU.
    
    Attributes
    ----------
    labels : list[int]
        List that contains int values that represent classes.
    overall_confusion_matrix : sklean.confusion_matrix object
        Container of the sum of all confusion matrices. Used to compute MIOU at the end.
    ignore_label : int
        A label representing parts that should be ignored during
        computation of metrics
        
    """
    
    def __init__(self, labels, ignore_label=255):
        
        self.labels = labels
        self.ignore_label = ignore_label
        self.overall_confusion_matrix = None
        
    def update_matrix(self, ground_truth, prediction):
        """Updates overall confusion matrix statistics.
        If you are working with 2D data, just .flatten() it before running this
        function.
        Parameters
        ----------
        groundtruth : array, shape = [n_samples]
            An array with groundtruth values
        prediction : array, shape = [n_samples]
            An array with predictions
        """
        
        # Mask-out value is ignored by default in the sklearn
        # read sources to see how that was handled
        # But sometimes all the elements in the groundtruth can
        # be equal to ignore value which will cause the crush
        # of scikit_learn.confusion_matrix(), this is why we check it here
        if (ground_truth == self.ignore_label).all():
            
            return
        
        current_confusion_matrix = confusion_matrix(y_true=ground_truth,
                                                    y_pred=prediction,
                                                    labels=self.labels)

        self.overall_confusion_matrix = current_confusion_matrix
    
    def compute_current_mean_intersection_over_union(self,smooth=1e-5):
        
        intersection = np.diag(self.overall_confusion_matrix)
        ground_truth_set = self.overall_confusion_matrix.sum(axis=1)
        predicted_set = self.overall_confusion_matrix.sum(axis=0)
        union =  ground_truth_set + predicted_set - intersection

        intersection_over_union = (intersection + smooth ) / (union.astype(np.float32) + smooth)
        mean_intersection_over_union = np.mean(intersection_over_union)
        
        return mean_intersection_over_union




class RunningDice():
    """
    Attributes
    ----------
    labels : list[int]
        List that contains int values that represent classes.
    ignore_label : int
        A label representing parts that should be ignored during
        computation of metrics
        
    """
    
    def __init__(self, labels, ignore_label=None):
        
        self.labels = labels
        self.ignore_label = ignore_label

    
    def compute_total_dice(self, ground_truth, prediction):
        """
        Parameters
        ----------
        groundtruth: An array with groundtruth values, shape = [N,H,W]
        prediction : An array with predictions, shape = [N,H,W]
        """
        
        total_dice = 0.
        for i in self.labels:
            if i != self.ignore_label:
                dice = self.compute_binary_dice((ground_truth==i).astype(np.float32),
                                               (prediction==i).astype(np.float32))
                
                total_dice += dice

        return total_dice / len(self.labels)


    def compute_binary_dice(self,true, pred, smooth=1e-5):
        assert true.shape == pred.shape
        true =  true.flatten().reshape(true.shape[0],-1)
        pred =  pred.flatten().reshape(pred.shape[0],-1)

        inter = np.sum(true * pred, axis=1)
        union = np.sum(true + pred, axis=1)

        dice = (2*inter + smooth) / (union + smooth)
        return dice.mean()
