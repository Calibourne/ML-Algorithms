from DecisionNode import DecisionNode
from pandas import DataFrame
from numpy import unique
from ScratchML.consts import cls_x, cls_y, labelName, labelIndex

class ClassifierNode(DecisionNode):
    """
    Classifier Node Definition

    Args:
        DecisionNode : The parent class
    """
    def __init__(self,data: DataFrame, left=None, right=None, depth:int=0):
        """
        Classifier Node Constructor
        
        Args:
            data (DataFrame): the node data.
            left (ClassifierNode, optional): the left child node. Defaults to None.
            right (ClassifierNode, optional): the right child node. Defaults to None.
            depth (int, optional): The depth of the node. Defaults to 0.
        """
        
        super().__init__(data, left, right, depth)
        
        self.gini = 1 # uncalibrated gini value
        self.predictedClass = ""
    
    
    def giniScore(self):
        """
        Calculate the gini score of the current node

        Returns:
            float: the gini score
        """
        
        totalSamples = len(self.data)
        classA = self.count_class_num(cls_x)
        pA = classA/totalSamples
        if totalSamples==0:
            return 1 
        return 2*pA(1-pA)
    
    
    def count_class_num(self,className):
        """
        Count the number of samples of a certain class in the current node

        Args:
            className (str): the class name

        Returns:
            int: the number of occurrences of the class
        """
        
        try:
            return len(self.data[self.data[:,labelIndex] == className])
        
        #if there are no samples from the requested class(a pure split)
        except Exception:
            return 0


    # threshold is used for numeric feature         
    def calc_gini_impurity(self,feature_index:int, threshold=None):
        """
        Calculate the gini impurity factor

        Args:
            feature_index (int): the feature index we calcu
            threshhold (int | float): the axis we divide the data with

        Returns:
            float: the gini impurity factor
        """
        
        # If the feature index is categorial
        if threshold is None:
            values = unique(self.data[:,feature_index])
            r = self.data[self.data[:,feature_index] == values[0]]
            l = self.data[self.data[:,feature_index] == values[1]]
            
        # Else, if the feature index is numerical
        else:
            r = self.data[self.data[:,feature_index] >= threshold]
            l = self.data[self.data[:,feature_index] < threshold]
            
        lP = len(l[l[:,labelIndex] == cls_y])
        rP = len(r[r[:,labelIndex] == cls_y])

        # if there is a clean split (all samples in one child node)
        if len(r)==0 or len(l)==0:
            return float('inf')
        
        p1 = lP/len(l)
        p2 = rP/len(r)

        # calculate the gini impurity
        
        giniL = 2*p1*(1-p1)
        giniR = 2*p2*(1-p2)

        total = len(r) + len(l)
        
        totalGini = (len(r)*giniR + len(l)*giniL)/total
        
        return totalGini
    
    
    
    def calc_gini_numeric_feature(self, feature_index: int):
        """
        Calculate the gini impurity for numeric feature

        Args:
            feature_index (int): the numeric feature index

        Returns:
            float: the gini impurity factor
        """

        # Sort the values of the new dataframe, increasing by the values of the nomericFeature column
        self.data[self.data[:, feature_index].argsort()]

        best_gini, best_threshold = 1, float('inf')

        # run on any adjacent values and find the best split axis
        for i in range(0, len(self.data)-1) :
            threshold = (self.data[i][feature_index]+self.data[i+1][feature_index])/2
            gini = self.calc_gini_impurity(feature_index, threshold)
            if gini < best_gini:
                best_gini = gini
                best_threshold = threshold

        return best_gini, best_threshold
     
    
    def _bestSplit(self):
        """
        Perform a node calibration, 
            and find the best split axis for the current node
        """
        
        min_gini = 1
        
        for i in range (len(self.data_labels)):
        
            # if self.data_labels[i]!=labelName :
            if i!=0 and self.data_labels[i]!=labelName :
                
                #in case that the feature is categorial
                if len(unique(self.data[:,i]))==2 : 
                    self.featureVal = 0
                    gini = self.calc_gini_impurity (i, None)
                    if gini < min_gini:
                        min_gini = gini
                        self.featureName = self.data_labels[i]
                        self.featureIndex = i
                        self.featureVal = 1
                        self.gini = gini
                        
                #in case that the feature is numeric        
                else :
                    numricGini, threshold = self.calc_gini_numeric_feature(i)
                    if numricGini < min_gini:
                        min_gini = numricGini
                        self.featureName = self.data_labels[i]
                        self.featureIndex = i
                        self.featureVal = threshold
                        self.gini = numricGini
        
    
    def make_leaf(self):
        """
        Make a leaf out of current Node
        """
        pA, pB = self.count_class_num(cls_x), self.count_class_num(cls_y)
        self.predictedClass = cls_x if pA > pB else cls_y
        
    
    def __str__(self):
        """
        String Representation

        Returns:
            str: the representation
        """
        return f"Predicting: {self.predictedClass}" if self.predictedClass!="" else f"Spliting on\n{self.featureName}={self.featureVal}"
