
from pandas import DataFrame
from numpy import mean, sum, square, subtract, unique, ndarray

from ScratchML.consts import labelNumericIndex, labelNumericName, labels_rgr
from ScratchML.DecisionTrees._abstract import DecisionNode, DecisionTree


class RegressionNode(DecisionNode):
    """
    Regression Node Definition

    Args:
        DecisionNode : The parent class
    """
    def __init__(self,data: DataFrame, left=None, right=None, depth:int=0):
        """
        Regression Node Constructor
        
        Args:
            data (DataFrame): the node data.
            left (ClassifierNode, optional): the left child node. Defaults to None.
            right (ClassifierNode, optional): the right child node. Defaults to None.
            depth (int, optional): The depth of the node. Defaults to 0.
        """
        
        super().__init__(data, left, right, depth)
        
        self.ssr = 0 # Uncalibrated ssr value
        
        self.predictedVal = mean(self.data[:, labelNumericIndex])


    # This function choose the best avg to split by choosing avg that gives the lowest ssr
    def calc_ssr_to_feature (self, feature_index:int):
        """
        Calculate the ssr for a feature 
        Args:
            feature_index (int): the feature index

        Returns:
            float: the calculated ssr value
        """


        # If the feature is categorial
        if len(unique(self.data[:,feature_index]))<=2:
            
            # if only one category, don't split
            if len(unique(self.data[:,feature_index]))<2:
                min_ssr = float('inf')
                best_avarage = float('inf')
            
            else:
                values = unique(self.data[:,feature_index]) # return array with thw two optional values
                class_right = self.data[self.data[:,feature_index] == values[0]]
                class_left = self.data[self.data[:,feature_index] == values[1]]
                right_ssr = self.calc_ssr(class_right)
                left_ssr = self.calc_ssr(class_left)
                feature_ssr = right_ssr + left_ssr
                min_ssr = feature_ssr

                best_avarage = 0
        else:
            
            # Sort the values of the new dataframe by the feature index column
            self.data[self.data[:,feature_index].argsort()]

            min_ssr = float('inf')
            best_avarage = float('inf')

            for i in range(0, len(self.data)-1) :
                
                avarage = (self.data[i][feature_index]+self.data[i+1][feature_index])/2
                class_right = self.data[self.data[:,feature_index] >= avarage]
                class_left = self.data[self.data[:,feature_index] < avarage]
                right_ssr = self.calc_ssr(class_right)
                left_ssr = self.calc_ssr(class_left)
                feature_ssr = right_ssr + left_ssr
                
                if feature_ssr < min_ssr:
                    min_ssr = feature_ssr
                    best_avarage = avarage

        return min_ssr, best_avarage

    def calc_ssr(self, data: ndarray):
        """
        Calculate the ssr of the current data

        Args:
            data (np.ndarray): the data

        Returns:
            float: the ssr value
        """

        label_values = data[ : , labelNumericIndex]

        avg = []
        
        [
            avg.append(mean(label_values) if len(label_values) > 0 else 0)
        ]
        
        ssr = sum((label_values - avg)**2)
        
        return ssr

    def _bestSplit(self):
        """
        Perform a node calibration, 
            and find the best split axis for the current node
        """

        min_ssr = float('inf')

        for i in range (len(self.data_labels)):

            if i!=0 and self.data_labels[i]!=labelNumericName :

                ssr_score, best_avg = self.calc_ssr_to_feature(i)

                if ssr_score < min_ssr:
                    min_ssr = ssr_score
                    self.ssr = ssr_score
                    self.featureName = self.data_labels[i]
                    self.featureIndex = i
                    self.featureVal = best_avg
    
    def __str__(self):
        return f"Predicting: {round(self.predictedVal,3)}" if self.left is None and self.right is None else f"Spliting on\n{self.featureName}={self.featureVal}"

class DTRegressor(DecisionTree):
    """
    A Regression Tree Definition
    
    Args:
        DecisionTree: the parent class
    """
    
    def __init__(self, data: DataFrame, maxDepth, minSample):
        """
        Regressor Constructor

        Args:
            data (pd.DataFrame): the data of the regressor
            maxDepth (int): the max depth of the regressor
            minSample (int): the minimum samples in a node in order to split
        """
        
        super().__init__(maxDepth, minSample)
        
        self.root = RegressionNode(data)

    
    def _split(self, node: RegressionNode):
        """Expansion on the _split(node) method of DecisionTree.
            Splits the node whether it is possible.
        
        Args:
            node (ReggressionNode): the node to split
        """

        dfA, dfB = super()._split(node)
        if len(dfA) > 0 and len(dfB) > 0: # in case we split the data correctly
            
            node.right = RegressionNode(dfA,None,None, node.depth+1)
            node.left = RegressionNode(dfB, None,None, node.depth+1)

    
    def _fit(self, node: RegressionNode):
        """Expansion on the _fit(node) method of DecisionTree 
            to build the regressor

        Args:
            node (ReggressionNode): the Classifier node
        """
        
        if self._can_split(node):
            super()._fit(node)
        if node is not None:
            self.graph.node(node.id, node.__str__())


    def _predict(self, node: RegressionNode ,samp):
        """
        Predict the samp value

        Args:
            node (ReggressionNode): the root
            samp (list): parameters list

        Returns:
            float: the predicted value
        """

        if (node.right is None) and (node.left is None) :

            return node.predictedVal

        return super()._predict(node, samp)
