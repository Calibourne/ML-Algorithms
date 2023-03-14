from DesicionTree import DecisionTree
from ClassifierNode import ClassifierNode
from pandas import DataFrame
from numpy import int64
from ScratchML.consts import cls_x, cls_y, labelName


class DTClassifier(DecisionTree):
    """
    A Classification Tree Definition
    
    Args:
        DecisionTree: the parent class
    """
    def __init__(self, data, maxDepth, minSample):
        """
        Classifier Constructor

        Args:
            data (pd.DataFrame): the data of the classifier
            maxDepth (int): the max depth of the classifier
            minSample (int): the minimum samples in a node in order to split
        """
        
        super().__init__(maxDepth, minSample)
        
        self.root = ClassifierNode(data)
            
    def _split(self, node: ClassifierNode):
        """
        Expansion on the _split(node) method of DecisionTree.
            Splits the node whether it is possible.
        
        Args:
            node (ClassifierNode): the node to split
        """
        
        dfA, dfB = super()._split(node)

        if len(dfA) > 0 and len(dfB) > 0:
            node.right = ClassifierNode(data=dfA,depth=node.depth+1)
            node.left = ClassifierNode(data=dfB,depth=node.depth+1)
        else:
            node.make_leaf()
            
        
    def _fit(self, node: ClassifierNode):
        """Expansion on the _fit(node) method of DecisionTree 
            to build the classifier

        Args:
            node (ClassifierNode): the Classifier node
        """
        
        if self._can_split(node) and node.gini != 0:
            super()._fit(node)

        else:
            if node is not None:
                classA = node.count_class_num(cls_x)
                if not isinstance(classA, int) and not isinstance(classA, int64):
                    classA = len(classA)
                classB = len(node.data)-classA
                if classA > classB:
                    node.predictedClass = cls_x
                else:
                    node.predictedClass = cls_y
        if node is not None:
            self.graph.node(node.id, node.__str__())
    
    
    def _predict(self, node: ClassifierNode ,samp):
        """
        Predict the samp value

        Args:
            node (ClassifierNode): the root
            samp (list): parameters list

        Returns:
            str: the predicted class
        """
        
        if node.right is None and node.left is None :
            return node.predictedClass

        return super()._predict(node, samp)


    def evaluate(self, data: DataFrame):
        """
        Evaluate the Classifier

        Args:
            data (DataFrame): the data to be evaluated

        Returns:
            dict: with accuracy, sensitivity, specificity, precision
        """

        TP = 0 # TRUE POSITIVE
        TN = 0 # TRUE NEGATIVE
        FP = 0 # FALSE POSITIVE 
        FN = 0 # FALSE NEGATIVE
        # y_pred = [
        #     self.predict(data.iloc[i])
        #     for i in range(0,len(data))
        # ]
        for i in range(len(data)):
            s = data.iloc[i]
            predicted_value = self.predict(s)
            
            if s[labelName] == predicted_value:
                if predicted_value == cls_x:
                    TP+=1 
                else:
                    TN+=1 
            else:
                if predicted_value == cls_x:
                    FP+=1 
                else:
                    FN+=1 

        metrics = {}
        metrics["accuracy"] = (TP+TN)/(TP+TN+FP+FN)
        metrics["sensitivity"] = TP/(TP+FN)
        metrics["specificity"] = TN/(TN+FP)
        metrics["precision"] = TP/(TP+FP)

        return metrics
