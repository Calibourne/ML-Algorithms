from pandas import DataFrame
class DecisionNode:
    """
    Abstract Decision Node Definition
    """
    cnt = 0
    def __init__(self, data: DataFrame, left=None, right=None, depth:int=0):
        """
        Abstract Decision Node Constructor
        Args:
            data (DataFrame): the node data.
            left (DecisionNode, optional): the left child node. Defaults to None.
            right (DecisionNode, optional): the right child node. Defaults to None.
            depth (int, optional): The depth of the node. Defaults to 0.
        """
        
        self.left = left
        self.right = right
        self.data = data.values
        self.data_labels = data.columns.values
        self.depth = depth
        self.featureName = ""
        self.featureIndex = 0
        self.featureVal = 0
        self.samples = len(data)
        
        # for visualization purposes
        self.id = str(DecisionNode.cnt)
        DecisionNode.cnt += 1
        
    def _bestSplit(self):
        """
        Get the best split of the decision node.
            This method is expanded later in the child classes
        """
        pass