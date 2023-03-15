from pandas import DataFrame
from numpy import unique
from IPython.display import display
from IPython import get_ipython
from graphviz import Digraph

from ScratchML._model_types import PredictionModel

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

class DecisionTree(PredictionModel):
    """
    Decision tree abstract definition
    """
    def __init__(self, maxDepth: int, minSample: int):
        """
        Decision Tree constructor

        Args:
            maxDepth (int): the max depth of the decision tree
            minSample (int): the least samples in a node allowed to be split
        """
        
        self.root : DecisionNode = None
        self.maxDepth = maxDepth
        self.minSample = minSample
        
        # for visualization purposes
        self.graph = Digraph(format="png", node_attr={"shape":"rectangle", "fontsize":"8"})

   
    def _can_split(self, node: DecisionNode):
        """
        Determine if the given node can be split

        Args:
            node (DecisionNode): the node to split

        Returns:
            bool: can we split the node?
        """
        
        if node is None:
            return False
        if (self.maxDepth > node.depth) and (self.minSample <= len(node.data)):
            return True
        return False
             
    
    def _split(self, node: DecisionNode):
        """
        Split the node on the most fitting axis
        
        Args:
            node (DecisionNode): the node to split
        """

        # perform node calibration
        node._bestSplit()

        # If the feature is categorial
        if len(unique(node.data[:,node.featureIndex]))<=2:
            dfB = node.data[node.data[:,node.featureIndex] == node.featureVal]
            dfA = node.data[node.data[:,node.featureIndex] != node.featureVal]
        
        # Else, if the feature to split of this node is numeric
        else:
            dfA = node.data[node.data[:,node.featureIndex] >= node.featureVal]
            dfB = node.data[node.data[:,node.featureIndex] < node.featureVal]
        
        dfA, dfB = DataFrame(dfA, columns=node.data_labels), DataFrame(dfB,columns=node.data_labels)
        
        return dfA, dfB
    
    
    def fit(self):
        """
        Fit the model to make a Decision Maker
        """
        
        self._fit(self.root)
        self.__connect_graph(self.root)
    
    
    def _fit(self, node: DecisionNode):
        """
        Recursive fitting function for Decision Maker.
            The expanded functionality is to be found in the children classes.
        Args:
            node (DecisionNode): the root of the decition maker
        """
        
        self._split(node)
        self._fit(node.left)
        self._fit(node.right)
    
    
    def __connect_graph(self,node: DecisionNode):
        """Recursively connect the graph of the Decision Maker
        
        Args:
            node (DecisionNode): The root of the decision maker
        """
        
        if node is None:
            return
        
        if node.left is not None:
            self.graph.edge(node.id, node.left.id)
            self.__connect_graph(node.left)
        
        if node.right is not None:
            self.graph.edge(node.id, node.right.id)
            self.__connect_graph(node.right)
        
    
    def predict(self,samp):
        """
        Predict the samp value based on he decision tree

        Args:
            samp (list): list of parameters

        Returns:
            any: the prediction
        """
        
        return self._predict(self.root,samp)
    
    
    def _predict(self, node: DecisionNode, samp):
        """
        Recursive prediction method
            Is expanded in the children classes

        Args:
            node (DecisionNode): the root
            samp (list): list of parameters

        Returns:
            any: the prediction
        """
        
        # If the feature is categorial
        if len(unique(node.data[:,node.featureIndex]))==2:
            if samp[node.featureIndex] == node.featureVal:
                return self._predict(node.left, samp)
            else:
                return self._predict(node.right, samp)
        # Else, if the feature is numeric
        else:
            if samp[node.featureName] >= node.featureVal:
                return self._predict(node.right, samp)
            else:
                return self._predict(node.left, samp)
    
    
    def display(self,in_console=True):
        """
        Display the classifier

        Args:
            in_console (bool, optional): Display right in the console. Defaults to True.
        """
        
        if in_console:
            display(self.graph)
        else:
            self.graph.view("decisionTree.dot")
            
            
    def is_notebook() -> bool:
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True   # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return False  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return False      # Probably standard Python interpreter
