"""A Module that implements the score update using statistical
methods.
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import List
import copy
import numpy as np

@dataclass
class StatParams:
    """Class for defining the  Statistical parameters
    Args:
        new_score_pt (np.array): new data point, a numpy M vector where M num of classes and
        filled with scores.
        curr_n (int): The current data point index (based 1)
        N (int): The number of data point to avrage if using stats like FMA. woof
        beta (float): The beta for using EMA.
        pre_score (np.array): prev data score, a numpy M vector where M num of classes and
        filled with scores.
        pre_score_list (list(np.array)): prev data score points, used only if using stats like FMA.
        class_effect (np.array): A numpy M vector where M num of classes and filled with
        contribution from each class. A vector with all ones is the diffult and all
        classes scores are added the same.
        It's usful when you have a class category of "unknown"
        so you culd give it a 0 effect on scores.
    """
    new_score_pt:np.array=0
    curr_n:int=0
    avg_n:int=10
    beta:float=0.9
    pre_score:np.array=0
    pre_score_list:List[np.ndarray]=field(default_factory=list)
    class_effect:np.array=None

    def initialize(self,class_num):
        """setting the deafult vars accurding to num of classes

        Args:
            class_num ([int]): num of classes
        """
        self.pre_score=np.zeros(class_num)
        if self.class_effect is None:
            self.class_effect=np.ones(class_num)

    def insert_score(self,score):
        """insert new score to the parameters

        Args:
            score (np.array): the new score
        """
        self.new_score_pt=score
        self.curr_n+=1

    def __copy__(self):
        """A mthod to copy a class instance

        Returns:
            StatParams: A copy of this instance.
        """
        return StatParams(
            new_score_pt=self.new_score_pt if isinstance(self.new_score_pt, int)
            else np.copy(self.new_score_pt),
            curr_n=self.curr_n,
            avg_n=self.avg_n,
            beta=self.beta,
            pre_score=self.pre_score if isinstance(self.pre_score, int)
            else np.copy(self.pre_score),
            pre_score_list=copy.deepcopy(self.pre_score_list),
            class_effect=np.copy(self.class_effect),
        )
#region Stat methods
def no_average(param:StatParams):
    """In short, this return score based on only current frame

    Args:
        param (StatParams): Statistical parameters

    Returns:
        The last score
    """
    return param.new_score_pt+param.pre_score*0

def cumulative_moving_average(param:StatParams):
    """CMA
    Average of all of the data up until the current datum point.
    CMA_{n}=(x_{n}+(n-1)*CMA_{n-1})/n
    Args:
        param (dictionary): [description]
    """
    curr_n=param.curr_n
    return (param.new_score_pt+(curr_n-1)*param.pre_score)/curr_n

def finite_moving_average(param:StatParams):
    """FMA
    Average of the last N data points up until the current datum point.
    FMA_{n}=FMA_{n-1}+(X_{n}-X_{n-N})/N

    Args:
        param (dictionary): [description]
    """
    num=param.avg_n
    pt_count=len(param.pre_score_list)
    param.pre_score_list.append(param.new_score_pt)
    if pt_count<num:
        if pt_count==0:
            return param.new_score_pt/num+param.pre_score*0
        return (param.new_score_pt)/num+param.pre_score
    last=param.pre_score_list.pop(0)
    return (param.new_score_pt-last)/num+param.pre_score

def exponential_moving_average(param:StatParams):
    """EMA
    Infinite impulse response filter that applies factor beta which decrease exponentially.
    EMA{n+1}=beta*EMA_{n}+(1-beta)*X_{n} for example beta=0.9 is about averaging on 10 samples.
    Args:
        param (dictionary): [0<=beta<1]
    """
    beta=param.beta
    return beta*param.pre_score+(1-beta)*param.new_score_pt
#endregion

class StatMethods(Enum):
    """Enum of statistical methods to use to calculate the score.

    Args:
        Enum ([method]): the statistical method to calculate the score with.
    """
    Non=no_average
    CMA=cumulative_moving_average
    FMA=finite_moving_average
    EMA=exponential_moving_average

class StatisticalCalculator:
    """A class that maintains the score using statistical analysis"""
    def __init__(
            self,
            parameters:StatParams=StatParams(),
            method:StatMethods=StatMethods.Non,
            class_num:int=2
        ):
        self.parameters = parameters
        self.method=method
        self.parameters.initialize(class_num)

    def update(self,new_score):
        """Update the score using the statistical method that was chosen

        Args:
            new_score (np.array): the last updated score from last frame

        Returns:
            [np.array]: the current statistical score
        """
        self.parameters.insert_score(new_score)
        self.parameters.pre_score=self.method(self.parameters)
        return self.parameters.pre_score

    def get_score(self):
        """return the last statistical score.

        Returns:
            [np.array]: the last statistical score
        """
        return self.parameters.pre_score

    def __copy__(self):
        """A mthod to copy a class instance

        Returns:
            StatisticalCalculator: A copy of this instance.
        """
        return StatisticalCalculator(
            parameters=self.parameters.__copy__(),
            method=self.method,
            class_num=len(self.parameters.pre_score),
        )
