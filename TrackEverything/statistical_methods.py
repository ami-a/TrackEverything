"""Temporal smoothing of classification scores.

Detections are noisy frame to frame. These helpers keep a running statistic per
tracked object so that a momentary bad classification does not flip an object
that has been classified consistently for the last hundred frames.
"""
import copy
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Callable, Optional, Union

import numpy as np


@dataclass
class StatParams:
    """Class for defining the statistical parameters.

    Args:
        new_score_pt (np.ndarray): new data point, a numpy M vector where M num of classes
            and filled with scores.
        curr_n (int): The current data point index (based 1).
        avg_n (int): The number of data points to average if using stats like FMA.
        beta (float): The beta for using EMA.
        pre_score (np.ndarray): prev data score, a numpy M vector where M num of classes
            and filled with scores.
        pre_score_list (list(np.ndarray)): prev data score points, used only if using
            stats like FMA.
        class_effect (np.ndarray): A numpy M vector where M num of classes, filled with the
            contribution of each class. A vector of all ones is the default, and all class
            scores contribute equally. It is useful when you have an "unknown" class
            category, which you can give a 0 effect on scores so that a frame in which
            the object was unrecognisable does not overwrite what you already know.
    """
    new_score_pt:Union[int, np.ndarray]=0
    curr_n:int=0
    avg_n:int=10
    beta:float=0.9
    pre_score:Union[int, np.ndarray]=0
    pre_score_list:list[np.ndarray]=field(default_factory=list)
    class_effect:Optional[np.ndarray]=None

    def __post_init__(self) -> None:
        #remember whether the caller chose these weights, so that an auto-generated
        #vector can be resized silently while a deliberate one is defended
        self._explicit_class_effect = self.class_effect is not None

    def initialize(self, class_num: int) -> None:
        """Set the default vars according to the number of classes.

        Args:
            class_num (int): num of classes.

        Raises:
            ValueError: an explicit ``class_effect`` was supplied whose length does not
                match ``class_num``.
        """
        self.pre_score=np.zeros(class_num)
        if not getattr(self, "_explicit_class_effect", self.class_effect is not None):
            #auto-generated weights: resize to whatever the real class count turns out to be
            self.class_effect=np.ones(class_num)
        elif len(self.class_effect)!=class_num:
            raise ValueError(
                f"class_effect has {len(self.class_effect)} entries but the scores have "
                f"{class_num} classes. Supply one weight per class, or leave it as None."
            )

    def insert_score(self, score: np.ndarray) -> None:
        """Insert a new score into the parameters.

        Args:
            score (np.ndarray): the new score.
        """
        self.new_score_pt=score
        self.curr_n+=1

    def __copy__(self) -> "StatParams":
        """Copy this class instance.

        Returns:
            StatParams: A copy of this instance.
        """
        new_params=StatParams(
            new_score_pt=self.new_score_pt if isinstance(self.new_score_pt, int)
            else np.copy(self.new_score_pt),
            curr_n=self.curr_n,
            avg_n=self.avg_n,
            beta=self.beta,
            pre_score=self.pre_score if isinstance(self.pre_score, int)
            else np.copy(self.pre_score),
            pre_score_list=copy.deepcopy(self.pre_score_list),
            class_effect=None if self.class_effect is None else np.copy(self.class_effect),
        )
        #carry the provenance over, otherwise an auto-generated vector would look
        #deliberate to the copy and refuse to resize
        new_params._explicit_class_effect=self._explicit_class_effect
        return new_params
#region Stat methods
def no_average(param: StatParams) -> np.ndarray:
    """Return a score based on only the current frame.

    Args:
        param (StatParams): Statistical parameters.

    Returns:
        np.ndarray: The last score.
    """
    return param.new_score_pt+param.pre_score*0

def cumulative_moving_average(param: StatParams) -> np.ndarray:
    """CMA - the average of all the data up until the current datum point.

    ``CMA_{n}=(x_{n}+(n-1)*CMA_{n-1})/n``

    Args:
        param (StatParams): Statistical parameters; uses ``curr_n``, ``new_score_pt``
            and ``pre_score``.

    Returns:
        np.ndarray: The cumulative moving average score.
    """
    curr_n=param.curr_n
    return (param.new_score_pt+(curr_n-1)*param.pre_score)/curr_n

def finite_moving_average(param: StatParams) -> np.ndarray:
    """FMA - the average of the last N data points up until the current datum point.

    ``FMA_{n}=FMA_{n-1}+(X_{n}-X_{n-N})/N``

    Args:
        param (StatParams): Statistical parameters; uses ``avg_n``, ``new_score_pt``,
            ``pre_score`` and ``pre_score_list``.

    Returns:
        np.ndarray: The finite moving average score.
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

def exponential_moving_average(param: StatParams) -> np.ndarray:
    """EMA - an infinite impulse response filter weighted by a beta that decreases
    exponentially.

    ``EMA{n+1}=beta*EMA_{n}+(1-beta)*X_{n}``; for example beta=0.9 is about
    averaging over 10 samples.

    Args:
        param (StatParams): Statistical parameters; uses ``beta``, ``pre_score``
            and ``new_score_pt``. Requires ``0<=beta<1``.

    Returns:
        np.ndarray: The exponential moving average score.
    """
    beta=param.beta
    return beta*param.pre_score+(1-beta)*param.new_score_pt
#endregion

class StatMethods(Enum):
    """Enum of statistical methods used to calculate the score.

    Members are callable, so ``StatMethods.EMA(params)`` and passing the member
    itself as ``StatisticalCalculator(method=...)`` both work. The values are
    wrapped in :func:`functools.partial` because a bare function in an ``Enum``
    body is treated as a method rather than a member, which would leave this
    enum with no members at all.
    """
    Non=partial(no_average)
    CMA=partial(cumulative_moving_average)
    FMA=partial(finite_moving_average)
    EMA=partial(exponential_moving_average)

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.value(*args, **kwargs)

class StatisticalCalculator:
    """A class that maintains the score using statistical analysis."""
    def __init__(
            self,
            parameters: Optional[StatParams] = None,
            method: Union[StatMethods, Callable[[StatParams], np.ndarray]] = StatMethods.Non,
            class_num: int = 2,
        ):
        """Build a calculator.

        Args:
            parameters (Optional[StatParams]): statistical parameters. A fresh
                :class:`StatParams` is created when omitted, so calculators never
                share state.
            method (Union[StatMethods, Callable]): the statistical method to apply.
            class_num (int): initial number of classes. Adjusted automatically on the
                first real score, so an incorrect value here is not fatal.
        """
        self.parameters = StatParams() if parameters is None else parameters
        self.method=method
        self.parameters.initialize(class_num)

    def _ensure_shape(self, new_score: np.ndarray) -> None:
        """Re-initialise if the real class count differs from the assumed one.

        ``class_num`` is only a guess until the first score arrives; a detector with
        no classification model produces 1 class and a 3-class model produces 3,
        neither of which matches the default of 2.

        Args:
            new_score (np.ndarray): the incoming score vector.
        """
        class_num=1 if np.ndim(new_score)==0 else len(new_score)
        if np.ndim(self.parameters.pre_score)==0 or \
                len(self.parameters.pre_score)!=class_num:
            self.parameters.initialize(class_num)

    def update(self, new_score: np.ndarray) -> np.ndarray:
        """Update the score using the chosen statistical method.

        The incoming score is weighted by ``class_effect`` before it enters the
        statistic, so a class configured with a low effect barely disturbs what has
        already been learned about the object.

        Args:
            new_score (np.ndarray): the last updated score from the last frame.

        Returns:
            np.ndarray: the current statistical score.
        """
        self._ensure_shape(new_score)
        self.parameters.insert_score(new_score*self.parameters.class_effect)
        self.parameters.pre_score=self.method(self.parameters)
        return self.parameters.pre_score

    def get_score(self) -> np.ndarray:
        """Return the last statistical score.

        Returns:
            np.ndarray: the last statistical score.
        """
        return self.parameters.pre_score

    def __copy__(self) -> "StatisticalCalculator":
        """Copy this class instance.

        Returns:
            StatisticalCalculator: A copy of this instance.
        """
        new_calc=StatisticalCalculator.__new__(StatisticalCalculator)
        new_calc.parameters=self.parameters.__copy__()
        new_calc.method=self.method
        return new_calc
