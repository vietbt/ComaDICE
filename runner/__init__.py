from runner.bc.continuous import Trainer as BC_Continuous_Trainer
from runner.bcq.continuous import Trainer as BCQ_Continuous_Trainer
from runner.cql.continuous import Trainer as CQL_Continuous_Trainer
from runner.icq.continuous import Trainer as ICQ_Continuous_Trainer
from runner.madice.continuous import Trainer as MADice_Continuous_Trainer
from runner.omar.continuous import Trainer as OMAR_Continuous_Trainer
from runner.omiga.continuous import Trainer as OMIGA_Continuous_Trainer

from runner.bc.discrete import Trainer as BC_Discrete_Trainer
from runner.madice.discrete import Trainer as MADice_Discrete_Trainer
from runner.omiga.discrete import Trainer as OMIGA_Discrete_Trainer


TRAINER_CONTINUOUS = {
    "bc": BC_Continuous_Trainer,
    "bcq": BCQ_Continuous_Trainer,
    "cql": CQL_Continuous_Trainer,
    "icq": ICQ_Continuous_Trainer,
    "madice": MADice_Continuous_Trainer,
    "omar": OMAR_Continuous_Trainer,
    "omiga": OMIGA_Continuous_Trainer,
}

TRAINER_DISCRETE = {
    "bc": BC_Discrete_Trainer,
    "madice": MADice_Discrete_Trainer,
    "omiga": OMIGA_Discrete_Trainer,
}