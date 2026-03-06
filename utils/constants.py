DEFAULT_RELATION_WEIGHTS = {
    "IsA": 0.5,
    "AtLocation": 0.5,
    "PartOf": 1.0,
    "Antonym": 0.75,
    "UsedFor": 1.0,
    "DistinctFrom": 1.0,
    "HasProperty": 0.75,
    "SimilarTo": 1.0,
    "CapableOf": 1.0,
    "Causes": 1.0,
    "MadeOf": 1.0,
    "ReceivesAction": 0.75,
    "HasPrerequisite": 0.75,
    "HasSubevent": 1.0,    
    "CreatedBy": 1.0,
    "LocatedNear": 1.0,
    "HasA":             1.0,  
}

ASSASSIN_PENALTY_WEIGHT   = 10.0

OPPONENT_PENALTY_WEIGHT   = 4.0

NEUTRAL_PENALTY_WEIGHT    = 0.5
NEUTRAL_PENALTY_THRESHOLD = 0.6   

COVERAGE_BONUS_PER_EXTRA_TARGET = 2.0