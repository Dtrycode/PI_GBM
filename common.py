# the variable shared among files
models = {} # store all models for the ease of use
parameters = {} # store all hyperparameters
predictions = {} # store predictions of privileged model

# some settings
all_feature_types = {
        'adult': {
            'pf': ['q']+['c']*2,
            'nf': ['q']*4+['c']*6
            },
        'diabetes_gender': {
            'pf': ['c'],
            'nf': ['q']*7+['c']*9
            },
        'dutch': {
            'pf': ['c'],
            'nf': ['c']*10
            },
        'bank': {
            'pf': ['c']*2,
            'nf': ['q']*6+['c']*8
            },
        'credit': {
            'pf': ['c']*3,
            'nf': ['q']*14+['c']*6
            },
        'compas': {
            'pf': ['c']*2,
            'nf': ['q']*2+['c']*4
            },
        'compas_viol': {
            'pf': ['c']*2,
            'nf': ['q']*2+['c']*4
            },
        'communities': {
            'pf': ['c'],
            'nf': ['q']*20
            },
        'student_mat': {
            'pf': ['c']*2,
            'nf': ['q']*14+['c']*16
            },
        'student_por': {
            'pf': ['c']*2,
            'nf': ['q']*14+['c']*16
            },
        'oulad': {
            'pf': ['c'],
            'nf': ['q']*2+['c']*7
            },
        'kdd': {
            'pf': ['c']*2,
            'nf': ['q']*7+['c']*14
            },
        'ns': {
            'pf': ['c']*2,
            'nf': ['q']*17
            },
        'rare': {
            'pf': ['c']*3,
            'nf': ['q']*10+['c']*56
            },
        'numom2b_a': {'pf': None, 'nf': None},
        'numom2b_b': {'pf': None, 'nf': None},
        'heartdisease': {'pf': None, 'nf': None},
        'carevaluation': {'pf': None, 'nf': None},
        'spambase': {'pf': None, 'nf': None}
        }

all_feature_types_all = {
        'adult': {
            'all': ['q']*5+['c']*8
            },
        'diabetes_gender': {
            'all': ['q']*7+['c']*10
            },
        'dutch': {
            'all': ['c']*11
            },
        'bank': {
            'all': ['q']*6+['c']*10
            },
        'credit': {
            'all': ['q']*14+['c']*9
            },
        'compas': {
            'all': ['q']*3+['c']*6
            },
        'compas_viol': {
            'all': ['q']*3+['c']*6
            },
        'communities': {
            'all': ['q']*20+['c']
            },
        'student_mat': {
            'all': ['q']*14+['c']*18
            },
        'student_por': {
            'all': ['q']*14+['c']*18
            },
        'oulad': {
            'all': ['q']*2+['c']*8
            },
        'kdd': {
            'all': ['q']*7+['c']*16
            },
        'numom2b_b': {
            'all': None
            },
        'ns': {
            'all': ['q']*17+['c']*2
            },
        'rare': {
            'all': ['q']*10+['c']*59
            }
        }

data_thred = {
        'adult': {
            'normal': 0.001,
            'pi': 0.003,
            'pi*': 0.001,
            'all': 0.318
            },
        'diabetes_gender': {
            'normal': 0.212,
            'pi': 0.268,
            'pi*': 0.236,
            'all': 0.227
            },
        'dutch': {
            'normal': 0.619,
            'pi': 0.537,
            'pi*': 0.591,
            'all': 0.507
            },
        'bank': {
            'normal': 0.364,
            'pi': 0.415,
            'pi*': 0.389,
            'all': 0.443
            },
        'credit': {
            'normal': 0.482,
            'pi': 0.483,
            'pi*': 0.510,
            'all': 0.482
            },
        'compas': {
            'normal': 0.019,
            'pi': 0.001,
            'pi*': 0.001,
            'all': 0.001
            },
        'compas_viol': {
            'normal': 0.493,
            'pi': 0.511,
            'pi*': 0.172,
            'all': 0.206
            },
        'communities': {
            'normal': 0.455,
            'pi': 0.397,
            'pi*': 0.345,
            'all': 0.349
            },
        'student_mat': {
            'normal': 0.530,
            'pi': 0.537,
            'pi*': 0.537,
            'all': 0.53
            },
        'student_por': {
            'normal': 0.659,
            'pi': 0.535,
            'pi*': 0.494,
            'all': 0.558
            },
        'oulad': {
            'normal': 0.001,
            'pi': 0.001,
            'pi*': 0.001,
            'all': 0.001
            },
        'kdd': {
            'normal': 0.5,
            'pi': 0.504,
            'pi*': 0.505,
            'all': 0.503
            },
        'numom2b_a': {
            'normal': 0.119,
            'pi': 0.100,
            'pi*': 0.147
            },
        'numom2b_b': {
            'normal': 0.122,
            'pi': 0.195,
            'pi*': 0.148,
            'all': 0.203
            },
        'ns': {
            'normal': 0.552,
            'pi': 0.553,
            'pi*': 0.553,
            'all': 0.552
            },
        'rare': {
            'normal': 0.162,
            'pi': 0.403,
            'pi*': 0.336,
            'all': 0.410
            },
        'heartdisease': {
            'normal': 0.481,
            'pi': 0.456,
            'pi*': 0.615
            },
        'carevaluation': {
            'normal': 0.516,
            'pi': 0.521,
            'pi*': 0.55
            },
        'spambase': {
            'normal': 0.438,
            'pi': 0.465,
            'pi*': 0.437
            }
        }

data_protect_group = {
        'adult': [1, 0, 0],
        'diabetes_gender': [0],
        'dutch': [1],
        'bank': [1, 1],
        'credit': [1, 0, 0],
        'compas': [0, 0],
        'compas_viol': [0, 0],
        'communities': [0],
        'student_mat': [0, 0],
        'student_por': [0, 0],
        'oulad': [0],
        'kdd': [0, 0],
        'numom2b_b': [0], # (1, 0, 0, 0, 0, 0, 0, 0) is the majority
        'ns': [1, 1],
        'rare': [1, 0, 1]
        }
