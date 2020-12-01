import sys
sys.path.append('../m2e2')
import torch
import json
from functools import partial
from src.dataflow.torch.Data import MultiTokenField, SparseField, EntityField
from torchtext.data import Field
from src.util.vocab import Vocab
from torchtext.vocab import Vectors
from src.dataflow.numpy.data_loader_grounding import GroundingDataset
from src.models.grounding import GroundingModel
from src.eval.Groundingtesting import GroundingTester, grounding_test
from src.engine.Groundingtraining import grounding_train
from src.engine.SRrunner import load_sr_model
from src.engine.EErunner import load_ee_model
from src.util.util_model import log

def get_embedding(device):
    
    with open("../m2e2/src/engine/out_m2e2/ee_hyps.json") as f:
        ee_hps = json.load(f)
    
    IMAGEIDField = SparseField(sequential=False, use_vocab=False, batch_first=True)
    SENTIDField = SparseField(sequential=False, use_vocab=False, batch_first=True)
    # IMAGEField = SparseField(sequential=False, use_vocab=False, batch_first=True)
    WordsField = Field(lower=True, include_lengths=True, batch_first=True)
    PosTagsField = Field(lower=True, batch_first=True)
    EntityLabelsField = MultiTokenField(lower=False, batch_first=True)
    AdjMatrixField = SparseField(sequential=False, use_vocab=False, batch_first=True)
    EntitiesField = EntityField(lower=False, batch_first=True, use_vocab=False)
    colcc = 'stanford-colcc'
    train_set = GroundingDataset(path='../m2e2/data/grounding_m2e2/grounding_train_10000.json',
                                 img_dir='../../data/voa/rawdata/img',
                                 fields={"id": ("IMAGEID", IMAGEIDField),
                                         "sentence_id": ("SENTID", SENTIDField),
                                         "words": ("WORDS", WordsField),
                                         "pos-tags": ("POSTAGS", PosTagsField),
                                         "golden-entity-mentions": ("ENTITYLABELS", EntityLabelsField),
                                         colcc: ("ADJM", AdjMatrixField),
                                         "all-entities": ("ENTITIES", EntitiesField),
                                         # "image": ("IMAGE", IMAGEField),
                                         },
                                 transform=None,
                                 amr=False,
                                 load_object=True,
                                 object_ontology_file='../m2e2/data/object/class-descriptions-boxable.csv',
                                 object_detection_pkl_file='../m2e2/data/voa/object_detect/det_results_voa_oi_1.pkl',
                                 object_detection_threshold=0.2,
                                 )


    dev_set = GroundingDataset(path='../m2e2/data/grounding_m2e2/grounding_valid_10000.json',
                                 img_dir='../m2e2/data/voa/rawdata/img',
                                 fields={"id": ("IMAGEID", IMAGEIDField),
                                         "sentence_id": ("SENTID", SENTIDField),
                                         "words": ("WORDS", WordsField),
                                         "pos-tags": ("POSTAGS", PosTagsField),
                                         "golden-entity-mentions": ("ENTITYLABELS", EntityLabelsField),
                                         colcc: ("ADJM", AdjMatrixField),
                                         "all-entities": ("ENTITIES", EntitiesField),
                                         # "image": ("IMAGE", IMAGEField),
                                         },
                                 transform=None,
                                 amr=False,
                                 load_object=True,
                                 object_ontology_file='../m2e2/data/object/class-descriptions-boxable.csv',
                                 object_detection_pkl_file='../m2e2/data/voa/object_detect/det_results_voa_oi_1.pkl',
                                 object_detection_threshold=0.2,
                                 )
    pretrained_embedding = Vectors("../m2e2/data/glove/glove.840B.300d.txt", ".", unk_init=partial(torch.nn.init.uniform_, a=-0.15, b=0.15))
    WordsField.build_vocab(train_set.WORDS, dev_set.WORDS, vectors=pretrained_embedding)

    state_dict = torch.load("../m2e2/src/engine/out_m2e2/model.pt")
    ee_model = load_ee_model(ee_hps, None, WordsField.vocab.vectors, device)
    own_state = ee_model.state_dict()
    for name, param in state_dict.items():
        model_name = name.split(".")[0]
        param_name = ".".join(name.split(".")[1:])
        if  model_name == "sr_model":
            continue
        else:
            own_state[param_name].copy_(param)
    
    return ee_model, WordsField
    
if __name__ == "__main__":
    model, WordsField = get_embedding("cpu")
    import numpy as np
    words = torch.LongTensor(np.ones((10,512))*100).to("cpu")
    length = np.array([512 for i in range(10)])
    model.get_common_feature(words, length)
    
    
    