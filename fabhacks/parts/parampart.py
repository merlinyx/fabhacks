from .part import Part
from scipy.spatial.distance import cdist

class ParamPart(Part):
    def __init__(self, part_params, scadfpath):
        super().__init__()
        self.part_params = part_params
        self.openscad_fpath = scadfpath

    def precompute_for_part(self, n=None):
        results_dict = {}
        for c in self.children:
            c.get_new_mcf()
        for i in range(len(self.children) - 1):
            childi = self.children[i]
            XA = childi.mcf_factory.generate_mcf_samples(sample_size=n)
            for j in range(i + 1, len(self.children)):
                childj = self.children[j]
                XB = childj.mcf_factory.generate_mcf_samples(sample_size=n)
                dists = cdist(XA, XB)
                results_dict[(childi.ohe.value, childj.ohe.value)] = (dists.min(), dists.max())
        self.children_pairwise_ranges = results_dict
