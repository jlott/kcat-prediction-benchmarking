


from kcatbench.model_wrapper.base import BaseModel


class CataProWrapper(BaseModel):
    name = "CataPro"

    def __init__(self):
        super().__init__()

    def _prepare_resources(self):
        