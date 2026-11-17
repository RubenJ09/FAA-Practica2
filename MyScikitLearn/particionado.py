from abc import ABC, abstractmethod

import numpy as np


class Particion:
    def __init__(self):
        self.indicesTrain = []
        self.indicesTest = []

class EstrategiaParticionado(ABC):
    def __init__(self):
        self.particiones = []

    @abstractmethod
    def crearParticiones(self, datos):
        pass


class ValidacionSimple(EstrategiaParticionado):
    def __init__(self, porcentajeTrain=0.8, numParticiones=1, seed=None):
        super().__init__()

        self.porcentajeTrain = porcentajeTrain
        self.numParticiones = numParticiones
        self.seed = seed

    def crearParticiones(self, datos):
        self.particiones = []
        rng = np.random.default_rng(self.seed)
        numeroDatos = len(datos)
        filasTrain = int(numeroDatos*self.porcentajeTrain)

        for _ in range(self.numParticiones):
            indices = rng.permutation(numeroDatos)
            particion = Particion()
            particion.indicesTrain = indices[:filasTrain].tolist()
            particion.indicesTest = indices[filasTrain:].tolist()
            self.particiones.append(particion)

        return self.particiones


class ValidacionCruzada(EstrategiaParticionado):
    def __init__(self, numParticiones=5, seed=None):
        super().__init__()

        self.numParticiones = numParticiones
        self.seed = seed

    def crearParticiones(self, datos):
        self.particiones = []

        rng = np.random.default_rng(self.seed)
        numeroDatos = len(datos)
        indices = rng.permutation(numeroDatos)

        folds = np.array_split(indices, self.numParticiones)

        for i in range(self.numParticiones):
            particion = Particion()
            particion.indicesTest = folds[i].tolist()
            indicesTrain = []
            for j in range(self.numParticiones):
                if j != i:
                    indicesTrain.extend(folds[j].tolist())
            particion.indicesTrain = indicesTrain
            self.particiones.append(particion)
        return self.particiones
