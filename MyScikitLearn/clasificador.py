from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.stats import norm

class Clasificador(ABC):
    @abstractmethod
    def entrenamiento(self, datosTrain, clasesTrain):
        pass

    @abstractmethod
    def clasifica(self, datosTest):
        pass

    def error(self, clasesReales, clasesPredichas):
        numErrores = 0

        for real, predicha in zip(clasesReales, clasesPredichas):
            if real != predicha:
                numErrores += 1

        return numErrores / len(clasesReales)

    def matrizConfusion(self, clasesReales, clasesPredichas):
        matriz = np.zeros((2, 2), dtype=int)

        for real, predicha in zip(clasesReales, clasesPredichas):
            matriz[int(real), int(predicha)] += 1

        return matriz

    def validacionErrores(self, estrategia, datos):
        particiones = estrategia.crearParticiones(datos)
        errores = []

        for particion in particiones:
            datosTrain = datos.iloc[particion.indicesTrain]
            datosTest = datos.iloc[particion.indicesTest]

            x_train = datosTrain.drop(columns=["Class"])
            y_train = datosTrain["Class"]

            x_test = datosTest.drop(columns=["Class"])
            y_test = datosTest["Class"]

            self.entrenamiento(x_train, y_train)

            predicciones = self.clasifica(x_test)

            errorParticion = self.error(y_test, predicciones)
            errores.append(errorParticion)

        return errores

    def validacion(self, estrategia, datos):
        errores = self.validacionErrores(estrategia, datos)
        return np.mean(errores)


class ClasificadorNaiveBayes(Clasificador):
    def __init__(self, laplace=False):
        self.laplace = laplace
        self.clases = None
        self.priors = {}
        self.stats_cat = {}
        self.stats_cont = {}
        self.columnas = None

    def entrenamiento(self, datosTrain, clasesTrain):
        self.columnas = datosTrain.columns
        self.clases = np.unique(clasesTrain)
        n_total = len(clasesTrain)

        self.priors = {}
        self.stats_cat = {}
        self.stats_cont = {}

        for c in self.clases:
            mask = (clasesTrain == c)
            n_c = mask.sum()
            self.priors[c] = n_c / n_total

            sub_datos = datosTrain[mask]
            self.stats_cat[c] = {}
            self.stats_cont[c] = {}

            for col in self.columnas:
                valores_col = datosTrain[col]
                if valores_col.dtype.kind == 'f' or valores_col.nunique() > 10:
                    media = sub_datos[col].mean()
                    var = sub_datos[col].var(ddof=0)
                    if var == 0 or pd.isna(var):
                        var = 1e-6
                    self.stats_cont[c][col] = (media, var)
                else:
                    conteo = sub_datos[col].value_counts().to_dict()
                    self.stats_cat[c][col] = conteo

    def clasifica(self, datosTest):
        predicciones = []
        for _, fila in datosTest.iterrows():
            prob_clases = {}
            for c in self.clases:
                log_prob = np.log(self.priors[c])
                for col in self.columnas:
                    val = fila[col]
                    if col in self.stats_cont[c]:
                        media, var = self.stats_cont[c][col]
                        std = np.sqrt(var)
                        p = norm.pdf(val, loc=media, scale=std)
                        if p == 0:
                            p = 1e-9
                        log_prob += np.log(p)
                    else:
                        conteo = self.stats_cat[c][col]
                        total_c = sum(conteo.values())
                        n_val = conteo.get(val, 0)
                        n_unicos = len(set(v for dic in self.stats_cat.values() for v in dic[col].keys()))
                        if self.laplace:
                            prob_val = (n_val + 1) / (total_c + n_unicos)
                        else:
                            prob_val = n_val / total_c if total_c > 0 else 0
                            if prob_val == 0:
                                prob_val = 1e-9
                        log_prob += np.log(prob_val)
                prob_clases[c] = log_prob
            clase_predicha = max(prob_clases, key=prob_clases.get)
            predicciones.append(clase_predicha)

        return predicciones

    def probabilidades(self, datosTest):
        probabilidades = []

        for _, fila in datosTest.iterrows():
            log_probs = []

            for c in self.clases:
                log_prob = np.log(self.priors[c])
                for col in self.columnas:
                    val = fila[col]
                    if col in self.stats_cont[c]:
                        media, var = self.stats_cont[c][col]
                        std = np.sqrt(var)
                        p = norm.pdf(val, loc=media, scale=std)
                        if p == 0:
                            p = 1e-9
                        log_prob += np.log(p)
                    else:
                        conteo = self.stats_cat[c][col]
                        total_c = sum(conteo.values())
                        n_val = conteo.get(val, 0)
                        n_unicos = len(set(v for dic in self.stats_cat.values() for v in dic[col].keys()))
                        if self.laplace:
                            prob_val = (n_val + 1) / (total_c + n_unicos)
                        else:
                            prob_val = n_val / total_c if total_c > 0 else 0
                            if prob_val == 0:
                                prob_val = 1e-9
                        log_prob += np.log(prob_val)
                log_probs.append(log_prob)
            log_probs = np.array(log_probs)

            probs = np.exp(log_probs - np.max(log_probs))
            probs = probs / np.sum(probs)

            indice_clase_1 = np.where(self.clases == 1)[0][0]
            probabilidades.append(probs[indice_clase_1])

        return np.array(probabilidades)


class ClasificadorKNN(Clasificador):
    def __init__(self, k=5):
        self.k = k
        self.datosTrain = None
        self.clasesTrain = None

    def entrenamiento(self, datosTrain, clasesTrain):
        self.datosTrain = datosTrain.values
        self.clasesTrain = np.array(clasesTrain)

    def clasifica(self, datosTest):
        predicciones = []
        test_vals = datosTest.values
        for fila in test_vals:
            distancias = np.sqrt(np.sum((self.datosTrain - fila) ** 2, axis=1))
            indices_vecinos = np.argsort(distancias)[:self.k]
            clases_vecinos = self.clasesTrain[indices_vecinos]
            valores, conteos = np.unique(clases_vecinos, return_counts=True)
            clase_mayoritaria = valores[np.argmax(conteos)]
            predicciones.append(clase_mayoritaria)

        return predicciones

    def probabilidades(self, datosTest):
        probabilidades = []
        test_vals = datosTest.values

        for fila in test_vals:
            distancias = np.sqrt(
                np.sum((self.datosTrain - fila) ** 2, axis=1)
            )

            indices_vecinos = np.argsort(distancias)[:self.k]
            clases_vecinos = self.clasesTrain[indices_vecinos]

            probabilidad = np.sum(clases_vecinos == 1) / len(clases_vecinos)
            probabilidades.append(probabilidad)

        return np.array(probabilidades)



class ClasificadorRegresionLogistica(Clasificador):
    def __init__(self, tasaAprendizaje=0.01, pasos=1000):
        self.tasaAprendizaje = tasaAprendizaje
        self.pasos = pasos
        self.pesos = None

    def sigmoide(self, valores):
        valores = np.clip(valores, -500, 500)
        return 1 / (1 + np.exp(-valores))

    def entrenamiento(self, datosTrain, clasesTrain):
        datos = datosTrain.to_numpy()
        datos = np.column_stack((np.ones(len(datos)), datos))
        clases = np.asarray(clasesTrain)

        self.pesos = np.zeros(datos.shape[1])

        for _ in range(self.pasos):
            probabilidades = self.sigmoide(datos @ self.pesos)
            self.pesos += self.tasaAprendizaje * (datos.T @ (clases - probabilidades)) / len(datos)

    def probabilidades(self, datosTest):
        datos = datosTest.to_numpy()
        datos = np.column_stack((np.ones(len(datos)), datos))
        return self.sigmoide(datos @ self.pesos)

    def clasifica(self, datosTest):
        probabilidades = self.probabilidades(datosTest)
        return (probabilidades >= 0.5).astype(int)
