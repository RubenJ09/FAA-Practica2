import pandas as pd
from pandas.api.types import is_numeric_dtype


class Datos:
    def __init__(self, fichero):
        self.datos = pd.read_csv(fichero)

        self.diccionarios = {}

        self.datos_numericos = self.datos.copy()

        self.crear_diccionarios()
        self.convertir_a_numerico()

    def crear_diccionarios(self):
        for columna in self.datos.columns:
            if not is_numeric_dtype(self.datos[columna]):
                valores = sorted(self.datos[columna].unique())
                diccionario = {valor: i for i, valor in enumerate(valores)}
                self.diccionarios[columna] = diccionario
            else:
                self.diccionarios[columna] = {}

    def convertir_a_numerico(self):
        for columna in self.datos.columns:
            diccionario = self.diccionarios[columna]

            if diccionario:
                self.datos_numericos[columna] = self.datos[columna].map(diccionario)

    def extrae_datos(self, indices=None):
        if indices is None:
            return self.datos_numericos

        return self.datos_numericos.iloc[indices]

    def estandarizarDatos(self, media=True, std=True, indices=None):
        datos_referencia = self.datos_numericos
        if indices is not None:
            datos_referencia = self.datos_numericos.iloc[indices]

        for columna in self.datos_numericos.columns:
            if columna == "Class":
                continue

            if is_numeric_dtype(self.datos_numericos[columna]):
                m = datos_referencia[columna].mean() if media else 0.0
                s = datos_referencia[columna].std(ddof=0) if std else 1.0
                if s == 0:
                    s = 1.0

                self.datos_numericos[columna] = (self.datos_numericos[columna] - m) / s
