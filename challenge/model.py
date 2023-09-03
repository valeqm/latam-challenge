import pandas as pd
from typing import Tuple, Union, List
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import xgboost as xgb

class DelayModel:

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        def get_period_day(date):
            date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
            morning_min = datetime.strptime("05:00", '%H:%M').time()
            morning_max = datetime.strptime("11:59", '%H:%M').time()
            afternoon_min = datetime.strptime("12:00", '%H:%M').time()
            afternoon_max = datetime.strptime("18:59", '%H:%M').time()
            evening_min = datetime.strptime("19:00", '%H:%M').time()
            evening_max = datetime.strptime("23:59", '%H:%M').time()
            night_min = datetime.strptime("00:00", '%H:%M').time()
            night_max = datetime.strptime("4:59", '%H:%M').time()

            if (date_time > morning_min and date_time < morning_max):
                return 'mañana'
            elif (date_time > afternoon_min and date_time < afternoon_max):
                return 'tarde'
            elif (
                (date_time > evening_min and date_time < evening_max) or
                (date_time > night_min and date_time < night_max)
            ):
                return 'noche'

        def is_high_season(fecha):
            fecha_año = int(fecha.split('-')[0])
            fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
            range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year=fecha_año)
            range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year=fecha_año)
            range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year=fecha_año)
            range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year=fecha_año)
            range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year=fecha_año)
            range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year=fecha_año)
            range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year=fecha_año)
            range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year=fecha_año)

            if ((fecha >= range1_min and fecha <= range1_max) or
                (fecha >= range2_min and fecha <= range2_max) or
                (fecha >= range3_min and fecha <= range3_max) or
                (fecha >= range4_min and fecha <= range4_max)):
                return 1
            else:
                return 0

        def get_min_diff(data):
            fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
            fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
            min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
            return min_diff

        # Aplicar las funciones de preprocesamiento a los datos
        data['period_day'] = data['Fecha-I'].apply(get_period_day)
        data['high_season'] = data['Fecha-I'].apply(is_high_season)
        data['min_diff'] = data.apply(get_min_diff, axis=1)

        threshold_in_minutes = 15
        data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)

        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix='MES')],
            axis=1
        )
        target = data['delay']

        if target_column is not None:
            return features, target
        else:
            return features
            

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        # Entrenar el modelo aquí
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)
        self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01)
        self._model.fit(x_train, y_train)
        
        return

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        # Realizar predicciones aquí
        xgboost_y_preds = self._model.predict(features)
        xgboost_y_preds = [1 if y_pred > 0.5 else 0 for y_pred in xgboost_y_preds]
        return xgboost_y_preds
