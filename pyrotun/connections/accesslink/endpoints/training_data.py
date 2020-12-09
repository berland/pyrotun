#!/usr/bin/env python

from .resource import Resource
from .training_data_transaction import TrainingDataTransaction


class TrainingData(Resource):
    """This resource allows partners to access their users' training data.

    https://www.polar.com/accesslink-api/?http#training-data
    """

    def get_exercise(self, user_id, access_token, transaction_id, exercise_id):
        response = self._post(
            endpoint=f"/users/{user_id}/exercise-transactions/{transaction_id}/exercises/{exercise_id}",
            access_token=access_token,
        )
        print(response)

    def create_transaction(self, user_id, access_token):
        """Initiate exercise transaction

        Check for new training data and create a new transaction if data is available.

        :param user_id: id of the user
        :param access_token: access token of the user
        """
        response = self._post(
                endpoint="/users/{}/exercise-transactions".format(user_id),
                access_token=access_token,
            )

        if not response:
            return None

        return TrainingDataTransaction(
            oauth=self.oauth,
            transaction_url=response["resource-uri"],
            user_id=user_id,
            access_token=access_token,
        )
