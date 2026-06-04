"""Combined predictor for the Garmin companion deployment (DESIGN_DECISIONS A8).

One KServe deployment serves all three models. The panel chose this over three
separate deployments + a decision API because the models share the same user context
(one online feature-store round-trip), all three scores are needed before any
recommendation, and a single pod means one cold-start surface and local partial-failure
handling.

Packaging: the three pickles are saved into ONE model directory in notebook 2, so they
all land under ``MODEL_FILES_PATH`` here. The readiness feature view is resolved via
model provenance (``model.get_feature_view()``); the stress and recovery feature views
are fetched by name. Predictions + features are logged via ``fv.log`` for monitoring.

Wire-up (notebook 3)::

    deployment = combined_model.deploy(
        name="garminrecommendation",
        script_file="deployments/predictor.py",
        resources={"num_instances": 1},                  # avoid scale-to-zero cold start
        inference_logger=InferenceLogger(mode="ALL"),
    )
"""

import os

import joblib
import numpy as np

import hopsworks

# Status thresholds for the §11 decision layer.
_RECOVERY_YELLOW_H = 8.0
_RECOVERY_RED_H = 24.0


class Predict(object):
    def __init__(self, async_logger=None, model=None):
        project = hopsworks.login()
        self.mr = project.get_model_registry()
        self.fs = project.get_feature_store()

        # Readiness FV via provenance of the registered combined model.
        self.hopsworks_model = model or self.mr.get_model("garmin_combined", version=1)
        self.readiness_fv = self.hopsworks_model.get_feature_view()
        # Best-effort feature logging. Prediction logging is disabled on this cluster
        # (the Metastore cannot create logging feature groups), so guard init_serving
        # so a missing logging setup can never block pod startup.
        if async_logger is not None:
            try:
                self.readiness_fv.init_serving(feature_logger=async_logger)
            except Exception as exc:  # pragma: no cover - logging must never break serving
                print(f"feature-logger init skipped: {exc}")

        # Stress + recovery feature views by name (different entities/keys).
        self.stress_fv = self.fs.get_feature_view("fv_stress_anomaly_realtime", 1)
        self.recovery_fv = self.fs.get_feature_view("fv_recovery_time_after_workout", 1)

        mfp = os.environ["MODEL_FILES_PATH"]
        self.readiness_model = joblib.load(mfp + "/readiness_model.pkl")
        self.stress_model = joblib.load(mfp + "/stress_model.pkl")
        self.recovery_model = joblib.load(mfp + "/recovery_model.pkl")
        print("Garmin combined predictor initialised")

    # -- helpers -------------------------------------------------------------------

    @staticmethod
    def _parse(inputs):
        """Accept [[user_id]] or [{'user_id':..., 'activity_id':..., ...}]."""
        first = inputs[0]
        if isinstance(first, dict):
            return first
        return {"user_id": first[0] if isinstance(first, (list, tuple)) else first}

    def _readiness(self, user_id, date):
        entry = {"user_id": user_id, "date": date}
        untransformed = self.readiness_fv.get_feature_vector(entry, transform=False)
        transformed = self.readiness_fv.transform(untransformed)
        proba = float(self.readiness_model.predict_proba(np.asarray(transformed).reshape(1, -1))[0].max())
        cls = str(self.readiness_model.predict(np.asarray(transformed).reshape(1, -1))[0])
        # Best-effort prediction logging (won't fail the request).
        try:
            self.readiness_fv.log(
                untransformed_features=[untransformed],
                transformed_features=[transformed],
                predictions=[[cls]],
                model=self.hopsworks_model,
            )
        except Exception as exc:  # pragma: no cover - logging must never break serving
            print(f"readiness logging skipped: {exc}")
        return cls, proba

    def _stress(self, user_id, request_parameters):
        fv_in = self.stress_fv.get_feature_vector(
            {"user_id": user_id}, request_parameters=request_parameters or None
        )
        score = float(self.stress_model.decision_function(np.asarray(fv_in).reshape(1, -1))[0])
        # IsolationForest: lower score = more anomalous. Map to [0,1] anomaly score.
        anomaly = float(1.0 / (1.0 + np.exp(score)))
        level = "high" if anomaly > 0.75 else "moderate" if anomaly > 0.5 else "normal"
        return level, anomaly

    def _recovery(self, user_id, activity_id):
        if activity_id in (None, "latest"):
            return None  # no active episode
        fv_in = self.recovery_fv.get_feature_vector({"user_id": user_id, "activity_id": activity_id})
        hours = float(self.recovery_model.predict(np.asarray(fv_in).reshape(1, -1))[0])
        return max(0.0, hours)

    @staticmethod
    def _combine(readiness_cls, stress_level, recovery_hours_remaining, pain_flag=False):
        """§11 max-risk decision layer."""
        red = (
            readiness_cls == "red"
            or stress_level == "high"
            or (recovery_hours_remaining is not None and recovery_hours_remaining > _RECOVERY_RED_H)
            or pain_flag
        )
        if red:
            return "red"
        yellow = (
            readiness_cls == "yellow"
            or stress_level == "moderate"
            or (recovery_hours_remaining is not None and recovery_hours_remaining > _RECOVERY_YELLOW_H)
        )
        return "yellow" if yellow else "green"

    # -- entrypoint ----------------------------------------------------------------

    def predict(self, inputs):
        req = self._parse(inputs)
        user_id = req["user_id"]
        date = req.get("date")
        activity_id = req.get("activity_id")
        request_parameters = req.get("request_parameters")

        readiness_cls, readiness_proba = "yellow", None
        stress_level, stress_score = "normal", None
        recovery_hours = None
        try:
            readiness_cls, readiness_proba = self._readiness(user_id, date)
        except Exception as exc:
            print(f"readiness failed: {exc}")
        try:
            stress_level, stress_score = self._stress(user_id, request_parameters)
        except Exception as exc:
            print(f"stress failed: {exc}")
        try:
            recovery_hours = self._recovery(user_id, activity_id)
        except Exception as exc:
            print(f"recovery failed: {exc}")

        overall = self._combine(readiness_cls, stress_level, recovery_hours)
        return {
            "predictions": {
                "overall_status": overall,
                "readiness": {"class": readiness_cls, "confidence": readiness_proba},
                "stress": {"level": stress_level, "score": stress_score},
                "recovery": {"predicted_recovery_hours_remaining": recovery_hours},
            }
        }
