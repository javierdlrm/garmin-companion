"""Optional KServe transformer for the Garmin deployment.

The chosen topology (DESIGN_DECISIONS A8) puts the feature-store lookup and the
decision layer inside the predictor, so a transformer is **not required**. It is
provided for the alternative pattern where you want a separate transformer pod to:

* enrich the request with on-demand realtime features (e.g. compute
  ``stress_last_30m`` from a payload of recent epochs and forward them as
  ``request_parameters``), and/or
* shape the human-facing response (``preprocess`` / ``postprocess``).

Attach via ``model.deploy(..., transformer=Transformer(script_file="deployments/transformer.py"))``.
"""


class Transformer(object):
    def __init__(self):
        # A transformer typically holds no model; it only reshapes I/O.
        print("Garmin transformer initialised")

    def preprocess(self, inputs):
        """Pass the request through; attach request_parameters if recent epochs given.

        Expected request, e.g.::
            {"instances": [{"user_id": "javier",
                             "recent_epochs": [{"stress_level": 40, "ts": "..."}, ...]}]}
        """
        instances = inputs.get("instances", inputs)
        out = []
        for inst in instances:
            if isinstance(inst, dict) and "recent_epochs" in inst:
                epochs = inst.pop("recent_epochs")
                stress_vals = [e.get("stress_level") for e in epochs if e.get("stress_level") is not None]
                inst.setdefault("request_parameters", {})
                if stress_vals:
                    inst["request_parameters"]["stress_last_2h"] = sum(stress_vals) / len(stress_vals)
            out.append(inst)
        return {"instances": out}

    def postprocess(self, outputs):
        """Return predictions unchanged (predictor already shapes the response)."""
        return outputs
